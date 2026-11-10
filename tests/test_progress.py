"""Progress, checkpoint selection, isolated timing and worker integration."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from astronaut_rl import experiment
from astronaut_rl.config import config_hash, load_config
from astronaut_rl.environment import Transition
from astronaut_rl.qlearning import QLearner


class ProgressTests(unittest.TestCase):
    def test_atomic_snapshot_creates_parent_and_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "progress.json"
            experiment._write_progress(path, {"phase": "training", "decision": 0})
            experiment._write_progress(path, {"phase": "complete", "decision": 2000})
            self.assertEqual(json.loads(path.read_text()),
                             {"phase": "complete", "decision": 2000})
            self.assertFalse(path.with_name("progress.json.tmp").exists())

    def test_training_snapshots_and_model_match_run_without_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "viewer" / "progress.json"
            snapshots = []
            write = experiment._write_progress

            def capture(snapshot_path, payload):
                write(snapshot_path, payload)
                snapshots.append(json.loads(Path(snapshot_path).read_text()))

            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(experiment, "_write_progress", side_effect=capture):
                    reported = experiment.train("fly", 11, 2000, root / "results",
                                                root / "models", progress_path=path)
                plain = experiment.train("fly", 11, 2000, root / "plain_results",
                                         root / "plain_models")
            self.assertEqual([s["phase"] for s in snapshots],
                             ["training", "validating", "training", "training", "validating", "evaluating", "failed_gate"])
            first, initial_validation, resumed, block, validation, evaluation, complete = snapshots
            self.assertEqual(first["decision"], 0)
            self.assertEqual(first["epsilon"], load_config("fly")["epsilon_start"])
            self.assertIsNone(first["td_mse"])
            self.assertEqual(block["decision"], 2000)
            self.assertEqual(block["td_mse"], reported["last_td_mse"])
            self.assertEqual(validation["evaluation_kind"], "validation")
            self.assertEqual(evaluation["evaluation_kind"], "test")
            self.assertGreater(complete["frames"], 2000)
            self.assertEqual(complete["budget"], 2000)
            self.assertEqual(complete["variant"], "fly")
            self.assertEqual(complete["seed"], 11)
            self.assertEqual(complete["greedy"], reported["greedy"])
            learner = QLearner.load(complete["model_path"])
            self.assertEqual(learner.metadata["budget"], 2000)
            self.assertEqual(learner.metadata["selected_step"], reported["selected_step"])
            self.assertEqual(config_hash(learner.metadata["config"]), learner.metadata["config_hash"])
            self.assertTrue((root / "results" / "fly_seed11_summary.json").is_file())
            for key in ("frames", "episodes", "states", "first_td_mse", "last_td_mse",
                        "greedy", "random", "coverage", "selected_step", "model_sha256",
                        "final_model_sha256"):
                self.assertEqual(reported[key], plain[key], key)

    def test_cli_progress_rejects_multiple_jobs(self):
        for argv in (["experiment", "--progress", "progress.json"],
                     ["experiment", "--progress", "progress.json", "--variant", "fly",
                      "--seeds", "11", "23"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as error:
                    experiment.main()
                self.assertEqual(error.exception.code, 2)

    def test_short_worker_keeps_artifacts_but_cannot_falsely_advance_a_gate(self):
        from astronaut_rl.training_job import TrainingJob
        configuration = load_config("fly")
        with tempfile.TemporaryDirectory() as directory:
            job = TrainingJob("fly", 11, budget=2000, run_parent=Path(directory),
                              profile="official")
            run_directory = job.directory
            try:
                deadline = time.monotonic() + 20
                while True:
                    progress = job.snapshot()
                    if job.process.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("Official worker did not finish 2,000 updates")
                    time.sleep(0.05)
                self.assertEqual(job.process.returncode, 1, job.log_path.read_text())
                self.assertEqual(progress["phase"], "blocked")
                self.assertIn("gate", progress["error"].lower())
                self.assertEqual((progress["decision"], progress["budget"]), (2000, 2000))
                self.assertEqual(Path(progress["model_path"]), job.model_path.resolve())
                learner = QLearner.load(job.model_path)
                self.assertEqual(learner.metadata["profile"], "official")
                self.assertEqual(learner.alpha, configuration["alpha"])
                self.assertEqual(learner.gamma, configuration["gamma"])
                self.assertEqual(learner.metadata["epsilon_start"], configuration["epsilon_start"])
                self.assertEqual(learner.metadata["state_spec"], configuration["state_spec"])
                self.assertEqual(learner.metadata["budget"], 2000)
                self.assertFalse(learner.metadata["qualified"])
                final = QLearner.load(job.model_path.with_name("fly_seed11_final.json.gz"))
                self.assertEqual(sum(sum(row) for row in final.visits.values()), 2000)
            finally:
                job.stop()
                self.assertIsNotNone(job.process.poll())
        self.assertFalse(run_directory.exists())

    def test_validation_frequency_does_not_change_final_learning_trajectory(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            frequent = experiment.train("fly", 11, 2000, root / "a_results", root / "a_models",
                                        checkpoint_interval=500)
            sparse = experiment.train("fly", 11, 2000, root / "b_results", root / "b_models",
                                      checkpoint_interval=2000)
            first = QLearner.load(root / "a_models" / frequent["final_model_path"])
            second = QLearner.load(root / "b_models" / sparse["final_model_path"])
            self.assertEqual(first.q, second.q)
            self.assertEqual(first.visits, second.visits)
            self.assertEqual(sum(sum(row) for row in first.visits.values()), 2000)
            self.assertEqual(frequent["frames"], sparse["frames"])
            self.assertEqual(frequent["episodes"], sparse["episodes"])

    def test_checkpoint_selection_timing_and_budget_interruption(self):
        clock = [0.0]
        resets, evaluations = [], []

        class FakeGame:
            def __init__(self, *args, **kwargs):
                self.total, self.episode_frame = 0, 0
                self.repeat = kwargs.get("repeat", 6)

            def reset(self, seed=None):
                resets.append(seed)
                self.episode_frame = 0
                return (0, 0)

            def step(self, action, on_frame):
                clock[0] += 1
                self.total += 1
                self.episode_frame += 1
                terminal = self.episode_frame == 3
                info = {"steps": self.episode_frame,
                        "collision": "ground" if terminal else "no", "oxygen_count": 0}
                for k in range(1, self.repeat + 1):
                    on_frame((), -1.0 if terminal and k == self.repeat else 0.0,
                             terminal and k == self.repeat, False, info, k)
                return Transition((self.total % 5, 0), -1.0 if terminal else 0.0,
                                  terminal, False, self.repeat, info)

            def close(self):
                pass

        def fake_evaluate(agent, variant, seeds, repeat=1, random_policy=False, **kwargs):
            clock[0] += 100
            evaluations.append((tuple(seeds), random_policy,
                                sum(sum(row) for row in agent.visits.values())))
            trained = sum(sum(row) for row in agent.visits.values()) > 0
            return [{"seed": seeds[0], "return": 0.0 if trained else -1.0,
                     "frames": 1200 if trained else 72, "decisions": 200 if trained else 12,
                     "survived": int(trained), "oxygen": 0, "collision": "no" if trained else "ground", "unseen_fraction": 0.0}]

        save, write = QLearner.save, experiment._write_progress

        def slow_save(agent, path, metadata=None):
            clock[0] += 10
            return save(agent, path, metadata=metadata)

        def slow_progress(path, payload):
            clock[0] += 5
            return write(path, payload)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            with mock.patch.object(experiment, "Game", FakeGame), \
                 mock.patch.object(experiment, "evaluate", side_effect=fake_evaluate), \
                 mock.patch.object(experiment.time, "perf_counter", side_effect=lambda: clock[0]), \
                 mock.patch.object(QLearner, "save", slow_save), \
                 mock.patch.object(experiment, "_write_progress", side_effect=slow_progress):
                record = experiment.train("fly", 11, 10, root / "results", root / "models",
                    checkpoint_interval=5, progress_path=root / "progress.json")
            self.assertEqual(record["train_seconds"], 10)
            self.assertEqual(record["validation_seconds"], 300)
            self.assertEqual(record["test_seconds"], 200)
            self.assertEqual(record["export_seconds"], 50)
            self.assertEqual(record["selected_step"], 5)  # CP0 fails; earliest passing tie.
            self.assertEqual(record["coverage"]["total_updates"], 10)
            self.assertEqual(record["selected"]["coverage"]["total_updates"], 5)
            self.assertEqual(resets, [11, None, None, None])
            validation, testing = load_config("fly")["validation_seeds"], load_config("fly")["test_seeds"]
            self.assertEqual(evaluations, [(tuple(validation), False, 0), (tuple(validation), False, 5), (tuple(validation), False, 10),
                                          (tuple(testing), False, 5), (tuple(testing), True, 5)])
            selected = QLearner.load(root / "models" / record["model_path"])
            final = QLearner.load(root / "models" / record["final_model_path"])
            self.assertEqual(sum(sum(row) for row in selected.visits.values()), 5)
            self.assertEqual(sum(sum(row) for row in final.visits.values()), 10)
            for row in record["validation_history"]:
                self.assertFalse(Path(row["model_identifier"]).is_absolute())
                self.assertTrue((root / "models" / row["model_identifier"]).is_file())
            with (root / "results" / "fly_seed11_episodes.csv").open(newline="") as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual(record["episodes"], 3)
            self.assertEqual([row["completed"] for row in episodes], ["1", "1", "1", "0"])
            self.assertEqual(episodes[-1]["budget_interrupted"], "1")
            self.assertEqual(episodes[-1]["collision"], "no")
            self.assertEqual(episodes[-1]["frames"], "6")
            self.assertEqual(episodes[-1]["decisions"], "1")
            self.assertEqual(record["frames"], 60)
            self.assertTrue(record["qualified"])

    def test_coverage_uses_visited_pairs_as_diagnostic_denominator(self):
        agent = QLearner()
        agent.visits = {(0,): [0, 2], (1,): [10, 100]}
        result = experiment.coverage_summary(agent, 10)
        self.assertEqual(result["visited_states"], 2)
        self.assertEqual(result["visited_state_action_pairs"], 3)
        self.assertEqual(result["zero_visit_actions_in_visited_states"], 1)
        self.assertEqual(result["total_updates"], 112)
        self.assertAlmostEqual(result["state_coverage_fraction"], 0.2)
        self.assertAlmostEqual(result["visited_pairs_lt10_fraction"], 1 / 3)
        self.assertAlmostEqual(result["visited_pairs_ge100_fraction"], 1 / 3)

    def test_cli_retains_failed_stage_and_never_starts_its_dependents(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            failed = {"variant": "fly", "seed": 11, "qualified": False, "gate": {"qualified": False}}
            argv = ["experiment", "--curriculum-to", "oxygen", "--seeds", "11", "--steps", "2000",
                    "--out", str(root / "results"), "--models", str(root / "models")]
            with mock.patch("sys.argv", argv), mock.patch.object(experiment, "train", return_value=failed) as run:
                with self.assertRaises(SystemExit) as error:
                    experiment.main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], "fly")
            manifest = json.loads((root / "results" / "run_manifest.json").read_text())
            self.assertEqual(manifest["records"], [failed])
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["curriculum_passed"])
            self.assertEqual(manifest["failed_stages"][0]["skipped_dependents"], ["asteroids", "oxygen"])

    def test_custom_config_cannot_misstate_source_or_state_capacity(self):
        for key, value in (("environment_commit", "another-commit"), ("state_capacity", 999)):
            configuration = load_config("fly")
            configuration[key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    experiment.train("fly", 11, 10, root / "results", root / "models", config=configuration)


if __name__ == "__main__":
    unittest.main()
