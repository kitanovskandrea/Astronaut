"""Cheap complete fixtures and deliberate inconsistencies for the verifier."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from astronaut_rl.experiment import train
from astronaut_rl.qlearning import QLearner
from astronaut_rl.verify import checked_records, model_path, verify


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.base = Path(cls.fixture.name)
        with contextlib.redirect_stdout(io.StringIO()):
            record = train("fly", 11, 2000, cls.base / "results", cls.base / "models",
                           checkpoint_interval=1000)
        manifest = {"complete": True, "training_seeds": [11], "variants": ["fly"],
                    "curriculum_passed": record["qualified"], "records": [record],
                    "failed_stages": [] if record["qualified"] else [{"seed": 11, "variant": "fly"}]}
        (cls.base / "results" / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results, self.models = self.root / "results", self.root / "models"
        shutil.copytree(self.base / "results", self.results)
        shutil.copytree(self.base / "models", self.models)
        self.manifest_path = self.results / "run_manifest.json"
        self.original = json.loads(self.manifest_path.read_text())

    def publish(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return verify(self.results, self.models)

    def test_valid_fixture_replays_test_and_both_validation_checkpoints(self):
        self.assertEqual(self.check(), {"episodes": 5, "checkpoints": 3, "models": 1})

    def test_empty_duplicate_and_missing_training_seed_records_are_rejected(self):
        empty = copy.deepcopy(self.original)
        empty["records"] = []
        duplicate = copy.deepcopy(self.original)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        missing = copy.deepcopy(self.original)
        missing["training_seeds"].append(23)
        for manifest, message in ((empty, "no completed"), (duplicate, "Duplicate"),
                                  (missing, "training seed set")):
            with self.subTest(message=message):
                self.publish(manifest)
                with self.assertRaisesRegex(ValueError, message):
                    self.check()

    def test_exact_checkpoint_schedule_rejects_missing_duplicate_and_reordered_rows(self):
        original = self.original["records"][0]["validation_history"]
        variants = (original[1:], [original[0], original[0], original[1]], list(reversed(original)))
        for history in variants:
            with self.subTest(steps=[row["step"] for row in history]):
                manifest = copy.deepcopy(self.original)
                manifest["records"][0]["validation_history"] = copy.deepcopy(history)
                self.publish(manifest)
                with self.assertRaisesRegex(ValueError, "checkpoint_schedule"):
                    self.check()

    def test_selected_flag_must_identify_actual_selected_step(self):
        manifest = copy.deepcopy(self.original)
        history = manifest["records"][0]["validation_history"]
        wrong = next(row["step"] for row in history if not row["selected"])
        for row in history:
            row["selected"] = int(row["step"] == wrong)
        self.publish(manifest)
        # There is still exactly one flag, but it now denotes the wrong table.
        with self.assertRaisesRegex(ValueError, "selected_flag"):
            self.check()

    def test_duplicate_selected_and_final_summaries_must_agree(self):
        mutations = ((["selected", "step"], 999),
                     (["selected", "validation", "frames_mean"], -1),
                     (["selected", "greedy", "survival_rate"], -1),
                     (["final", "validation", "oxygen_mean"], -1),
                     (["final", "step"], 999),
                     (["final", "model_path"], "another.json.gz"))
        for keys, value in mutations:
            with self.subTest(keys=keys):
                manifest = copy.deepcopy(self.original)
                cursor = manifest["records"][0]
                for key in keys[:-1]:
                    cursor = cursor[key]
                cursor[keys[-1]] = value
                self.publish(manifest)
                with self.assertRaises(ValueError):
                    self.check()

    def test_actual_serialized_agent_parameters_must_match_metadata(self):
        relative = self.original["records"][0]["model_path"]
        path = self.models / relative
        original_bytes = path.read_bytes()
        for attribute, value in (("gamma", 0.5), ("alpha", 0.5), ("seed", 99), ("actions", 3)):
            with self.subTest(attribute=attribute):
                path.write_bytes(original_bytes)
                agent = QLearner.load(path)
                setattr(agent, attribute, value)
                if attribute == "actions":
                    for row in agent.q.values():
                        row.append(0.0)
                    for row in agent.visits.values():
                        row.append(0)
                agent.save(path)
                manifest = copy.deepcopy(self.original)
                manifest["records"][0]["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.publish(manifest)
                # Recomputing the checksum cannot hide a disagreement between
                # the actual agent values and the declared frozen experiment.
                with self.assertRaisesRegex(ValueError, "actual_" + attribute):
                    self.check()

    def test_record_hyperparameters_and_protocol_cannot_disagree_with_config(self):
        for change in ("gamma", "test_seeds"):
            with self.subTest(change=change):
                manifest = copy.deepcopy(self.original)
                record = manifest["records"][0]
                if change == "gamma":
                    record["gamma"] = 0.5
                else:
                    record["protocol"]["test_seeds"] = [9999]
                self.publish(manifest)
                with self.assertRaises(ValueError):
                    self.check()

    def test_model_paths_reject_absolute_and_parent_escape(self):
        for relative in (str((self.models / "fly_seed11.json.gz").resolve()), "../outside.json.gz"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                model_path(self.models, relative)

    def test_unbacked_validation_episode_group_is_rejected(self):
        path = self.results / "fly_seed11_validation_episodes.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames, rows = reader.fieldnames, list(reader)
        extra = dict(rows[0])
        extra["checkpoint_step"] = "9999"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows + [extra])
        with self.assertRaisesRegex(ValueError, "episode groups"):
            self.check()

    def test_dependent_stage_cannot_claim_progress_after_failed_flight_gate(self):
        manifest = copy.deepcopy(self.original)
        manifest["variants"] = ["fly", "asteroids"]
        manifest["records"][0]["qualified"] = False
        child = copy.deepcopy(manifest["records"][0])
        child["variant"] = "asteroids"
        manifest["records"].append(child)
        with self.assertRaisesRegex(ValueError, "advanced after a failed"):
            checked_records(manifest)


if __name__ == "__main__":
    unittest.main()
