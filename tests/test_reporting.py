"""Analytic checks of aggregation and output, independent of game dynamics."""
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
import xml.etree.ElementTree as ET

from astronaut_rl import plots, poster, report


class ReportingTests(unittest.TestCase):
    def fixture(self, root):
        results = root / "results"
        results.mkdir()
        records = []
        for variant, capacity in (("fly", 72), ("asteroids", 2880), ("oxygen", 103680)):
            for index, seed in enumerate((11, 23, 37, 51, 79)):
                frames = 100 + 100 * index
                budget = 200000 if variant == "fly" else 1000000
                summary = {"episodes": 1 if variant == "fly" else 100,
                           "return_mean": -.5 + index * .1, "frames_mean": frames,
                           "survival_rate": .1 * index,
                           "oxygen_mean": index * .5 if variant == "oxygen" else 0,
                           "oxygen_per_1000_frames": index if variant == "oxygen" else 0,
                           "combined_success_rate": index * .02 if variant == "oxygen" else 0,
                           "collision_ground_rate": .2, "collision_ceiling_rate": .1,
                           "collision_asteroid_rate": .3 if variant != "fly" else 0,
                           "unseen_fraction": .01 * index}
                baseline = {**summary, "frames_mean": 100, "survival_rate": 0,
                            "oxygen_mean": 0, "oxygen_per_1000_frames": 0,
                            "combined_success_rate": 0}
                history = [{"step": step, **summary} for step in (0, budget)]
                record = {"variant": variant, "seed": seed, "profile": "official",
                          "budget": budget, "budget_unit": "decision_updates", "repeat": 6, "alpha": .08, "gamma": .9995**6, "gamma_per_frame": .9995,
                          "state_capacity": capacity, "selected_step": 50000,
                          "train_seconds": (index + 1) * 10, "frames": 100000,
                          "states": 10 + index, "greedy": summary, "random": baseline,
                          "coverage": {"visited_states": 10 + index, "total_updates": 100000,
                                       "visited_pairs_lt10_fraction": .2 + index * .01},
                          "first_target_step": None if index < 2 else 50000,
                          "stability_step": None, "qualified": True, "initialized_rows": 0 if variant == "fly" else 100,
                          "stage_cost": {"decision_updates": 100000, "physics_frames": 600000, "train_seconds": (index + 1)*10},
                          "cumulative_cost": {"decision_updates": 100000*(1+(("fly", "asteroids", "oxygen").index(variant))), "physics_frames": 600000, "train_seconds": (index + 1)*10*(1+(("fly", "asteroids", "oxygen").index(variant)))},
                          "config": {"epsilon_start": 1,
                          "epsilon_end": .05, "epsilon_decay_fraction": .7},
                          "environment_commit": "0db0652ad3b8cece05d476c56637a3dcb56557cd",
                          "validation_history": history}
                records.append(record)
                with (results / ("%s_seed%s_training.csv" % (variant, seed))).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("step", "td_mse", "visited_states"))
                    writer.writeheader()
                    for step in (2000, budget):
                        writer.writerow({"step": step, "td_mse": .01 * (index + 1), "visited_states": 10 + index})
        manifest = {"records": records, "python": "3.12.14", "machine": "arm64"}
        (results / "run_manifest.json").write_text(json.dumps(manifest))
        return results, manifest

    def test_sample_sd_uses_independent_policy_means(self):
        with tempfile.TemporaryDirectory() as directory:
            results, manifest = self.fixture(Path(directory))
            # Episode counts do not weight the independent training-policy mean.
            for index, row in enumerate(manifest["records"][:5]):
                row["greedy"]["episodes"] = index + 1
            (results / "run_manifest.json").write_text(json.dumps(manifest))
            aggregate = report.aggregate(results)
            metric = aggregate["variants"]["fly"]["greedy"]["frames_mean"]
            self.assertEqual(metric["n"], 5)
            self.assertEqual(aggregate["variants"]["fly"]["budget"]["mean"], 200000)
            self.assertEqual(aggregate["variants"]["oxygen"]["budget"]["mean"], 1000000)
            self.assertEqual(aggregate["variants"]["oxygen"]["cumulative_cost"]["train_seconds"]["mean"], 90)
            self.assertEqual(aggregate["variants"]["oxygen"]["stage_cost"]["train_seconds"]["mean"], 30)
            self.assertEqual(metric["mean"], 300)
            self.assertAlmostEqual(metric["sd"], statistics.stdev((100, 200, 300, 400, 500)))
            self.assertEqual(aggregate["variants"]["fly"]["random"]["frames_mean"]["sd"], 0)
            target = aggregate["variants"]["fly"]["first_target_step"]
            self.assertEqual(target["reached_runs"], 3)
            self.assertEqual(target["steps_among_reached"]["mean"], 50000)
            self.assertIsNone(aggregate["variants"]["fly"]["stability_step"]["steps_among_reached"])

    def test_missing_checkpoint_does_not_silently_change_band_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            results, manifest = self.fixture(Path(directory))
            manifest["records"][0]["validation_history"].pop()
            (results / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Unmatched validation"):
                report.aggregate(results)

    def test_primary_report_rejects_shaped_legacy_records(self):
        with tempfile.TemporaryDirectory() as directory:
            results, manifest = self.fixture(Path(directory))
            manifest["records"][0]["profile"] = "oxygen_v2"
            (results / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "official reward"):
                report.aggregate(results)

    def test_outputs_use_data_and_valid_standard_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, manifest = self.fixture(root)
            summary_path, markdown_path = report.build(results, root / "RESULTS.md")
            self.assertEqual(json.loads(summary_path.read_text())["variants"]["oxygen"]["state_capacity"], 103680)
            self.assertIn("selected", markdown_path.read_text())
            table_width = None
            for line in markdown_path.read_text().splitlines():
                if line.startswith("|"):
                    if table_width is None: table_width = line.count("|")
                    self.assertEqual(line.count("|"), table_width, line)
                else:
                    table_width = None
            paths = plots.build_all(results, root / "figures")
            self.assertEqual(len(paths), 7)
            for path in paths:
                tree = ET.parse(path)
                self.assertEqual(tree.getroot().tag, "{http://www.w3.org/2000/svg}svg")
                self.assertIn("sample SD", path.read_text())
                self.assertNotIn("minimum-maximum", path.read_text())
            self.assertIn("Q decisions", (root / "figures/loss.svg").read_text())
            pdf = poster.build(results, root / "poster.pdf")
            content = pdf.read_bytes()
            self.assertTrue(content.startswith(b"%PDF-1.4"))
            self.assertTrue(content.endswith(b"%%EOF\n"))
            self.assertIn(b"No shaping.", content)
            self.assertIn(b"103,680 states", content)

    def test_nonfinite_metrics_cannot_be_published(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                report.describe((1, invalid))


if __name__ == "__main__":
    unittest.main()
