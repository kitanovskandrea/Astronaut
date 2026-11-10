"""Aggregate run-level metrics; independent training policies are the replicates.

All statistics use Python's standard library. Sample SD is calculated across
run means, never by treating test episodes as independent training runs.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ORDER = ("fly", "asteroids", "oxygen")
LABELS = {"fly": "Flight", "asteroids": "+ Asteroids", "oxygen": "+ Oxygen"}
TEST_METRICS = ("return_mean", "frames_mean", "survival_rate", "oxygen_mean",
                "oxygen_per_1000_frames", "combined_success_rate", "unseen_fraction",
                "collision_ground_rate", "collision_ceiling_rate", "collision_asteroid_rate")
CURVE_METRICS = ("frames_mean", "survival_rate", "oxygen_mean", "return_mean",
                 "oxygen_per_1000_frames", "combined_success_rate")


def finite(value, name="value"):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % name)
    return number


def describe(values):
    """Mean and sample SD, with a zero-width descriptive SD for n=1."""
    values = [finite(value) for value in values]
    if not values:
        raise ValueError("Cannot aggregate an empty sample")
    return {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values), "min": min(values), "max": max(values)}


def groups_for(records):
    groups = {}
    for record in records:
        variant = record["variant"]
        if variant not in ORDER:
            raise ValueError("Unknown variant: %s" % variant)
        groups.setdefault(variant, []).append(record)
    if not groups:
        raise ValueError("Manifest has no runs")
    for variant, rows in groups.items():
        seeds = [row["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate training seed for %s" % variant)
    return {variant: groups[variant] for variant in ORDER if variant in groups}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _history(results, record):
    if "validation_history" in record:
        return record["validation_history"]
    return read_csv(Path(results) / ("%s_seed%s_validation.csv" % (record["variant"], record["seed"])))


def validation_curves(results, groups):
    """Require complete matched checkpoints; never silently vary band sample size."""
    curves = {}
    for variant, records in groups.items():
        histories = []
        for record in records:
            history = _history(results, record)
            by_step = {}
            for row in history:
                step = int(row["step"])
                if step in by_step:
                    raise ValueError("Duplicate validation step")
                by_step[step] = row
            if not by_step:
                raise ValueError("No validation checkpoints for %s" % variant)
            histories.append(by_step)
        expected = set(histories[0])
        if any(set(history) != expected for history in histories):
            raise ValueError("Unmatched validation checkpoints for %s" % variant)
        curves[variant] = []
        for step in sorted(expected):
            point = {"step": step, "n": len(records)}
            for metric in CURVE_METRICS:
                if all(metric in history[step] for history in histories):
                    point[metric] = describe([history[step][metric] for history in histories])
            curves[variant].append(point)
    return curves


def training_curves(results, groups, metric="td_mse"):
    curves = {}
    for variant, records in groups.items():
        histories = []
        for record in records:
            rows = read_csv(Path(results) / ("%s_seed%s_training.csv" % (variant, record["seed"])))
            by_step = {int(row["step"]): finite(row[metric], metric) for row in rows}
            if len(by_step) != len(rows) or not rows:
                raise ValueError("Empty or duplicate training blocks for %s" % variant)
            histories.append(by_step)
        expected = set(histories[0])
        if any(set(history) != expected for history in histories):
            raise ValueError("Unmatched training blocks for %s" % variant)
        curves[variant] = [{"step": step, **describe([history[step] for history in histories])}
                           for step in sorted(expected)]
    return curves


def _optional_description(records, key):
    values = [row[key] for row in records if row.get(key) is not None]
    return describe(values) if values else None


def aggregate(results):
    results = Path(results)
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    groups = groups_for(manifest["records"])
    if any(row.get("budget_unit", manifest.get("budget_unit")) != "decision_updates" or int(row["repeat"]) != 6 for row in manifest["records"]):
        raise ValueError("Curriculum report requires decision-update budgets and six-frame actions; archive earlier reports separately")
    variants = {}
    for variant, rows in groups.items():
        profile = {row.get("profile") for row in rows}
        if profile != {"official"}:
            raise ValueError("Primary report requires only the official reward profile")
        capacities = {int(row["state_capacity"]) for row in rows}
        if len(capacities) != 1:
            raise ValueError("State capacity differs within %s" % variant)
        entry = {"n": len(rows), "training_seeds": [row["seed"] for row in rows],
                 "state_capacity": capacities.pop(),
                 "budget": describe([row["budget"] for row in rows]),
                 "train_seconds": describe([row["train_seconds"] for row in rows]),
                 "frames": describe([row["frames"] for row in rows]),
                 "states": describe([row["states"] for row in rows]),
                 "selected_step": describe([row["selected_step"] for row in rows]),
                 "greedy": {}, "random": {}}
        for policy in ("greedy", "random"):
            for metric in TEST_METRICS:
                if all(metric in row[policy] for row in rows):
                    entry[policy][metric] = describe([row[policy][metric] for row in rows])
            entry[policy]["episodes_per_policy"] = [int(row[policy]["episodes"]) for row in rows]
        coverage_keys = set.intersection(*(set(row.get("coverage", {})) for row in rows))
        entry["final_coverage"] = {
            key: describe([row["coverage"][key] for row in rows])
            for key in sorted(coverage_keys)
            if all(isinstance(row["coverage"][key], (int, float)) for row in rows)}
        entry["initialized_rows"] = describe([row.get("initialized_rows", row.get("transferred_rows", 0)) for row in rows])
        entry["qualified_runs"] = sum(bool(row.get("qualified", row.get("gate", {}).get("qualified", False))) for row in rows)
        entry["stage_cost"] = {
            key: describe([row.get("stage_cost", {}).get(key, row[fallback]) for row in rows])
            for key, fallback in (("decision_updates", "budget"), ("physics_frames", "frames"), ("train_seconds", "train_seconds"))}
        entry["cumulative_cost"] = {
            key: describe([row.get("cumulative_cost", {}).get(key, row.get("cumulative", {}).get(key, row.get("stage_cost", {}).get(key, row[fallback]))) for row in rows])
            for key, fallback in (("decision_updates", "budget"), ("physics_frames", "frames"), ("train_seconds", "train_seconds"))}
        gate_keys = ("minimum_survival_rate", "minimum_oxygen_mean", "baseline_survival_rate", "baseline_oxygen_mean")
        entry["gate"] = {key: describe([row["gate"][key] for row in rows]) for key in gate_keys
                         if all(key in row.get("gate", {}) for row in rows)}
        for key in ("first_target_step", "stability_step"):
            entry[key] = {"reached_runs": sum(row.get(key) is not None for row in rows),
                          "total_runs": len(rows), "steps_among_reached": _optional_description(rows, key)}
        variants[variant] = entry
    return {"schema": "astronaut-aggregate-v2", "budget_unit": manifest.get("budget_unit", manifest["records"][0].get("budget_unit", "decision_updates")), "replicate": "independent curriculum chain",
            "dispersion": "sample standard deviation (n-1 denominator)",
            "environment_commit": manifest["records"][0].get("environment_commit"),
            "variants": variants, "validation_curves": validation_curves(results, groups),
            "limitations": [
                "Safety-constrained checkpoint selection uses validation only; final test metrics do not select a checkpoint.",
                "Each policy is averaged over its test episodes before aggregation across training seeds.",
                "Flight is deterministic from its default reset; one test episode per policy is sufficient.",
                "SD bands describe chain variation; they are not confidence intervals.",
                "The same random baseline seeds do not create independent baseline replicates.",
                "Low TD error under zero rewards does not establish good play or convergence.",
                "Training time excludes validation and file export; it depends on machine and load.",
            ]}


def formatted(stat, digits=2, scale=1.0):
    return ("{:.%df} +/- {:.%df}" % (digits, digits)).format(stat["mean"] * scale, stat["sd"] * scale)


def markdown(summary, manifest):
    rows = ["# Measured results", "", "All values below use the official course rewards: -1 for collision, +1 for",
            "an oxygen pickup without collision, and 0 otherwise. No reward shaping is",
            "used. Means +/- sample SD are across independent training policies, after",
            "averaging the episodes for each policy. Curriculum chains are independent;",
            "stages inside a chain inherit learned values. The checkpoint for each run is",
            "selected using validation only; the selected policy is then frozen for testing.", "",
            "## Selected-policy test performance", "",
            "| Variant | Policies | Test episodes / policy | Frames | Full survival (%) | Tanks / episode | Return |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for variant, entry in summary["variants"].items():
        g = entry["greedy"]
        counts = sorted(set(g["episodes_per_policy"]))
        rows.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            LABELS[variant], entry["n"], ", ".join(map(str, counts)), formatted(g["frames_mean"], 1),
            formatted(g["survival_rate"], 1, 100), formatted(g["oxygen_mean"]), formatted(g["return_mean"])))
    rows += ["", "Flight has no stochastic reset/spawn variations: its episode seeds do not",
             "create different scenarios. Training-seed variability still measures the",
             "independently explored policies listed in the table.", "", "## Random baseline", "",
             "| Variant | Frames | Full survival (%) | Tanks / episode |", "| --- | ---: | ---: | ---: |"]
    for variant, entry in summary["variants"].items():
        g = entry["random"]
        rows.append("| %s | %s | %s | %s |" % (LABELS[variant], formatted(g["frames_mean"], 1),
                    formatted(g["survival_rate"], 1, 100), formatted(g["oxygen_mean"])))
    rows += ["", "The random policy uses the same test seeds. Its repeated evaluations across",
             "training-seed records are the same baseline, not additional independent samples.", "",
             "## Allocated stage cost, cumulative cost and selected checkpoints", "",
             "| Variant | Q decisions | Physics steps | Incremental seconds | Cumulative seconds | Selected Q decision | Qualified chains |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant, entry in summary["variants"].items():
        rows.append("| %s | %s | %s | %s | %s | %s | %s/%s |" % (LABELS[variant],
                    formatted(entry["budget"], 0), formatted(entry["frames"], 0),
                    formatted(entry["stage_cost"]["train_seconds"]), formatted(entry["cumulative_cost"]["train_seconds"]),
                    formatted(entry["selected_step"], 0), entry["qualified_runs"], entry["n"]))
    rows += ["", "Each completed stage continues to its allocation even if an earlier checkpoint is selected.",
             "Flight and later stages have different Q-decision budgets; decisions execute up",
             "to six actual physics frames. Cumulative costs include full prerequisite training",
             "within each chain. This is not an equal-budget, independent-from-zero comparison.",
             "A stage that fails its behavioral gate is recorded as unqualified and is not",
             "promoted to the next stage. The main model file holds the selected policy; the separate final",
             "checkpoint records the end-of-budget learner. The timer covers training",
             "interaction and Q updates, excluding checkpoint validation, model export and rendering.", "",
             "Timing protocol: " + manifest.get("timing_protocol", "recorded in the experiment manifest") + ".", "",
             "## Oxygen and collision diagnostics", "",
             "| Variant | Tanks / 1,000 steps | Complete with >=3 tanks (%) | Ground (%) | Ceiling (%) | Asteroid (%) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant, entry in summary["variants"].items():
        g = entry["greedy"]
        metric_values = []
        for key, scale in (("oxygen_per_1000_frames", 1), ("combined_success_rate", 100),
                           ("collision_ground_rate", 100), ("collision_ceiling_rate", 100),
                           ("collision_asteroid_rate", 100)):
            metric_values.append(formatted(g[key], 2, scale) if key in g else "not recorded")
        rows.append("| %s | %s |" % (LABELS[variant], " | ".join(metric_values)))
    rows += ["", "Tank rate is calculated separately for each policy as 1,000 times total tanks",
             "divided by total survived physics steps, then aggregated across training seeds.",
             "Collision percentages describe outcomes; longer survival can expose a policy",
             "to more asteroids, so they must be read alongside frames and completion.", "",
             "## Coverage and sample efficiency", ""]
    rows += ["| Variant | Inherited rows | Stored rows | Updated states | Updated pairs | Updated pairs with <10 updates (%) | Absent test rows (%) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant, entry in summary["variants"].items():
        coverage = entry["final_coverage"]
        def cell(key, digits=1, scale=1):
            return formatted(coverage[key], digits, scale) if key in coverage else "not recorded"
        unseen = entry["greedy"].get("unseen_fraction")
        rows.append("| %s | %s | %s | %s | %s | %s | %s |" % (LABELS[variant], formatted(entry["initialized_rows"], 0), formatted(entry["states"], 0), cell("visited_states"),
                    cell("visited_state_action_pairs"), cell("visited_pairs_lt10_fraction", 1, 100),
                    formatted(unseen, 2, 100) if unseen else "not recorded"))
    rows += ["", "Coverage columns count actual child-stage updates, not copied parent values.",
             "Stored inherited rows can have zero child visits. Unseen test decisions",
             "refer to absent rows in the selected table; a present inherited row need not have",
             "received any child update. The complete",
             "coverage statistics for both tables are in the run records; `aggregate.json`",
             "contains means/SD for all numeric full-budget coverage fields.", ""]
    for variant, entry in summary["variants"].items():
        rows.append("- **%s**" % LABELS[variant])
        for key, label in (("first_target_step", "First target checkpoint"), ("stability_step", "Consecutive gate stability")):
            target = entry[key]
            step = target["steps_among_reached"]
            text = formatted(step, 0) + " Q decisions among reached runs" if step else "not reached within budget"
            rows.append("  %s: %s/%s policies; %s." % (label, target["reached_runs"], target["total_runs"], text))
    rows += ["", "Behavioral promotion gates are frozen engineering requirements, not professor-set",
             "grading thresholds. Oxygen collection is ranked only after the survival floor",
             "is met, and two consecutive checks are required for promotion. Runs that never",
             "reach a target are reported explicitly, not replaced by zero steps.", "",
             "## Learning curves and reproduction", "",
             "`figures/loss.svg` shows pre-update squared TD residuals in fixed-decision blocks.",
             "Flight and avoidance residuals decrease strongly. Oxygen residuals persist and rise later;",
             "this run demonstrates better collection behavior, not zero-loss convergence for oxygen.",
             "`survival.svg`, `completion.svg` and `oxygen.svg` show frozen validation",
             "performance at checkpoint zero and later checkpoints; `coverage.svg` shows updated states.",
             "The x axis is Q decisions; phase budgets differ and actual physics steps are reported separately.",
             "Lines are policy means and bands are +/- one sample SD (clipped at the",
             "physical metric bounds when needed). Checkpoint curves use validation,",
             "while the tables above use the separate final test set.", "",
             "`training_time.svg` and `performance.svg` compare costs and selected-policy",
             "test performance. Raw episode CSVs, validation CSVs and model hashes are",
             "retained. Regenerate this report with `python -m astronaut_rl.report`.", "",
             "Validation seeds are 9590000..9590049 for stochastic stages; final test seeds are",
             "9600000..9600099. Flight uses one deterministic episode at each evaluation.", "",
             "The figures show finite-budget behavior. Sparse zero rewards can produce",
             "small TD residuals even for weak policies, so low loss alone is not evidence",
             "of success or theoretical convergence. Compact state bins still merge",
             "continuous situations. Sample SD across chains is not a confidence interval.", ""]
    return "\n".join(rows)


def build(results, markdown_out=None):
    results = Path(results)
    summary = aggregate(results)
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    summary_path = results / "aggregate.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out else results.parent / "RESULTS.md"
    markdown_path.write_text(markdown(summary, manifest), encoding="utf-8")
    return summary_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Markdown path (default RESULTS.md beside results/)")
    args = parser.parse_args()
    for path in build(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
