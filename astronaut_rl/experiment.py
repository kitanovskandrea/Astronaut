"""Gated flight -> avoidance -> oxygen curriculum using genuine tabular Q-learning.

Official rewards/physics are unchanged. Macro rewards and bootstraps use a
physical-frame discount, and child tables start from learned parent values.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

from .config import config_hash, load_config
from .environment import Game, VARIANTS, ENV_COMMIT
from .qlearning import QLearner
from .state import StateEncoder
from .transfer import PARENTS, initialize_from_parent, inspect_parent


def mean(values):
    return statistics.fmean(values) if values else 0.0


def evaluate(agent, variant, seeds, repeat=None, random_policy=False, render=False,
             profile=None, state_spec=None):
    """Evaluate without changing Q values, visit counts or the agent's RNG."""
    repeat = agent.metadata.get("repeat", 1) if repeat is None else repeat
    profile = profile if profile is not None else agent.metadata.get("profile", "official")
    state_spec = state_spec if state_spec is not None else agent.metadata.get("state_spec")
    game = Game(variant, repeat, render, profile=profile, gamma=agent.gamma, state_spec=state_spec)
    rows = []
    try:
        for seed in seeds:
            state = game.reset(seed)
            rng = random.Random(seed + 7000000)
            total, frames, decisions, unseen = 0.0, 0, 0, 0
            while True:
                unseen += int(state not in agent.q)
                action = rng.randrange(agent.actions) if random_policy else agent.greedy(state)
                transition = game.step(action)
                frames += transition.frames
                decisions += 1
                total += transition.reward
                state = transition.state
                if transition.terminated or transition.truncated:
                    rows.append({"seed": seed, "return": total, "frames": frames,
                                 "decisions": decisions,
                                 "survived": int(transition.truncated and not transition.terminated),
                                 "oxygen": int(transition.info.get("oxygen_count", 0)),
                                 "collision": transition.info["collision"],
                                 "unseen_fraction": unseen / decisions})
                    break
    finally:
        game.close()
    return rows


def evaluation_summary(rows):
    total_frames = sum(row["frames"] for row in rows)
    total_oxygen = sum(row["oxygen"] for row in rows)
    summary = {"episodes": len(rows), "return_mean": mean([r["return"] for r in rows]),
               "frames_mean": mean([r["frames"] for r in rows]),
               "survival_rate": mean([r["survived"] for r in rows]),
               "oxygen_mean": mean([r["oxygen"] for r in rows]),
               "oxygen_per_1000_frames": 1000 * total_oxygen / total_frames if total_frames else 0.0,
               "combined_success_rate": mean([int(r["survived"] and r["oxygen"] >= 3) for r in rows]),
               "unseen_fraction": mean([r["unseen_fraction"] for r in rows])}
    for collision in ("ground", "ceiling", "asteroid"):
        summary["collision_%s_rate" % collision] = mean([int(r["collision"] == collision) for r in rows])
    return summary


def coverage_summary(agent, state_capacity):
    """Describe actual state/action updates; theoretical coverage is diagnostic."""
    counts = [count for row in agent.visits.values() for count in row if count > 0]
    states = sum(any(count > 0 for count in row) for row in agent.visits.values())
    pairs = len(counts)
    possible_pairs = state_capacity * agent.actions
    return {"state_capacity": state_capacity, "possible_state_action_pairs": possible_pairs,
            "visited_states": states, "visited_state_action_pairs": pairs,
            "zero_visit_actions_in_visited_states": states * agent.actions - pairs,
            "state_coverage_fraction": states / state_capacity if state_capacity else 0.0,
            "state_action_coverage_fraction": pairs / possible_pairs if possible_pairs else 0.0,
            "visited_pairs_lt10_fraction": mean([int(v < 10) for v in counts]),
            "visited_pairs_ge10_fraction": mean([int(v >= 10) for v in counts]),
            "visited_pairs_ge50_fraction": mean([int(v >= 50) for v in counts]),
            "visited_pairs_ge100_fraction": mean([int(v >= 100) for v in counts]),
            "min_visits": min(counts) if counts else 0,
            "median_visits": statistics.median(counts) if counts else 0.0,
            "max_visits": max(counts) if counts else 0,
            "total_updates": sum(counts)}


def validation_score(variant, metrics, gate=None):
    """Safety constrained selection; earliest ties are retained by the caller."""
    if variant == "oxygen":
        floor = (gate or {}).get("minimum_survival_rate", 0.9)
        eligible = metrics["survival_rate"] >= floor
        return ((1, metrics["oxygen_mean"], metrics["survival_rate"], metrics["frames_mean"])
                if eligible else (0, metrics["survival_rate"], metrics["oxygen_mean"], metrics["frames_mean"]))
    return metrics["survival_rate"], metrics["frames_mean"], metrics["return_mean"]


def derive_gate(config, baseline):
    settings = config["gate"]
    survival = settings["minimum_survival_rate"]
    oxygen = settings["minimum_oxygen_mean"]
    if config["variant"] == "oxygen":
        survival = max(survival, baseline["survival_rate"] - settings["survival_retention_margin"])
        oxygen = max(oxygen, baseline["oxygen_mean"] + settings["minimum_oxygen_improvement"])
    return {**settings, "minimum_survival_rate": survival, "minimum_oxygen_mean": oxygen,
            "baseline_survival_rate": baseline["survival_rate"],
            "baseline_oxygen_mean": baseline["oxygen_mean"],
            "required_consecutive_passes": config["required_consecutive_passes"]}


def gate_passed(metrics, gate):
    return (metrics["survival_rate"] >= gate["minimum_survival_rate"]
            and metrics["oxygen_mean"] >= gate["minimum_oxygen_mean"])


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def discounted_transition(game, action, gamma_per_frame):
    """Return a real macro transition and its frame-discounted Q reward."""
    reward = 0.0
    observed = 0

    def frame_callback(observation, raw_reward, terminated, truncated, info, frame):
        nonlocal reward, observed
        if frame != observed + 1:
            raise ValueError("Physics callback frames are not sequential")
        reward += gamma_per_frame ** (frame - 1) * raw_reward
        observed += 1

    transition = game.step(action, on_frame=frame_callback)
    if observed != transition.frames or not 1 <= observed <= game.repeat:
        raise ValueError("Macro transition and physics callbacks disagree")
    return transition, reward


def _resolved_config(variant, budget, repeat, alpha, gamma, profile, epsilon_start,
                     configuration, checkpoint_interval, validation_seeds, test_seeds,
                     gamma_per_frame):
    config = json.loads(json.dumps(configuration if configuration is not None else load_config(variant), allow_nan=False))
    config.update(variant=variant, steps=budget, budget_unit="decision_updates")
    for key, value in (("repeat", repeat), ("alpha", alpha), ("profile", profile),
                       ("epsilon_start", epsilon_start), ("checkpoint_interval", checkpoint_interval),
                       ("validation_seeds", validation_seeds), ("test_seeds", test_seeds)):
        if value is not None:
            config[key] = list(value) if key.endswith("_seeds") else value
    if gamma_per_frame is not None:
        config["gamma_per_frame"] = gamma_per_frame
    elif gamma is not None:
        config["gamma_per_frame"] = gamma ** (1 / config["repeat"])
    config["gamma"] = config["gamma_per_frame"] ** config["repeat"]
    config["epsilon_decay_decisions"] = max(1, int(budget * config["epsilon_decay_fraction"]))
    if type(budget) is not int or budget < 1:
        raise ValueError("Decision budget must be a positive integer")
    if config["profile"] != "official" or type(config["repeat"]) is not int or config["repeat"] < 1:
        raise ValueError("Use official rewards and a positive integer action repeat")
    if config["environment_commit"] != ENV_COMMIT:
        raise ValueError("Configuration requires another course environment commit")
    if config["state_capacity"] != StateEncoder(variant, config["state_spec"]).capacity:
        raise ValueError("state_capacity differs from the frozen state thresholds")
    if type(config["checkpoint_interval"]) is not int or config["checkpoint_interval"] < 1:
        raise ValueError("Checkpoint interval must be a positive integer")
    for key in ("epsilon_start", "epsilon_end", "epsilon_decay_fraction", "gamma_per_frame"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid " + key)
        if key in ("epsilon_decay_fraction", "gamma_per_frame") and value == 0:
            raise ValueError("Invalid " + key)
    if config["epsilon_start"] < config["epsilon_end"]:
        raise ValueError("epsilon_start must be at least epsilon_end")
    if config["parent_variant"] != PARENTS[variant] or config["initialization"] != ("zeros" if variant == "fly" else "parent_q"):
        raise ValueError("Invalid curriculum initialization")
    if type(config["required_consecutive_passes"]) is not int or config["required_consecutive_passes"] < 1:
        raise ValueError("Invalid consecutive checkpoint requirement")
    for key in ("validation_seeds", "test_seeds"):
        seeds = config[key]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(key + " must contain distinct nonnegative integer seeds")
    if set(config["validation_seeds"]) & set(config["test_seeds"]):
        raise ValueError("Validation and test seeds must be disjoint")
    return config


def train(variant, seed, budget, out, model_dir, repeat=None, alpha=None, gamma=None,
          progress_path=None, profile=None, epsilon_start=None, *, config=None,
          checkpoint_interval=None, validation_seeds=None, test_seeds=None,
          gamma_per_frame=None, parent_model=None, parent_record=None, progress_context=None):
    """Run a fixed-budget stage; a failed gate is recorded without advancement."""
    config = _resolved_config(variant, budget, repeat, alpha, gamma, profile, epsilon_start,
        config, checkpoint_interval, validation_seeds, test_seeds, gamma_per_frame)
    if type(seed) is not int or seed < 0 or seed in config["validation_seeds"] or seed in config["test_seeds"]:
        raise ValueError("Training seed must be nonnegative and separate from evaluation")
    out, model_dir = Path(out), Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{variant}_seed{seed}"
    model_path, final_path = model_dir / (prefix + ".json.gz"), model_dir / (prefix + "_final.json.gz")
    checkpoint_dir = model_dir / "checkpoints" / prefix
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initialization_start = time.perf_counter()
    agent = QLearner(alpha=config["alpha"], gamma=config["gamma"], seed=seed)
    initialization = {"type": "zeros", "parent_variant": None, "parent_model_path": None,
                      "parent_model_sha256": None, "parent_seed": None, "parent_updates": 0,
                      "parent_physics_frames": 0, "transferred_rows": 0}
    if variant != "fly":
        if parent_model is None:
            raise ValueError("The child stage requires its qualified learned parent model")
        parent, initialization = inspect_parent(parent_model, config, seed, model_dir)
        initialization["transferred_rows"] = initialize_from_parent(agent, parent, variant, config["state_spec"])
    elif parent_model is not None:
        raise ValueError("Flight must start with a fresh zero Q table")
    initialized_rows = len(agent.q)
    initialization_seconds = time.perf_counter() - initialization_start
    metadata = {"variant": variant, "seed": seed, "budget": budget, "repeat": config["repeat"],
                "alpha": agent.alpha, "gamma": agent.gamma, "gamma_per_frame": config["gamma_per_frame"],
                "profile": "official", "epsilon_start": config["epsilon_start"],
                "epsilon_end": config["epsilon_end"], "epsilon_decay_fraction": config["epsilon_decay_fraction"],
                "state_spec": config["state_spec"], "state_capacity": config["state_capacity"],
                "config": config, "config_hash": config_hash(config), "environment_commit": ENV_COMMIT,
                "budget_unit": "decision_updates", "initialization": initialization,
                "initialized_rows": initialized_rows, "transferred_rows": initialization["transferred_rows"]}
    agent.metadata = metadata
    game = Game(variant, repeat=config["repeat"], profile="official", gamma=agent.gamma, state_spec=config["state_spec"])
    state = game.reset(seed)
    logs, episode_log, validations, validation_episodes = [], [], [], []
    block_errors, block_raw_rewards = [], []
    episodes = frames = decision = ep_frames = ep_decisions = 0
    ep_return = ep_loss = 0.0
    epsilon = config["epsilon_start"]
    train_seconds = validation_seconds = export_seconds = 0.0
    active_start = None
    selected_step = selected_validation = selected_checkpoint = best_score = None
    gate = protocol = None
    consecutive_passes = 0
    first_target_step = stability_step = None
    last_info = None

    def pause_timer():
        nonlocal active_start, train_seconds
        if active_start is not None:
            train_seconds += time.perf_counter() - active_start
            active_start = None

    def elapsed_training():
        return train_seconds + (time.perf_counter() - active_start if active_start is not None else 0.0)

    def progress(phase, **extra):
        if progress_path is not None:
            context = progress_context or {}
            _write_progress(progress_path, {**context, "phase": phase, "variant": variant, "seed": seed,
                "decision": decision, "step": decision, "budget": budget, "episodes": episodes,
                "frames": frames, "states": len(agent.visits), "visited_states": len(agent.visits),
                "stored_q_rows": len(agent.q), "initialized_rows": initialized_rows,
                "epsilon": epsilon, "td_mse": logs[-1]["td_mse"] if logs else None,
                "train_seconds": elapsed_training(), "selected_step": selected_step,
                "overall_decision": context.get("ancestor_decisions", 0) + decision, **extra})

    def checkpoint():
        nonlocal validation_seconds, export_seconds, gate, protocol, selected_step
        nonlocal selected_validation, selected_checkpoint, best_score, consecutive_passes
        nonlocal first_target_step, stability_step
        progress("validating", evaluation_kind="validation")
        evaluation_start = time.perf_counter()
        rows = evaluate(agent, variant, config["validation_seeds"], config["repeat"],
                        profile="official", state_spec=config["state_spec"])
        metrics = evaluation_summary(rows)
        validation_seconds += time.perf_counter() - evaluation_start
        if gate is None:
            gate = derive_gate(config, metrics)
            protocol = {"budget_unit": "decision_updates", "repeat": config["repeat"],
                "gamma_per_frame": config["gamma_per_frame"], "reward": "unchanged_official",
                "q_reward": "sum(gamma_per_frame**(k-1)*raw_frame_reward)",
                "bootstrap_discount": "gamma_per_frame**actual_executed_frames",
                "selection": "survival_constraint_then_collection" if variant == "oxygen" else "survival_first_lexicographic",
                "validation_seeds": config["validation_seeds"], "test_seeds": config["test_seeds"],
                "validation_target": {"survival_rate": gate["minimum_survival_rate"], "oxygen_mean": gate["minimum_oxygen_mean"]},
                "gate": gate, "stability_required_checkpoints": config["required_consecutive_passes"],
                "training_reset": "seed_once_then_continue", "checkpoint_zero": True}
            metadata["protocol"] = protocol
            agent.metadata = metadata
        passing = gate_passed(metrics, gate)
        consecutive_passes = consecutive_passes + 1 if passing else 0
        if passing and first_target_step is None:
            first_target_step = decision
        if consecutive_passes >= gate["required_consecutive_passes"] and stability_step is None:
            stability_step = decision
        path = checkpoint_dir / f"step{decision:07d}.json.gz"
        export_start = time.perf_counter()
        snapshot = {**metadata, "updates": decision, "checkpoint_step": decision, "physics_frames": frames}
        agent.save(path, metadata=snapshot)
        score = validation_score(variant, metrics, gate)
        if best_score is None or score > best_score:
            best_score, selected_step, selected_validation = score, decision, metrics
            selected_checkpoint = str(path.relative_to(model_dir))
        coverage = coverage_summary(agent, config["state_capacity"])
        validations.append({"checkpoint_idx": len(validations) + 1, "step": decision, "decision": decision,
            "frames": frames, "variant": variant, "seed": seed, "model_identifier": str(path.relative_to(model_dir)),
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "train_seconds": train_seconds,
            "selected": 0, "eligible": int(metrics["survival_rate"] >= gate["minimum_survival_rate"]),
            "passing": int(passing), "target_met": int(passing), "consecutive_passes": consecutive_passes,
            **metrics, **{"coverage_" + key: value for key, value in coverage.items()}})
        validation_episodes.extend({"checkpoint_step": decision, **row} for row in rows)
        export_seconds += time.perf_counter() - export_start

    def episode_row(completed, interrupted, transition=None):
        info = transition.info if transition is not None else last_info
        return {"variant": variant, "seed": seed, "episode": episodes if completed else episodes + 1,
            "decision": decision, "step": decision, "frames": ep_frames, "decisions": ep_decisions,
            "return": ep_return, "mean_td_loss": ep_loss / ep_decisions,
            "epsilon": epsilon, "oxygen": int(info.get("oxygen_count", 0)), "collision": info["collision"],
            "terminated": int(transition.terminated) if transition else 0,
            "truncated": int(transition.truncated) if transition else 0,
            "completed": int(completed), "budget_interrupted": int(interrupted),
            "train_seconds": elapsed_training(), "visited_states": len(agent.visits)}

    try:
        progress("training")
        checkpoint()  # Frozen inherited/zero policy is an explicit fallback.
        progress("training")
        active_start = time.perf_counter()
        for decision in range(1, budget + 1):
            fraction = max(0.0, 1 - decision / (config["epsilon_decay_fraction"] * budget))
            epsilon = config["epsilon_end"] + (config["epsilon_start"] - config["epsilon_end"]) * fraction
            action = agent.choose(state, epsilon)
            transition, q_reward = discounted_transition(game, action, config["gamma_per_frame"])
            delta = agent.update(state, action, q_reward, transition.state, transition.terminated,
                                 discount=config["gamma_per_frame"] ** transition.frames)
            block_errors.append(delta * delta)
            block_raw_rewards.append(transition.reward)
            frames += transition.frames
            ep_frames += transition.frames
            ep_decisions += 1
            ep_return += transition.reward
            ep_loss += delta * delta
            state, last_info = transition.state, transition.info
            if transition.terminated or transition.truncated:
                episodes += 1
                episode_log.append(episode_row(True, False, transition))
                ep_frames = ep_decisions = 0
                ep_return = ep_loss = 0.0
                if decision < budget:
                    state = game.reset(None)
            if decision % 2000 == 0 or decision == budget:
                logs.append({"step": decision, "decision": decision, "frames": frames, "episodes": episodes,
                    "epsilon": epsilon, "td_mse": mean(block_errors),
                    "mean_reward_per_decision": mean(block_raw_rewards), "visited_states": len(agent.visits),
                    "stored_q_rows": len(agent.q), "initialized_rows": initialized_rows,
                    "train_seconds": elapsed_training()})
                block_errors.clear(); block_raw_rewards.clear()
                if progress_path is not None:
                    pause_timer(); progress("training"); active_start = time.perf_counter()
            if decision % config["checkpoint_interval"] == 0 or decision == budget:
                pause_timer(); checkpoint()
                if decision < budget:
                    progress("training"); active_start = time.perf_counter()
        pause_timer()
        if ep_decisions:
            episode_log.append(episode_row(False, True))
    finally:
        pause_timer(); game.close()

    selected_passed = gate_passed(selected_validation, gate)
    qualified = bool(selected_passed and stability_step is not None)
    gate_result = {**gate, "selected_passed": selected_passed, "stability_step": stability_step,
                   "qualified": qualified}
    export_start = time.perf_counter()
    selected = QLearner.load(model_dir / selected_checkpoint)
    selected.metadata.update(selected_step=selected_step, model_role="selected", qualified=qualified, gate=gate_result)
    selected.save(model_path)
    agent.save(final_path, metadata={**metadata, "updates": budget, "checkpoint_step": budget,
        "physics_frames": frames, "selected_step": selected_step, "model_role": "final",
        "qualified": qualified, "gate": gate_result})
    full_coverage = coverage_summary(agent, config["state_capacity"])
    selected_coverage = coverage_summary(selected, config["state_capacity"])
    export_seconds += time.perf_counter() - export_start
    progress("evaluating", evaluation_kind="test")
    evaluation_start = time.perf_counter()
    greedy = evaluate(selected, variant, config["test_seeds"], config["repeat"])
    baseline = evaluate(selected, variant, config["test_seeds"], config["repeat"], random_policy=True)
    test_seconds = time.perf_counter() - evaluation_start
    greedy_summary, random_summary = evaluation_summary(greedy), evaluation_summary(baseline)
    for row in validations:
        row["selected"] = int(row["step"] == selected_step)
    export_start = time.perf_counter()
    for suffix, rows in (("training", logs), ("episodes", episode_log), ("validation", validations),
                         ("validation_episodes", validation_episodes), ("evaluation", greedy), ("random", baseline)):
        write_csv(out / f"{prefix}_{suffix}.csv", rows)
    selected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    export_seconds += time.perf_counter() - export_start
    prior = (parent_record or {}).get("cumulative", {"decision_updates": 0, "physics_frames": 0, "train_seconds": 0.0})
    stage_cost = {"decision_updates": budget, "physics_frames": frames, "train_seconds": train_seconds}
    cumulative = {key: stage_cost[key] + prior[key] for key in stage_cost}
    record = {**metadata, "updates": budget, "train_seconds": train_seconds,
        "initialization_seconds": initialization_seconds, "validation_seconds": validation_seconds,
        "test_seconds": test_seconds, "export_seconds": export_seconds, "frames": frames, "episodes": episodes,
        "states": len(agent.q), "first_td_mse": logs[0]["td_mse"], "last_td_mse": logs[-1]["td_mse"],
        "greedy": greedy_summary, "random": random_summary, "qualified": qualified, "gate": gate_result,
        "stage_cost": stage_cost, "cumulative": cumulative, "selected_step": selected_step,
        "model_path": model_path.name, "model_sha256": selected_hash,
        "final_model_path": final_path.name, "final_model_sha256": final_hash,
        "coverage": full_coverage, "first_target_step": first_target_step, "stability_step": stability_step,
        "selected": {"step": selected_step, "validation": selected_validation, "greedy": greedy_summary,
                     "checkpoint_path": selected_checkpoint, "coverage": selected_coverage},
        "final": {"step": budget, "validation": {key: validations[-1][key] for key in selected_validation},
                  "model_path": final_path.name, "model_sha256": final_hash, "coverage": full_coverage},
        "validation_history": validations}
    (out / f"{prefix}_summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    progress("complete" if qualified else "failed_gate", model_path=str(model_path.resolve()),
             greedy=greedy_summary, qualified=qualified, gate=gate_result,
             error=None if qualified else f"{variant} did not pass its safety/collection gate; dependent stages were not started")
    print(json.dumps(record, allow_nan=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--curriculum-to", choices=VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 79])
    parser.add_argument("--steps", "--decisions", dest="steps", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gamma-per-frame", type=float, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--profile", choices=("official",), default="official")
    args = parser.parse_args()
    target = args.curriculum_to or ("oxygen" if args.variant == "all" else args.variant)
    if args.curriculum_to is not None and args.variant not in ("all", args.curriculum_to):
        parser.error("--variant and --curriculum-to disagree")
    if args.steps is not None and args.steps < 1:
        parser.error("Decision budget must be positive")
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        parser.error("Training seeds must be distinct nonnegative integers")
    if args.progress is not None and len(args.seeds) != 1:
        parser.error("--progress requires exactly one training seed")
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    stages = VARIANTS[:VARIANTS.index(target) + 1]
    settings = {variant: load_config(variant) for variant in stages}
    budgets = {variant: args.steps if args.steps is not None else settings[variant]["steps"] for variant in stages}
    manifest = {"schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "training_seeds": args.seeds, "variants": list(stages),
        "target_variant": target, "budget_unit": "decision_updates", "records": [], "failed_stages": [],
        "timing_protocol": "sequential per-seed curriculum; active training excludes initialization/validation/test/export/progress",
        "complete": False, "curriculum_passed": False}
    for seed in args.seeds:
        previous = None
        ancestor_decisions = 0
        for index, variant in enumerate(stages):
            parent_model = args.models / previous["model_path"] if previous is not None else None
            try:
                record = train(variant, seed, budgets[variant], args.out, args.models, args.repeat,
                    args.alpha, args.gamma, args.progress, args.profile, args.epsilon_start,
                    config=settings[variant], checkpoint_interval=args.checkpoint_interval,
                    gamma_per_frame=args.gamma_per_frame, parent_model=parent_model, parent_record=previous,
                    progress_context={"target_variant": target, "curriculum_stage_index": index + 1,
                        "curriculum_stage_count": len(stages), "overall_budget": sum(budgets.values()),
                        "ancestor_decisions": ancestor_decisions})
            except ValueError as error:
                parser.error(str(error))
            manifest["records"].append(record)
            _write_progress(args.out / "run_manifest.json", manifest)
            if not record["qualified"]:
                manifest["failed_stages"].append({"seed": seed, "variant": variant,
                    "skipped_dependents": list(stages[index + 1:]), "gate": record["gate"]})
                break
            previous = record
            ancestor_decisions += budgets[variant]
    manifest["complete"] = True
    manifest["curriculum_passed"] = not manifest["failed_stages"]
    _write_progress(args.out / "run_manifest.json", manifest)
    if not manifest["curriculum_passed"]:
        parser.exit(1, "Curriculum gate failed; completed stage artifacts were retained and dependent stages were skipped.\n")


if __name__ == "__main__":
    main()
