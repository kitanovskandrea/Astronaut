"""Finite Q-table initialization from a qualified learned parent policy.

There is no flight controller here. Every appended child state initially has
exactly the parent's two learned values, then receives ordinary Q updates.
Visit counts always describe the current stage, never inherited experience.
"""
import hashlib
from pathlib import Path

from .environment import ENV_COMMIT
from .qlearning import QLearner
from .state import StateEncoder, child_suffixes

PARENTS = {"fly": None, "asteroids": "fly", "oxygen": "asteroids"}


def validate_prefix(parent_variant, parent_spec, child_variant, child_spec):
    if PARENTS.get(child_variant) != parent_variant:
        raise ValueError("Curriculum transfer must follow flight -> asteroids -> oxygen")
    parent = StateEncoder(parent_variant, parent_spec).spec
    child = StateEncoder(child_variant, child_spec).spec
    fields = ["type", "height", "velocity"]
    if parent_variant == "asteroids":
        fields += ["object_time", "asteroid_dy", "asteroid_projection_cap", "projection_factor"]
    for field in fields:
        if parent.get(field) != child.get(field):
            raise ValueError("Parent and child state prefixes differ: " + field)
    if child["type"] == "next_event":
        raise ValueError("Curriculum states must retain the parent hazard prefix")


def initialize_from_parent(agent, parent, variant, state_spec):
    """Copy values into a fresh child table; return the initialized row count."""
    if agent.q or agent.visits:
        raise ValueError("Transfer requires a fresh child Q table")
    validate_prefix(parent.metadata["variant"], parent.metadata["state_spec"], variant, state_spec)
    suffixes = child_suffixes(variant, state_spec)
    for state, values in parent.q.items():
        for suffix in suffixes:
            agent.q[state + suffix] = values[:]
    return len(agent.q)


def inspect_parent(path, child_config, seed, model_dir):
    """Load a compatible, qualified same-seed predecessor with portable provenance."""
    path, root = Path(path).resolve(), Path(model_dir).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Parent model must be inside the run's model directory")
    parent = QLearner.load(path)
    metadata = parent.metadata
    if metadata.get("variant") != PARENTS[child_config["variant"]]:
        raise ValueError("Incorrect parent stage for curriculum transfer")
    if metadata.get("seed") != seed or parent.seed != seed:
        raise ValueError("Parent must belong to the same independent training seed")
    if metadata.get("profile") != "official" or metadata.get("environment_commit") != ENV_COMMIT:
        raise ValueError("Parent does not use the pinned official environment and rewards")
    if metadata.get("qualified") is not True:
        raise ValueError("Parent stage did not pass its curriculum gate")
    if metadata.get("model_role") != "selected":
        raise ValueError("Parent must be the qualified selected policy, not a final/checkpoint table")
    if metadata.get("repeat") != child_config["repeat"] or parent.gamma != child_config["gamma"]:
        raise ValueError("Parent and child action/discount units differ")
    validate_prefix(metadata["variant"], metadata["state_spec"], child_config["variant"], child_config["state_spec"])
    info = {"type": "parent_q", "parent_variant": metadata["variant"],
            "parent_model_path": str(path.relative_to(root)),
            "parent_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parent_seed": seed, "parent_updates": metadata["updates"],
            "parent_physics_frames": metadata["physics_frames"], "transferred_rows": 0}
    return parent, info


def verify_initialization(child, parent, variant, state_spec):
    """Check a saved checkpoint-zero table against its actual parent values."""
    validate_prefix(parent.metadata["variant"], parent.metadata["state_spec"], variant, state_spec)
    suffixes = child_suffixes(variant, state_spec)
    if len(child.q) != len(parent.q) * len(suffixes):
        raise ValueError("Initial child table has the wrong transferred row count")
    if any(count for row in child.visits.values() for count in row):
        raise ValueError("Checkpoint zero inherited visit counts")
    for state, values in parent.q.items():
        for suffix in suffixes:
            if child.q.get(state + suffix) != values:
                raise ValueError("Checkpoint zero differs from learned parent Q values")
