"""Thin adapter around the pinned, unmodified official Astronaut environment."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import famnit_gym
from famnit_gym.envs.astronaut.astronaut_env import gym
from famnit_gym.wrappers.astronaut import Asteroids, Oxygen
from famnit_gym.wrappers.astronaut.asteroids import pygame as course_display

VARIANTS = ("fly", "asteroids", "oxygen")
ENV_COMMIT = "0db0652ad3b8cece05d476c56637a3dcb56557cd"


def default_profile(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return "official"


@lru_cache(maxsize=1)
def check_environment():
    """Version 0.2 was reused upstream: verify code, not just its version label."""
    provenance = Path(__file__).resolve().parents[1] / "vendor" / "PROVENANCE.json"
    expected = json.loads(provenance.read_text())
    package_parent = Path(famnit_gym.__file__).resolve().parent.parent
    for relative, checksum in expected["source_sha256"].items():
        path = package_parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(
                "The installed famnit-gym source differs from the submitted experiment. "
                "Both upstream snapshots use version 0.2. From the project folder run: "
                "python -m pip install --force-reinstall --no-deps "
                "vendor/famnit_gym-0.2-py3-none-any.whl")
    return expected["commit"]


def make_env(variant, render=False):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    check_environment()
    # No options override: rewards, gravity, thrust, speed and horizon are the
    # course defaults. The wrappers still report their real returned features.
    env = gym.make("famnit_gym/Astronaut-v1", render_mode="human" if render else None,
                   disable_env_checker=True)
    if variant in ("asteroids", "oxygen"):
        env = Asteroids(env)
    if variant == "oxygen":
        env = Oxygen(env)
    return env


@dataclass
class Transition:
    state: tuple
    reward: float
    terminated: bool
    truncated: bool
    frames: int
    info: dict


class WindowClosed(Exception):
    """The user closed the replay; this is not a game outcome."""


class Game:
    def __init__(self, variant, repeat=None, render=False, profile="official", gamma=None,
                 state_spec=None):
        if profile != "official":
            raise ValueError("This experiment uses profile='official'. Earlier shaped models "
                             "require the archived project and its original course snapshot.")
        if repeat is None or gamma is None or state_spec is None:
            from .config import load_config
            settings = load_config(variant)
            repeat = settings['repeat'] if repeat is None else repeat
            gamma = settings['gamma'] if gamma is None else gamma
            state_spec = settings['state_spec'] if state_spec is None else state_spec
        if type(repeat) is not int or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        from .state import StateEncoder
        self.profile = profile
        self.env = make_env(variant, render)
        self.encoder = StateEncoder(variant, state_spec)
        self.state_spec = self.encoder.spec
        self.repeat = repeat
        self.render = render
        self.gamma = gamma

    def reset(self, seed=None):
        # New upstream wrappers use the environment's own np_random. A seed is
        # supplied once at training start; reset(None) continues that sequence.
        self.encoder.reset()
        obs, info = self.env.reset(seed=seed)
        return self.encoder.encode(obs, info)

    def step(self, action, on_frame=None):
        total = 0.0
        for frame in range(1, self.repeat + 1):
            obs, reward, term, trunc, info = self.env.step(action)
            if self.render:
                for event in course_display.event.get():
                    if event.type == course_display.QUIT or (
                        event.type == course_display.KEYDOWN and event.key == course_display.K_ESCAPE
                    ):
                        raise WindowClosed()
            total += float(reward)
            state = self.encoder.encode(obs, info)
            if on_frame is not None:
                on_frame(obs, float(reward), bool(term), bool(trunc), info, frame)
            if term or trunc:
                break
        return Transition(state, total, bool(term), bool(trunc), frame, info)

    def close(self):
        self.env.close()
