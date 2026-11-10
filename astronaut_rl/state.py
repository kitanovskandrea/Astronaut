"""Finite public-observation states, with exact thresholds stored per model.

Binning is an agent-side approximation to continuous game state. No encoder
chooses an action, changes a reward, or reads a private game variable.
"""
from bisect import bisect_right
from copy import deepcopy
from itertools import product
import math

VARIANTS = ("fly", "asteroids", "oxygen")
ASTRONAUT_X = 300.0
SCROLL_SPEED = 500.0
Y_EDGES = tuple(114.0 + index * (572.0 / 6.0) for index in range(1, 6))
V_EDGES = (-120.0, -25.0, 25.0, 120.0)
TIME_EDGES = (0.25, 0.6, 1.2)
DY_EDGES = (-90.0, 90.0)
ABSENT_OXYGEN = (4, 1)
STATE_CAPACITIES = {"fly": 30, "asteroids": 360, "oxygen": 4680}

PROJECTED_Y_EDGES = (160, 230, 310, 400, 490, 570, 640)
PROJECTED_V_EDGES = (-300, -180, -90, -30, 30, 90, 180, 300)
PROJECTED_TIME_EDGES = (.18, .4, .7, 1.2)
PROJECTED_ASTEROID_EDGES = (-240, -160, -100, 0, 100, 160, 240)
PROJECTED_OXYGEN_EDGES = (-220, -120, -50, 50, 120, 220)


def default_spec(variant):
    """Return a fresh JSON-compatible description of the coarse hypothesis."""
    if variant not in VARIANTS:
        raise ValueError("Unknown game variant: %s" % variant)
    return {"type": "grid", "height": list(Y_EDGES), "velocity": list(V_EDGES),
            "object_time": list(TIME_EDGES), "asteroid_dy": list(DY_EDGES),
            "oxygen_dy": list(DY_EDGES)}


def arrival_time(object_x):
    """Seconds until the object's center reaches the fixed astronaut x."""
    return max(0.0, (object_x - ASTRONAUT_X) / SCROLL_SPEED)


def projected_spec(variant):
    """Fresh thresholds for the learned, prefix-preserving curriculum.

    Projection estimates relative clearance under the observed velocity. It
    does not select thrust or simulate a preferred action.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown game variant: %s" % variant)
    return {"type": "projected", "height": list(PROJECTED_Y_EDGES),
            "velocity": list(PROJECTED_V_EDGES),
            "object_time": list(PROJECTED_TIME_EDGES),
            "asteroid_dy": list(PROJECTED_ASTEROID_EDGES),
            "oxygen_dy": list(PROJECTED_OXYGEN_EDGES),
            "asteroid_projection_cap": .75, "oxygen_projection_cap": .6,
            "projection_factor": 1.0}


def _spec(variant, specification):
    if variant not in VARIANTS:
        raise ValueError("Unknown game variant: %s" % variant)
    result = default_spec(variant) if specification is None else deepcopy(specification)
    if not isinstance(result, dict):
        raise ValueError("State specification must be a dictionary")
    result.setdefault("type", "grid")
    if result["type"] not in ("grid", "next_event", "projected"):
        raise ValueError("Unknown state representation")
    if result["type"] == "next_event" and variant != "oxygen":
        raise ValueError("next_event representation requires oxygen")
    required = ["height", "velocity"]
    if variant != "fly":
        required += ["object_time", "asteroid_dy"]
    if variant == "oxygen":
        required += ["oxygen_dy"]
    if result["type"] == "next_event":
        result.setdefault("lookahead_dy", [-90.0, 90.0])
        required += ["lookahead_dy"]
    for name in required:
        values = result.get(name)
        if not isinstance(values, (list, tuple)):
            raise ValueError("State thresholds must be a list: %s" % name)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            raise ValueError("State thresholds must be finite numbers: %s" % name)
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("State thresholds must increase strictly: %s" % name)
        result[name] = list(map(float, values))
    if result["type"] == "projected":
        for name, default in (("asteroid_projection_cap", .75),
                              ("oxygen_projection_cap", .6),
                              ("projection_factor", 1.0)):
            value = result.setdefault(name, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("Projection settings must be finite and nonnegative: " + name)
            result[name] = float(value)
    return result


def _capacity(variant, spec):
    flight = (len(spec["height"]) + 1) * (len(spec["velocity"]) + 1)
    if variant == "fly":
        return flight
    times = len(spec["object_time"]) + 1
    asteroid = times * (len(spec["asteroid_dy"]) + 1)
    if variant == "asteroids":
        return flight * asteroid
    oxygen = times * (len(spec["oxygen_dy"]) + 1)
    if spec["type"] == "next_event":
        # Asteroid next: coarse available-tank lookahead or an absent code.
        # Oxygen next: detailed oxygen error; lookahead is the absent code.
        return flight * (asteroid * (len(spec["lookahead_dy"]) + 2) + oxygen)
    return flight * asteroid * (oxygen + 1)


def state_capacity(variant, spec=None):
    return _capacity(variant, _spec(variant, spec))


def child_suffixes(variant, spec=None):
    """All appended object codes when expanding a learned parent Q table."""
    normalized = _spec(variant, spec)
    if variant == "fly" or normalized["type"] == "next_event":
        raise ValueError("Transfer suffixes require a grid/projected child task")
    times = len(normalized["object_time"]) + 1
    edges = normalized["asteroid_dy"] if variant == "asteroids" else normalized["oxygen_dy"]
    codes = tuple(product(range(times), range(len(edges) + 1)))
    if variant == "oxygen":
        absent_dy = (len(edges) // 2 if normalized["type"] == "projected"
                     else min(1, len(edges)))
        codes += ((times, absent_dy),)
    return codes


class StateEncoder:
    """Map public variables to integer tuples and track tank availability.

    Collected tanks retain coordinates upstream. A separate absent code avoids
    treating such coordinates as a target. All equality boundaries use the
    higher bin. The specification is copied so checkpoint settings are stable.
    """
    def __init__(self, variant, state_spec=None):
        self.variant = variant
        self.spec = _spec(variant, state_spec)
        self.capacity = _capacity(variant, self.spec)
        self.reset()

    def reset(self):
        self.available = True
        self.previous_oxygen_x = None

    def _object(self, x, object_y, astronaut_y, edges, velocity=0.0, cap=0.0):
        tau = arrival_time(x)
        error = object_y - astronaut_y
        if self.spec["type"] == "projected":
            error -= velocity * min(tau, cap) * self.spec["projection_factor"]
        return (bisect_right(self.spec["object_time"], tau), bisect_right(edges, error))

    def encode(self, observation, info):
        values = dict(zip(info["variables"], map(float, observation)))
        y = values["astronaut_y"]
        state = (bisect_right(self.spec["height"], y),
                 bisect_right(self.spec["velocity"], values["astronaut_v"]))
        if self.variant == "fly":
            return state
        asteroid = self._object(values["asteroid_x"], values["asteroid_y"], y,
                                self.spec["asteroid_dy"], values["astronaut_v"],
                                self.spec.get("asteroid_projection_cap", 0.0))
        if self.variant == "asteroids":
            return state + asteroid
        x = values["oxygen_x"]
        if self.previous_oxygen_x is not None and x > self.previous_oxygen_x + 100.0:
            self.available = True
        if info.get("oxygen_picked") == "yes":
            self.available = False
        self.previous_oxygen_x = x
        if self.spec["type"] == "next_event":
            asteroid_next = not self.available or values["asteroid_x"] < x
            lookahead_absent = len(self.spec["lookahead_dy"]) + 1
            lookahead = (bisect_right(self.spec["lookahead_dy"], values["oxygen_y"] - y)
                         if self.available and asteroid_next else lookahead_absent)
            obj = (asteroid if asteroid_next else
                   self._object(x, values["oxygen_y"], y, self.spec["oxygen_dy"]))
            return state + (int(asteroid_next),) + obj + (lookahead,)
        absent = (len(self.spec["object_time"]) + 1,
                  len(self.spec["oxygen_dy"]) // 2 if self.spec["type"] == "projected"
                  else min(1, len(self.spec["oxygen_dy"])))
        oxygen = (self._object(x, values["oxygen_y"], y, self.spec["oxygen_dy"],
                              values["astronaut_v"],
                              self.spec.get("oxygen_projection_cap", 0.0))
                  if self.available else absent)
        return state + asteroid + oxygen
