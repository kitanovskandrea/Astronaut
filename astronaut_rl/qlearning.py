"""Original, sparse tabular Q-learning using only the Python standard library.

Rows are created by updates or explicitly initialized from a learned parent.
Visit counters count only this stage's updates. Action selection reads absent
states as all-zero rows without inserting them, so evaluation cannot grow or
otherwise change the learned Q table.
"""

import gzip
import json
import math
from pathlib import Path
import random


_FORMAT = "astronaut-tabular-q-v1"


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(value):
        raise ValueError("%s must be a finite number" % name)
    return float(value)


def _probability(value, name):
    value = _finite_number(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError("%s must be between 0 and 1" % name)
    return value


def _state(state):
    """Accept flat tuples of JSON scalars, including discretized integers."""
    if not isinstance(state, tuple):
        raise ValueError("state must be a tuple of JSON scalar values")
    for value in state:
        if value is None or isinstance(value, (str, bool, int)):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        raise ValueError("state values must be finite JSON scalars")
    return state


class QLearner:
    """One-step off-policy Q-learning with epsilon-greedy exploration.

    ``update`` returns the signed TD residual *before* changing the table.
    Squaring that residual provides a diagnostic TD loss; Q-learning itself
    uses the signed residual, not gradient descent on a fitted loss model.

    ``terminated`` means a true terminal state.  A caller must pass False for
    a time-limit truncation so its final observation can still bootstrap.
    ``discount`` can override gamma for a transition spanning several frames.
    """

    def __init__(self, actions=2, alpha=0.15, gamma=0.99, seed=0):
        if isinstance(actions, bool) or not isinstance(actions, int) or actions < 1:
            raise ValueError("actions must be a positive integer")
        self.actions = actions
        self.alpha = _probability(alpha, "alpha")
        if self.alpha == 0.0:
            raise ValueError("alpha must be greater than 0")
        self.gamma = _probability(gamma, "gamma")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.rng = random.Random(seed)
        self.q = {}
        self.visits = {}
        self.metadata = {}
        self._zeros = (0.0,) * actions

    def greedy(self, state):
        """Select the first maximizing action, without consuming randomness."""
        values = self.q.get(_state(state), self._zeros)
        return max(range(self.actions), key=values.__getitem__)

    def choose(self, state, epsilon):
        """Select a training action; exploration and ties use the seeded RNG."""
        state = _state(state)
        epsilon = _probability(epsilon, "epsilon")
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.actions)
        values = self.q.get(state, self._zeros)
        best = max(values)
        return self.rng.choice([a for a, value in enumerate(values) if value == best])

    def update(self, state, action, reward, next_state, terminated, discount=None):
        """Apply Q(s,a) += alpha * (r + discount * max Q(s',.) - Q(s,a))."""
        state = _state(state)
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer")
        if not 0 <= action < self.actions:
            raise ValueError("action is outside the action space")
        reward = _finite_number(reward, "reward")
        if not isinstance(terminated, bool):
            raise ValueError("terminated must be a boolean")
        factor = self.gamma if discount is None else _probability(discount, "discount")
        bootstrap = 0.0
        if not terminated:
            next_state = _state(next_state)
            bootstrap = max(self.q.get(next_state, self._zeros))
        # Compute the target before inserting/updating the current row. This
        # also handles self-transitions using the original Q values.
        values = self.q.get(state, self._zeros)
        residual = reward + factor * bootstrap - values[action]
        new_value = values[action] + self.alpha * residual
        if not math.isfinite(residual) or not math.isfinite(new_value):
            raise ValueError("transition produced a non-finite Q value")
        if state not in self.q:
            self.q[state] = [0.0] * self.actions
        if state not in self.visits:
            self.visits[state] = [0] * self.actions
        self.q[state][action] = new_value
        self.visits[state][action] += 1
        return residual

    def save(self, path, metadata=None):
        """Write a sparse JSON checkpoint; a .gz suffix enables gzip.

        Metadata is stored for reproducibility and exposed on a loaded agent.
        Random-generator state is deliberately not a training-resume contract.
        """
        payload = {
            "format": _FORMAT,
            "actions": self.actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "seed": self.seed,
            "metadata": self.metadata if metadata is None else metadata,
            "rows": [
                {"state": list(_state(state)), "q": values,
                 "visits": self.visits.get(state, [0] * self.actions)}
                for state, values in self.q.items()
            ],
        }
        # Validate serialization before opening the destination file.
        serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        path = Path(path)
        if path.suffix == ".gz":
            # A gzip header normally contains both the destination filename and
            # the current time. Suppress them so the same learned table yields
            # the same bytes and model hash on repeated runs.
            with path.open("wb") as handle:
                with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                    compressed.write(serialized.encode("utf-8"))
        else:
            path.write_text(serialized, encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Read and validate a checkpoint written by :meth:`save`."""
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise ValueError("unsupported Q-learning checkpoint format")
        try:
            learner = cls(payload["actions"], payload["alpha"], payload["gamma"],
                          payload.get("seed", 0))
            rows = payload["rows"]
            if not isinstance(rows, list):
                raise ValueError("checkpoint rows must be a list")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("state"), list):
                    raise ValueError("checkpoint row must contain a state list")
                state = _state(tuple(row["state"]))
                if state in learner.q:
                    raise ValueError("duplicate state in checkpoint")
                values = row["q"]
                visits = row.get("visits", [0] * learner.actions)
                if not isinstance(values, list) or len(values) != learner.actions:
                    raise ValueError("checkpoint Q row has wrong number of actions")
                if not isinstance(visits, list) or len(visits) != learner.actions:
                    raise ValueError("checkpoint visit row has wrong number of actions")
                if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in visits):
                    raise ValueError("checkpoint visits must be nonnegative integers")
                learner.q[state] = [_finite_number(v, "Q value") for v in values]
                learner.visits[state] = visits[:]
            learner.metadata = payload.get("metadata", {})
        except (KeyError, TypeError) as error:
            raise ValueError("malformed Q-learning checkpoint") from error
        return learner

