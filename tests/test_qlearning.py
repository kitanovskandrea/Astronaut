"""Numerical and reproducibility checks for the original Q learner."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from astronaut_rl.qlearning import QLearner, linear_epsilon


class QLearningTests(unittest.TestCase):
    def test_exact_bellman_update(self):
        agent = QLearner(alpha=0.25, gamma=0.9)
        agent.q[(1,)] = [4.0, 2.0]
        agent.q[(2,)] = [3.0, 8.0]
        residual = agent.update((1,), 1, 2.0, (2,), False)
        self.assertAlmostEqual(residual, 7.2)
        self.assertAlmostEqual(agent.q[(1,)][1], 3.8)
        self.assertEqual(agent.q[(1,)][0], 4.0)
        self.assertEqual(agent.visits[(1,)], [0, 1])

    def test_terminal_and_time_limit_bootstrap_are_different(self):
        terminal = QLearner(alpha=1.0, gamma=0.9)
        truncated = QLearner(alpha=1.0, gamma=0.9)
        for agent in (terminal, truncated):
            agent.q[(2,)] = [10.0, 0.0]
        terminal.update((1,), 0, 1.0, None, True)
        truncated.update((1,), 0, 1.0, (2,), False)
        self.assertEqual(terminal.q[(1,)][0], 1.0)
        self.assertEqual(truncated.q[(1,)][0], 10.0)

    def test_discount_override_and_self_transition(self):
        agent = QLearner(alpha=0.5, gamma=0.9)
        agent.q[(1,)] = [4.0, 2.0]
        residual = agent.update((1,), 0, 1.0, (1,), False, discount=0.25)
        self.assertEqual(residual, -2.0)
        self.assertEqual(agent.q[(1,)], [3.0, 2.0])

    def test_seeded_exploration_and_random_ties(self):
        first = QLearner(actions=3, seed=82)
        second = QLearner(actions=3, seed=82)
        sequence = [first.choose((0,), epsilon=0.3) for _ in range(100)]
        self.assertEqual(sequence, [second.choose((0,), 0.3) for _ in range(100)])
        self.assertEqual(set(sequence), {0, 1, 2})
        ties = QLearner(actions=3, seed=11)
        self.assertEqual({ties.choose((0,), 0.0) for _ in range(100)}, {0, 1, 2})

    def test_evaluation_and_unknown_states_do_not_mutate_table_or_rng(self):
        agent = QLearner(actions=3, seed=14)
        agent.q[(1,)] = [1.0, 3.0, 3.0]
        before = copy.deepcopy(agent.q)
        rng_before = agent.rng.getstate()
        for _ in range(20):
            self.assertEqual(agent.greedy((1,)), 1)
            self.assertEqual(agent.greedy((99,)), 0)
        self.assertEqual(agent.q, before)
        self.assertEqual(agent.rng.getstate(), rng_before)
        agent.choose((88,), 0.5)
        self.assertEqual(agent.q, before)
        self.assertEqual(agent.visits, {})

    def test_update_stores_only_current_state(self):
        agent = QLearner()
        agent.update((1,), 0, 2.0, (99,), False)
        self.assertEqual(set(agent.q), {(1,)})
        self.assertEqual(set(agent.visits), {(1,)})

    def test_json_and_gzip_checkpoint_roundtrip(self):
        agent = QLearner(actions=3, alpha=0.2, gamma=0.8, seed=7)
        agent.update((1, -2, 0), 2, 5.0, (2, 0, 1), False)
        agent.update((1, -2, 0), 1, -2.0, None, True)
        metadata = {"variant": "asteroids", "episodes": 2}
        with tempfile.TemporaryDirectory() as directory:
            for name in ("table.json", "table.json.gz"):
                path = Path(directory) / name
                agent.save(path, metadata=metadata)
                restored = QLearner.load(path)
                self.assertEqual(restored.q, agent.q)
                self.assertEqual(restored.visits, agent.visits)
                self.assertEqual(restored.metadata, metadata)
                self.assertEqual(restored.actions, 3)
                self.assertEqual(restored.alpha, 0.2)
                self.assertEqual(restored.gamma, 0.8)
                self.assertEqual(restored.greedy((1, -2, 0)), agent.greedy((1, -2, 0)))
                self.assertEqual(set(restored.q), {(1, -2, 0)})

    def test_invalid_parameters_and_transitions(self):
        for kwargs in ({"actions": 0}, {"actions": True}, {"alpha": 0},
                       {"alpha": float("nan")}, {"gamma": 1.1}, {"seed": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QLearner(**kwargs)
        agent = QLearner()
        for action, reward, state, terminal in ((2, 1, (2,), False),
                                                (True, 1, (2,), False),
                                                (0, float("inf"), (2,), False),
                                                (0, 1, None, False),
                                                (0, 1, (2,), 1)):
            with self.assertRaises(ValueError):
                agent.update((1,), action, reward, state, terminal)
        self.assertEqual(agent.q, {})
        with self.assertRaises(ValueError):
            agent.choose((0,), -0.1)

    def test_gzip_checkpoint_is_independent_of_filename_and_save_time(self):
        agent = QLearner(seed=47)
        agent.update((1, 2), 1, 3.0, (2, 3), False)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json.gz"
            second = Path(directory) / "different_name.json.gz"
            with mock.patch("gzip.time.time", return_value=1000000000):
                agent.save(first, metadata={"seed": 47})
            with mock.patch("gzip.time.time", return_value=1900000000):
                agent.save(second, metadata={"seed": 47})
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(QLearner.load(first).q, agent.q)
            self.assertEqual(QLearner.load(second).metadata, {"seed": 47})

    def test_invalid_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            payload = {"format": "astronaut-tabular-q-v1", "actions": 2,
                       "alpha": 0.1, "gamma": 0.9,
                       "rows": [{"state": [0], "q": [1.0]}]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                QLearner.load(path)

    def test_epsilon_schedule(self):
        self.assertEqual(linear_epsilon(0, 10), 1.0)
        self.assertAlmostEqual(linear_epsilon(5, 10), 0.525)
        self.assertAlmostEqual(linear_epsilon(10, 10), 0.05)
        self.assertAlmostEqual(linear_epsilon(100, 10), 0.05)


if __name__ == "__main__":
    unittest.main()
