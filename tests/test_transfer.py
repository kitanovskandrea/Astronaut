"""Tabular inheritance and physical-frame macro reward semantics."""
import tempfile
import unittest
from pathlib import Path

from astronaut_rl.config import load_config
from astronaut_rl.environment import ENV_COMMIT, Transition
from astronaut_rl.experiment import derive_gate, discounted_transition, gate_passed, validation_score
from astronaut_rl.qlearning import QLearner
from astronaut_rl.state import child_suffixes
from astronaut_rl.transfer import initialize_from_parent, inspect_parent, verify_initialization


class TransferTests(unittest.TestCase):
    def parent(self, variant):
        cfg = load_config(variant)
        parent = QLearner(alpha=cfg['alpha'], gamma=cfg['gamma'], seed=101)
        state = (2, 3) if variant == 'fly' else (2, 3, 1, 4)
        parent.q[state] = [-0.8, -0.2]
        parent.visits[state] = [9, 11]
        parent.metadata = {'variant': variant, 'state_spec': cfg['state_spec'], 'seed': 101,
            'repeat': cfg['repeat'], 'profile': 'official', 'environment_commit': ENV_COMMIT,
            'qualified': True, 'updates': 20, 'physics_frames': 120, 'model_role': 'selected'}
        return parent

    def test_every_child_state_gets_parent_values_but_not_visit_counts_or_aliases(self):
        for parent_variant, variant in (('fly', 'asteroids'), ('asteroids', 'oxygen')):
            with self.subTest(variant=variant):
                parent = self.parent(parent_variant)
                cfg = load_config(variant)
                child = QLearner(seed=101)
                count = initialize_from_parent(child, parent, variant, cfg['state_spec'])
                suffixes = child_suffixes(variant, cfg['state_spec'])
                self.assertEqual(count, len(suffixes))
                self.assertEqual(count, 40 if variant == 'asteroids' else 36)
                self.assertEqual(child.visits, {})
                verify_initialization(child, parent, variant, cfg['state_spec'])
                keys = list(child.q)
                self.assertEqual(child.q[keys[0]], [-0.8, -0.2])
                child.q[keys[0]][0] = -99
                self.assertEqual(child.q[keys[1]][0], -0.8)
                self.assertEqual(next(iter(parent.q.values()))[0], -0.8)
                self.assertEqual(sum(sum(row) for row in parent.visits.values()), 20)
                with self.assertRaises(ValueError):
                    verify_initialization(child, parent, variant, cfg['state_spec'])

    def test_incompatible_prefix_or_unqualified_parent_is_rejected(self):
        parent = self.parent('fly')
        cfg = load_config('asteroids')
        wrong = dict(cfg['state_spec'])
        wrong['height'] = [100, 200]
        with self.assertRaises(ValueError):
            initialize_from_parent(QLearner(), parent, 'asteroids', wrong)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fly.json.gz'
            parent.metadata['qualified'] = False
            parent.save(path)
            with self.assertRaisesRegex(ValueError, 'gate'):
                inspect_parent(path, cfg, 101, directory)
            parent.metadata['qualified'] = True
            parent.save(path)
            with self.assertRaisesRegex(ValueError, 'same independent'):
                inspect_parent(path, cfg, 202, directory)
            loaded, info = inspect_parent(path, cfg, 101, directory)
            self.assertEqual(loaded.q, parent.q)
            self.assertEqual(info['parent_model_path'], 'fly.json.gz')
            self.assertEqual(info['parent_updates'], 20)
            parent.metadata['model_role'] = 'final'
            parent.save(path)
            with self.assertRaisesRegex(ValueError, 'selected policy'):
                inspect_parent(path, cfg, 101, directory)

    def test_macro_reward_and_short_transition_discount_are_physical_frame_based(self):
        class FakeGame:
            repeat = 6
            def __init__(self, terminal): self.terminal = terminal
            def step(self, action, on_frame):
                for k, reward in enumerate((1.0, 0.0, -1.0), 1):
                    on_frame((), reward, self.terminal and k == 3,
                             not self.terminal and k == 3, {}, k)
                return Transition((1,), 0.0, self.terminal, not self.terminal, 3, {})
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                transition, reward = discounted_transition(FakeGame(terminal), 1, 0.9)
                self.assertAlmostEqual(reward, 1 - 0.9 ** 2)
                agent = QLearner(alpha=1, gamma=0.9 ** 6)
                agent.q[(1,)] = [4.0, 1.0]
                agent.update((0,), 1, reward, transition.state, transition.terminated,
                             discount=0.9 ** transition.frames)
                expected = reward if terminal else reward + 0.9 ** 3 * 4
                self.assertAlmostEqual(agent.q[(0,)][1], expected)

    def test_oxygen_selection_keeps_safety_before_tank_count_and_requires_improvement(self):
        cfg = load_config('oxygen')
        baseline = {'survival_rate': 0.98, 'oxygen_mean': 2.9}
        gate = derive_gate(cfg, baseline)
        self.assertAlmostEqual(gate['minimum_survival_rate'], 0.93)
        self.assertAlmostEqual(gate['minimum_oxygen_mean'], 3.15)
        unsafe = {'survival_rate': 0.9, 'oxygen_mean': 20, 'frames_mean': 1100}
        safe = {'survival_rate': 0.96, 'oxygen_mean': 3.2, 'frames_mean': 1150}
        self.assertGreater(validation_score('oxygen', safe, gate), validation_score('oxygen', unsafe, gate))
        self.assertTrue(gate_passed(safe, gate))
        self.assertFalse(gate_passed(baseline, gate))
        self.assertFalse(gate_passed(unsafe, gate))


if __name__ == '__main__': unittest.main()
