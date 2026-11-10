"""Public-state contracts, threshold boundaries, ranges and tank history."""
import json
from pathlib import Path
import random
import unittest
from bisect import bisect_right

from astronaut_rl.state import (ABSENT_OXYGEN, DY_EDGES, STATE_CAPACITIES,
                                TIME_EDGES, V_EDGES, Y_EDGES, StateEncoder,
                                arrival_time, child_suffixes, default_spec, projected_spec, state_capacity)


def encode(encoder, y=400.0, velocity=0.0, asteroid_x=800.0,
           asteroid_y=550.0, oxygen_x=1300.0, oxygen_y=300.0,
           picked="no", reverse=False):
    values = {"astronaut_y": y, "astronaut_v": velocity}
    if encoder.variant != "fly":
        values.update(asteroid_x=asteroid_x, asteroid_y=asteroid_y)
    if encoder.variant == "oxygen":
        values.update(oxygen_x=oxygen_x, oxygen_y=oxygen_y)
    names = list(values)
    if reverse:
        names.reverse()
    return encoder.encode([values[name] for name in names],
                          {"variables": names, "oxygen_picked": picked})


class ReferenceProjectedEncoder:
    """Frozen independent pilot formula for integration trace comparisons."""
    def __init__(self, variant):
        self.variant = variant
        self.available = True
        self.previous_x = None

    def encode(self, observation, info):
        data = dict(zip(info["variables"], map(float, observation)))
        y, velocity = data["astronaut_y"], data["astronaut_v"]
        result = (bisect_right([160, 230, 310, 400, 490, 570, 640], y),
                  bisect_right([-300, -180, -90, -30, 30, 90, 180, 300], velocity))
        if self.variant == "fly":
            return result
        tau = max(0, (data["asteroid_x"] - 300) / 500)
        error = data["asteroid_y"] - y - velocity * min(tau, .75)
        result += (bisect_right([.18, .4, .7, 1.2], tau),
                   bisect_right([-240, -160, -100, 0, 100, 160, 240], error))
        if self.variant == "asteroids":
            return result
        x = data["oxygen_x"]
        if self.previous_x is not None and x > self.previous_x + 100:
            self.available = True
        if info["oxygen_picked"] == "yes":
            self.available = False
        self.previous_x = x
        tau = max(0, (x - 300) / 500)
        error = data["oxygen_y"] - y - velocity * min(tau, .6)
        return result + ((bisect_right([.18, .4, .7, 1.2], tau),
                          bisect_right([-220, -120, -50, 50, 120, 220], error))
                         if self.available else (5, 3))


class StateTests(unittest.TestCase):
    def test_projected_public_observation_trace_matches_independent_pilot_formula(self):
        # Public states, frame-by-frame, including pickups and respawns. This
        # check does not depend on a scratch directory or a trained artifact.
        from astronaut_rl.environment import Game
        for variant in ("fly", "asteroids", "oxygen"):
            game = Game(variant, repeat=6, state_spec=projected_spec(variant))
            reference = ReferenceProjectedEncoder(variant)
            try:
                obs, info = game.env.reset(seed=303050)
                game.encoder.reset()
                state = game.encoder.encode(obs, info)
                expected = reference.encode(obs, info)
                self.assertEqual(state, expected)
                rng = random.Random(1207)
                for _ in range(200):
                    latest = []
                    transition = game.step(rng.randrange(2), on_frame=lambda observation, reward,
                                           terminated, truncated, frame_info, frame:
                                           latest.append(reference.encode(observation, frame_info)))
                    self.assertEqual(transition.state, latest[-1])
                    if transition.terminated or transition.truncated:
                        break
            finally:
                game.close()

    def test_saved_projected_q_greedy_traces_match_independent_pilot_formula(self):
        # During development these files may not exist yet. Submission models
        # contain exact stored specs; the finished archive exercises all stages.
        from astronaut_rl.environment import Game
        from astronaut_rl.qlearning import QLearner
        directory = Path(__file__).resolve().parents[1] / "models"
        agents = {}
        for variant in ("fly", "asteroids", "oxygen"):
            for path in sorted(directory.glob(variant + "_seed*.json.gz")):
                agent = QLearner.load(path)
                stored = agent.metadata.get("state_spec", {})
                if stored.get("type") == "projected" and StateEncoder(variant, stored).spec == StateEncoder(variant, projected_spec(variant)).spec:
                    agents[variant] = agent
                    break
        if len(agents) != 3:
            self.skipTest("All three saved projected curriculum policies have not been produced yet")
        for variant, agent in agents.items():
            game = Game(variant, repeat=agent.metadata["repeat"], gamma=.9995,
                        state_spec=agent.metadata["state_spec"])
            reference = ReferenceProjectedEncoder(variant)
            before_rng = agent.rng.getstate()
            try:
                obs, info = game.env.reset(seed=303050)
                game.encoder.reset()
                state = game.encoder.encode(obs, info)
                expected = reference.encode(obs, info)
                while True:
                    self.assertEqual(state, expected)
                    self.assertEqual(agent.greedy(state), agent.greedy(expected))
                    latest = []
                    transition = game.step(agent.greedy(expected), on_frame=lambda observation,
                                           reward, terminated, truncated, frame_info, frame:
                                           latest.append(reference.encode(observation, frame_info)))
                    state, expected = transition.state, latest[-1]
                    if transition.terminated or transition.truncated:
                        self.assertEqual(state, expected)
                        break
            finally:
                game.close()
            self.assertEqual(agent.rng.getstate(), before_rng)

    def test_projected_capacity_and_parent_prefix_are_preserved(self):
        states = {}
        for variant, capacity in (("fly", 72), ("asteroids", 2880), ("oxygen", 103680)):
            encoder = StateEncoder(variant, projected_spec(variant))
            self.assertEqual(encoder.capacity, capacity)
            states[variant] = encode(encoder, velocity=100, asteroid_x=550, oxygen_x=800)
        self.assertEqual(states["asteroids"][:2], states["fly"])
        self.assertEqual(states["oxygen"][:4], states["asteroids"])

    def test_projection_caps_and_zero_factor_have_explicit_expected_bins(self):
        spec = projected_spec("oxygen")
        # At y400/v100: asteroid tau2 is capped at.75, projected error75;
        # oxygen tau2 is capped at.6, projected error40.
        state = encode(StateEncoder("oxygen", spec), velocity=100,
                       asteroid_x=1300, asteroid_y=550, oxygen_x=1300, oxygen_y=500)
        self.assertEqual(state, (4, 6, 4, 4, 4, 3))
        spec["projection_factor"] = 0
        unprojected = encode(StateEncoder("oxygen", spec), velocity=100,
                             asteroid_x=1300, asteroid_y=550, oxygen_x=1300, oxygen_y=500)
        self.assertEqual(unprojected, (4, 6, 4, 5, 4, 4))

    def test_projected_absent_tank_code_matches_curriculum_transfer_suffix(self):
        encoder = StateEncoder("oxygen", projected_spec("oxygen"))
        self.assertEqual(encode(encoder, oxygen_x=333, picked="yes")[-2:], (5, 3))
        self.assertEqual(encode(encoder, oxygen_x=325)[-2:], (5, 3))
        self.assertNotEqual(encode(encoder, oxygen_x=1300)[-2:], (5, 3))
        asteroid_codes = child_suffixes("asteroids", projected_spec("asteroids"))
        oxygen_codes = child_suffixes("oxygen", projected_spec("oxygen"))
        self.assertEqual(len(asteroid_codes), 40)
        self.assertEqual(len(oxygen_codes), 36)
        self.assertEqual(len(set(oxygen_codes)), 36)
        self.assertEqual(oxygen_codes[-1], (5, 3))

    def test_projection_parameters_are_validated_and_copied(self):
        for name in ("asteroid_projection_cap", "oxygen_projection_cap", "projection_factor"):
            for bad in (True, -1, float("nan"), float("inf"), "bad"):
                spec = projected_spec("oxygen")
                spec[name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    StateEncoder("oxygen", spec)

    def test_variant_dimensions_public_mapping_and_integer_keys(self):
        for variant, size in (("fly", 2), ("asteroids", 4), ("oxygen", 6)):
            with self.subTest(variant=variant):
                state = encode(StateEncoder(variant))
                self.assertEqual(len(state), size)
                self.assertTrue(all(type(value) is int for value in state))
                self.assertEqual(state, encode(StateEncoder(variant), reverse=True))
                self.assertEqual(len({state: "row"}), 1)

    def test_exact_boundaries_belong_to_higher_bin(self):
        for index, edge in enumerate(Y_EDGES):
            self.assertEqual(encode(StateEncoder("fly"), y=edge)[0], index + 1)
        for index, edge in enumerate(V_EDGES):
            self.assertEqual(encode(StateEncoder("fly"), velocity=edge)[1], index + 1)
        for index, edge in enumerate(TIME_EDGES):
            state = encode(StateEncoder("asteroids"), asteroid_x=300 + 500 * edge)
            self.assertEqual(state[2], index + 1)
        for index, edge in enumerate(DY_EDGES):
            state = encode(StateEncoder("asteroids"), asteroid_y=400 + edge)
            self.assertEqual(state[3], index + 1)

    def test_signed_motion_object_offset_and_arrival_categories(self):
        negative = encode(StateEncoder("asteroids"), velocity=-100, asteroid_y=200,
                          asteroid_x=350)
        positive = encode(StateEncoder("asteroids"), velocity=100, asteroid_y=600,
                          asteroid_x=1300)
        self.assertLess(negative[1], positive[1])
        self.assertLess(negative[2], positive[2])
        self.assertLess(negative[3], positive[3])
        self.assertEqual(arrival_time(100), 0.0)
        self.assertEqual(arrival_time(550), 0.5)

    def test_outlying_finite_observations_stay_in_bin_ranges(self):
        rng = random.Random(412)
        for variant in ("fly", "asteroids", "oxygen"):
            for _ in range(100):
                state = encode(StateEncoder(variant),
                               **dict(zip(("y", "velocity", "asteroid_x", "asteroid_y",
                                           "oxygen_x", "oxygen_y"),
                                          (rng.uniform(-100000, 100000) for _ in range(6)))))
                limits = (6, 5) + ((4, 3) if variant != "fly" else ())
                if variant == "oxygen":
                    limits += (4, 3)
                self.assertTrue(all(0 <= value < limit for value, limit in zip(state, limits)))

    def test_pickup_history_respawn_and_reset(self):
        encoder = StateEncoder("oxygen")
        initial = encode(encoder, oxygen_x=340)
        self.assertNotEqual(initial[-2:], ABSENT_OXYGEN)
        self.assertEqual(encode(encoder, oxygen_x=333, picked="yes")[-2:], ABSENT_OXYGEN)
        self.assertEqual(encode(encoder, oxygen_x=325)[-2:], ABSENT_OXYGEN)
        self.assertNotEqual(encode(encoder, oxygen_x=1300)[-2:], ABSENT_OXYGEN)
        encode(encoder, oxygen_x=333, picked="yes")
        encoder.reset()
        self.assertEqual(encode(encoder, oxygen_x=340), initial)

    def test_spec_records_capacities_without_sharing_mutable_data(self):
        self.assertEqual(STATE_CAPACITIES, {"fly": 6 * 5,
                                          "asteroids": 6 * 5 * 4 * 3,
                                          "oxygen": 6 * 5 * 4 * 3 * (4 * 3 + 1)})
        spec = default_spec("oxygen")
        self.assertEqual(json.loads(json.dumps(spec)), spec)
        encoder = StateEncoder("oxygen", spec)
        spec["height"].clear()
        self.assertEqual(len(encoder.spec["height"]), 5)
        self.assertEqual(len(default_spec("oxygen")["height"]), 5)
        self.assertEqual(encoder.capacity, state_capacity("oxygen", encoder.spec))

    def test_next_event_configuration_uses_only_public_history(self):
        spec = default_spec("oxygen")
        spec.update(type="next_event", lookahead_dy=[-90, 90])
        encoder = StateEncoder("oxygen", spec)
        asteroid_next = encode(encoder, asteroid_x=800, oxygen_x=1300)
        self.assertEqual(asteroid_next[2], 1)
        self.assertLess(asteroid_next[-1], 3)
        oxygen_next = encode(encoder, asteroid_x=1300, oxygen_x=800)
        self.assertEqual(oxygen_next[2], 0)
        self.assertEqual(oxygen_next[-1], 3)
        collected = encode(encoder, asteroid_x=1300, oxygen_x=333, picked="yes")
        self.assertEqual(collected[2], 1)
        self.assertEqual(collected[-1], 3)
        self.assertEqual(encoder.capacity, 30 * (12 * 4 + 12))

    def test_invalid_threshold_specifications_are_rejected(self):
        for edges in ([0, 0], [10, -10], [float("nan")], [True], "bad"):
            spec = default_spec("fly")
            spec["velocity"] = edges
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                StateEncoder("fly", spec)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            StateEncoder("unknown")


if __name__ == "__main__":
    unittest.main()
