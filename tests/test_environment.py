"""Integration checks against the exact unmodified course snapshot."""
import unittest
from unittest.mock import patch
from astronaut_rl.environment import Game, VARIANTS, check_environment
from astronaut_rl.qlearning import QLearner


class EnvironmentTests(unittest.TestCase):
    def test_source_snapshot_is_checked_even_when_version_label_is_reused(self):
        self.assertEqual(check_environment(), '0db0652ad3b8cece05d476c56637a3dcb56557cd')

    def test_official_options_and_default_rewards(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                game=Game(variant)
                try:
                    game.reset(101)
                    options=game.env.unwrapped._options
                    self.assertEqual({k:options[k] for k in ('speed','gravity','thrust','max_steps','reward_function')},
                                     {'speed':500,'gravity':400,'thrust':1000,'max_steps':1200,'reward_function':None})
                    step=game.step(0)
                    self.assertEqual(step.reward,0)
                    while not step.terminated and not step.truncated:
                        step=game.step(0)
                    self.assertTrue(step.terminated)
                    self.assertEqual(step.reward,-1)
                finally: game.close()

    def test_new_official_net_acceleration_and_observation_variables(self):
        for variant,n in [('fly',2),('asteroids',4),('oxygen',6)]:
            game=Game(variant)
            try:
                game.reset(101)
                obs,reward,term,trunc,info=game.env.step(1)
                values=dict(zip(info['variables'],map(float,obs)))
                self.assertEqual(len(obs),n)
                self.assertAlmostEqual(values['astronaut_v'],10.0) # +600 / 60
                self.assertAlmostEqual(values['astronaut_y'],400+10/60,places=4)
                self.assertEqual(reward,0)
            finally: game.close()

    def test_seeded_resets_repeat_and_none_advances_private_rng(self):
        for variant in ('asteroids','oxygen'):
            game=Game(variant)
            try:
                traces=[]
                for seed in (5,None,5,None):
                    state=game.reset(seed)
                    obs,_,_,_,info=game.env.step(0)
                    traces.append((state,tuple(map(float,obs))))
                self.assertEqual(traces[0],traces[2])
                self.assertEqual(traces[1],traces[3])
                self.assertNotEqual(traces[0][1][3:],traces[1][1][3:])
            finally: game.close()

    def test_separate_evaluation_environment_does_not_change_training_rng(self):
        def trajectory(with_other):
            game=Game('oxygen')
            try:
                game.reset(51)
                result=[]
                for i in range(20):
                    if with_other and i==7:
                        other=Game('oxygen', repeat=1)
                        try:
                            other.reset(2000)
                            for _ in range(60): other.step(0)
                        finally: other.close()
                    result.append(game.reset(None))
                return result
            finally: game.close()
        self.assertEqual(trajectory(False),trajectory(True))

    def test_per_frame_callback_preserves_rewards_and_transitions(self):
        for variant in VARIANTS:
            game=Game(variant,repeat=6)
            try:
                game.reset(7); seen=[]
                a=game.step(1,on_frame=lambda obs,r,t,x,info,i:seen.append((r,info['steps'],i)))
                game.reset(7); b=game.step(1)
                self.assertEqual(a,b)
                self.assertEqual([s[2] for s in seen],list(range(1,7)))
                self.assertAlmostEqual(sum(s[0] for s in seen),a.reward)
            finally: game.close()

    def test_collision_stops_repeat_and_has_no_survival_bonus(self):
        game=Game('fly',repeat=1200)
        try:
            game.reset(3); t=game.step(0)
            self.assertTrue(t.terminated)
            self.assertFalse(t.truncated)
            self.assertLess(t.frames,1200)
            self.assertEqual(t.reward,-1)
            self.assertEqual(t.info['collision'],'ground')
        finally: game.close()

    def test_actual_pickup_gives_one_and_has_public_availability_event(self):
        # Test-only level control, never used for learning or evaluation policies.
        found=False
        for seed in range(20):
            game=Game('oxygen')
            try:
                game.reset(seed); y,v=400.,0.
                for _ in range(210):
                    action=int(v < .6*(400-y)-2)
                    obs,r,term,trunc,info=game.env.step(action)
                    values=dict(zip(info['variables'],map(float,obs)))
                    y,v=values['astronaut_y'],values['astronaut_v']
                    if info['oxygen_picked']=='yes':
                        self.assertEqual(r,-1 if term else 1)
                        found=True; break
                    if term or trunc: break
            finally: game.close()
            if found: break
        self.assertTrue(found,'No real oxygen pickup observed in course fixture')

if __name__=='__main__': unittest.main()
