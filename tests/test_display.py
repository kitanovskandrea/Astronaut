"""Exercise the actual SDL event queue in isolated offscreen processes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class DisplayTests(unittest.TestCase):
    def run_display(self, source):
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
        result = subprocess.run([sys.executable, "-c", source],
                                cwd=Path(__file__).resolve().parents[1],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_rendered_frames_drain_real_events_in_all_variants(self):
        self.run_display('''
from astronaut_rl.environment import Game, VARIANTS, course_display as display
for variant in VARIANTS:
    game = Game(variant, repeat=1, render=True)
    try:
        game.reset(9001000)
        game.step(0)
        display.event.post(display.event.Event(display.USEREVENT))
        game.step(0)
        assert not display.event.peek(display.USEREVENT), variant
    finally:
        game.close()
assert not display.display.get_init()
''')

    def check_exit(self, event_expression):
        output = self.run_display('''
import sys
from unittest.mock import patch
from astronaut_rl.environment import course_display as display
from astronaut_rl.play import main
original = display.event.get
def request_exit():
    display.event.post(''' + event_expression + ''')
    return original()
sys.argv = ["play", "models/fly_seed11.json.gz", "--seed", "9001000", "--native"]
with patch.object(display.event, "get", side_effect=request_exit):
    main()
assert not display.display.get_init(), "Replay must close graphics on user exit"
''')
        self.assertIn("Replay stopped.", output)

    def test_close_button_stops_replay_cleanly(self):
        self.check_exit("display.event.Event(display.QUIT)")

    def test_escape_stops_replay_cleanly(self):
        self.check_exit("display.event.Event(display.KEYDOWN, key=display.K_ESCAPE)")

    def test_headless_game_does_not_touch_display_events(self):
        from unittest.mock import patch
        from astronaut_rl.environment import Game, course_display
        game = Game("fly", render=False)
        try:
            game.reset(9001000)
            with patch.object(course_display.event, "get", side_effect=AssertionError("headless event access")):
                game.step(0)
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
