"""Actual scene streaming, smooth cadence, unchanged policies, and fresh training."""
import json
import re
import subprocess
import sys
import time
import unittest
import urllib.request
import urllib.error
import shutil
from pathlib import Path


class ViewerTests(unittest.TestCase):
    def test_actual_scene_stream_assets_switch_train_restart_and_stop(self):
        root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen([sys.executable, "-u", "-m", "astronaut_rl.play",
                                    "models/fly_seed11.json.gz", "--seed", "9001000",
                                    "--port", "0", "--no-open", "--training-steps", "1000000"],
                                   cwd=root, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        created_run = None
        try:
            first = process.stdout.readline()
            self.assertIn("Astronaut replay: http://127.0.0.1:", first)
            url = first.split("Astronaut replay: ", 1)[1].strip()
            html = urllib.request.urlopen(url, timeout=5).read().decode()
            token = re.search(r'const token=("[^"]+")', html)
            self.assertIsNotNone(token)
            token = json.loads(token.group(1))
            self.assertNotIn("Play yourself", html)
            self.assertNotIn("keydown", html)
            self.assertIn("Train from scratch", html)
            self.assertIn("requestAnimationFrame", html)
            self.assertIn("new EventSource('/events')", html)
            self.assertNotIn("frame.png", html)

            def get_status():
                return json.load(urllib.request.urlopen(url + "status", timeout=5))

            def post(path, payload):
                request = urllib.request.Request(url + path, json.dumps(payload).encode(),
                           {"Content-Type": "application/json", "X-Astronaut-Token": token})
                return urllib.request.urlopen(request, timeout=5).read()

            def wait_variant(variant, previous_frame=0):
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    state = get_status()
                    if state["variant"] == variant and state["frame_id"] > previous_frame and state["frames"] > 0:
                        return state
                    time.sleep(.05)
                self.fail("No real physics frame for " + variant)

            state = wait_variant("fly")
            scene = json.load(urllib.request.urlopen(url + "scene", timeout=5))
            self.assertEqual(scene["variant"], "fly")
            self.assertGreater(scene["frames"], 0)
            self.assertEqual(scene["entities"][0]["id"], "astronaut")
            self.assertIn(scene["entities"][0]["sprite"],
                          ("astronaut_idle.png", "astronaut_thrust.png"))
            from astronaut_rl.viewer import ASSET_DIRECTORY, ASSETS
            for name in ASSETS:
                with urllib.request.urlopen(url + "assets/" + name, timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), "image/png")
                    self.assertEqual(response.read(), (ASSET_DIRECTORY / name).read_bytes())
            for unknown in ("assets/../environment.py", "assets/unknown.png", "frame.png"):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(url + unknown, timeout=5)
                self.assertEqual(error.exception.code, 404)

            # Read the real stream rather than infer animation quality from
            # status polling. The sequence should advance per physics frame,
            # even though Q decisions hold one action for six such frames.
            scenes, sizes, arrivals = [], [], []
            with urllib.request.urlopen(url + "events", timeout=5) as stream:
                self.assertEqual(stream.headers.get_content_type(), "text/event-stream")
                while len(scenes) < 36:
                    line = stream.readline()
                    if not line:
                        self.fail("Scene stream closed unexpectedly")
                    if line.startswith(b"data: "):
                        snapshot = json.loads(line[6:])
                        if snapshot and snapshot["frames"] > 0:
                            scenes.append(snapshot)
                            sizes.append(len(line))
                            arrivals.append(time.perf_counter())
            frames = [s["frames"] for s in scenes]
            self.assertTrue(all(b > a for a, b in zip(frames, frames[1:])))
            self.assertGreaterEqual(sum(b - a == 1 for a, b in zip(frames, frames[1:])), 30)
            fps = (len(scenes) - 1) / (arrivals[-1] - arrivals[0])
            self.assertGreater(fps, 42)
            self.assertLess(fps, 85)
            self.assertLess(sum(sizes) / len(sizes), 2000)
            print("Scene transport: %.1f frames/s; %.0f bytes/frame" %
                  (fps, sum(sizes) / len(sizes)), flush=True)
            post("start", {"variant": "oxygen", "seed": 9001000})
            state = wait_variant("oxygen", state["frame_id"])
            self.assertEqual(state["phase"], "Playing")
            post("start", {"variant": "oxygen", "seed": 9001000})
            wait_variant("oxygen", state["frame_id"])
            with self.assertRaises(urllib.error.HTTPError) as error:
                post("input", {"pressed": True})
            self.assertEqual(error.exception.code, 404)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post("start", {"variant": "fly", "seed": 9001000, "mode": "human"})
            self.assertEqual(error.exception.code, 400)
            post("train", {"variant": "fly", "seed": 9001000})
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                status = get_status()
                progress = status.get("learning") or {}
                if progress.get("phase") == "error":
                    self.fail(str(progress))
                if progress.get("phase") == "complete" and status["variant"] == "fly" and status["frames"] > 0:
                    created_run = Path(progress["model_path"]).parent.parent
                    break
                time.sleep(.05)
            else:
                self.fail("Fresh training did not automatically replay the new model")
            from astronaut_rl.qlearning import QLearner
            learned = QLearner.load(progress["model_path"])
            self.assertEqual(sum(sum(v) for v in learned.visits.values()), learned.metadata["updates"])
            self.assertTrue(learned.metadata["qualified"])
            self.assertEqual(learned.metadata["budget"], 1000000)
            self.assertEqual(learned.metadata["variant"], "fly")
            # Variant names alone do not prove the matching agent was loaded.
            # The test below also covers the model-selection restart regression.
            post("stop", {})
            process.wait(timeout=8)
            self.assertEqual(process.returncode, 0, process.stderr.read())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
            if created_run is not None:
                shutil.rmtree(created_run)

    def test_restarting_switched_variant_keeps_its_own_agent(self):
        from astronaut_rl.viewer import Replay
        replay = Replay("models/fly_seed11.json.gz", 9001000)
        try:
            replay.start("oxygen", 9001000)
            first = replay.agent.q
            self.assertEqual(replay.agent.metadata["variant"], "oxygen")
            replay.start("oxygen", 9001000)
            self.assertEqual(replay.agent.metadata["variant"], "oxygen")
            self.assertEqual(first, replay.agent.q)
            replay.start("fly", 9001000)
            self.assertEqual(replay.agent.metadata["variant"], "fly")
        finally:
            if replay.game is not None:
                replay.game.close()

    def test_browser_snapshots_match_headless_game_and_decision_cadence(self):
        from unittest.mock import patch
        from astronaut_rl.experiment import evaluate
        from astronaut_rl.viewer import Replay
        for variant in ("fly", "asteroids", "oxygen"):
            with self.subTest(variant=variant):
                replay = Replay("models/%s_seed11.json.gz" % variant, 9001000)
                expected = evaluate(replay.agent, variant, [9001000],
                                    repeat=replay.agent.metadata["repeat"])[0]
                before = {s: row[:] for s, row in replay.agent.q.items()}
                replay.start(variant, 9001000)
                try:
                    with patch.object(replay, "_pace"), patch.object(
                            replay.agent, "greedy", wraps=replay.agent.greedy) as choose:
                        while replay.running:
                            replay.advance()
                    self.assertEqual(replay.data["frames"], expected["frames"])
                    self.assertEqual(replay.data["oxygen"], expected["oxygen"])
                    self.assertAlmostEqual(replay.data["reward"], expected["return"])
                    self.assertEqual(replay.data["phase"],
                                     "Survived" if expected["survived"] else "Collision")
                    repeat = replay.agent.metadata["repeat"]
                    self.assertEqual(choose.call_count, (expected["frames"] + repeat - 1) // repeat)
                    self.assertEqual(replay.agent.q, before)
                    self.assertEqual(replay.scene["frames"], expected["frames"])
                    astronaut = replay.game.env.unwrapped._astronaut
                    entity = replay.scene["entities"][0]
                    self.assertAlmostEqual(entity["y"],
                        800 - astronaut["y"] - astronaut["anchor_y"], places=2)
                finally:
                    replay.game.close()

    def test_asteroid_id_follows_front_to_back_on_respawn(self):
        from unittest.mock import patch
        from astronaut_rl.viewer import Replay
        replay = Replay("models/asteroids_seed11.json.gz", 9001000)
        replay.start("asteroids", 9001000)
        try:
            original = next(e for e in replay.scene["entities"] if e["id"].startswith("asteroid-"))
            with patch.object(replay, "_pace"):
                while replay.running and replay.data["frames"] < 132:
                    replay.advance()
            current = {e["id"]: e for e in replay.scene["entities"]}
            self.assertIn(original["id"], current)
            self.assertLess(current[original["id"]]["x"], 300)
            self.assertEqual(current[original["id"]]["sprite"], original["sprite"])
            self.assertGreater(len([e for e in current if e.startswith("asteroid-")]), 1)
        finally:
            replay.game.close()


if __name__ == "__main__":
    unittest.main()
