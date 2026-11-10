"""Run genuine Q-learning in a separate process with isolated environments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from .config import load_config


class TrainingJob:
    def __init__(self, variant, seed, budget=None, run_parent=None, profile="official"):
        settings = load_config(variant)
        self.variant, self.seed = variant, seed
        self.budget = settings['steps'] if budget is None else budget
        project = Path(__file__).resolve().parents[1]
        parent = Path(run_parent) if run_parent is not None else project / "runs"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=variant + "_", dir=parent))
        self.progress_path = self.directory / "progress.json"
        self.model_path = self.directory / "models" / f"{variant}_seed{seed}.json.gz"
        self.log_path = self.directory / "training.log"
        command = [sys.executable, "-u", "-m", "astronaut_rl.experiment",
                   "--curriculum-to", variant, "--seeds", str(seed),
                   "--out", str(self.directory / "results"),
                   "--models", str(self.directory / "models"),
                   "--progress", str(self.progress_path), "--profile", profile]
        if budget is not None:
            command += ["--steps", str(budget)]
        with self.log_path.open("w") as log:
            self.process = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
        self.last = {"phase": "training", "decision": 0, "budget": self.budget,
                     "episodes": 0, "states": 0,
                     "epsilon": settings['epsilon_start'], "td_mse": None}

    def snapshot(self):
        try:
            self.last = json.loads(self.progress_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        code = self.process.poll()
        if code is not None and code != 0:
            if self.last.get('phase') in ('blocked', 'failed_gate'):
                self.last['phase'] = 'blocked'
                self.last['reason'] = self.last.get('reason') or self.last.get('error')
                return dict(self.last)
            return {**self.last, "phase": "error", "error": self.log_path.read_text()[-1600:]}
        if code == 0 and self.last.get("phase") != "complete":
            return {**self.last, "phase": "error", "error": "Training ended without a completed checkpoint."}
        return dict(self.last)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
