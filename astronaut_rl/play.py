"""Replay a saved Q table in a browser, native window, or headlessly."""
import argparse
import json
from pathlib import Path
from .experiment import evaluate, evaluation_summary
from .qlearning import QLearner
from .environment import WindowClosed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path, nargs="?",
                   default=Path(__file__).resolve().parents[1] / "models/oxygen_seed11.json.gz")
    p.add_argument("--seed", type=int, default=9590000)
    p.add_argument("--episodes", type=int, default=1)
    display = p.add_mutually_exclusive_group()
    display.add_argument("--headless", action="store_true", help="Print evaluation data without a display")
    display.add_argument("--native", action="store_true", help="Use a native desktop window instead of the browser")
    p.add_argument("--port", type=int, default=8765, help="Local browser port (0 selects a free port)")
    p.add_argument("--no-open", action="store_true", help="Print the browser URL without opening it automatically")
    p.add_argument("--train", action="store_true", help="Learn from a fresh Q table, then fly autonomously")
    p.add_argument("--training-steps", type=int, default=None,
                   help="Environment steps per fresh run (default: frozen task configuration)")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    if not 0 <= args.port <= 65535:
        p.error("port must be between 0 and 65535")
    if args.training_steps is not None and args.training_steps < 2000:
        p.error("training-steps must be at least 2000")
    if args.train and (args.headless or args.native):
        p.error("--train uses the browser; use astronaut_rl.experiment for headless training")
    if not args.headless and not args.native:
        from .viewer import serve
        serve(args.model, args.seed, args.episodes, args.port, not args.no_open,
              train_first=args.train, training_budget=args.training_steps)
        return
    agent = QLearner.load(args.model)
    try:
        rows = evaluate(agent, agent.metadata["variant"],
                        range(args.seed, args.seed + args.episodes),
                        repeat=agent.metadata["repeat"], render=not args.headless)
    except WindowClosed:
        print("Replay stopped. Window closed or Escape pressed.")
        return
    print(json.dumps({"episodes": rows, "summary": evaluation_summary(rows)}, indent=2))


if __name__ == "__main__":
    main()
