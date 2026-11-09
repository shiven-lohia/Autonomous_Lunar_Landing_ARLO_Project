"""Run a full comparison or an individual agent."""

import argparse
from pathlib import Path

from garl_lander.config import load_config
from garl_lander.experiment import AGENTS, run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--agent", choices=AGENTS, help="Default: all five agents")
    parser.add_argument("--output", type=Path, default=Path("results/raw"))
    parser.add_argument("--resume", type=Path, help="Reuse completed runs; archive and restart interrupted agents")
    parser.add_argument("--workers", type=int, default=1, help="Independent seed processes; each uses configured Torch threads")
    args = parser.parse_args()
    if args.agent == "ppo_short":
        parser.error("Run the suite to derive PPO-short's budget from the hybrid")
    run_suite(load_config(args.config), args.output, (args.agent,) if args.agent else AGENTS, args.resume, args.workers)


if __name__ == "__main__":
    main()
