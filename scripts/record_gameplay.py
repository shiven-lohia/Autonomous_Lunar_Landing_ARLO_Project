import argparse
from pathlib import Path

from garl_lander.experiment import AGENTS
from garl_lander.recording import record


def main():
    parser = argparse.ArgumentParser(description="Record selected saved policies without training")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--agent", choices=AGENTS, default="hybrid")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--selection", choices=["first", "median", "highest"], default="median")
    args = parser.parse_args()
    print(record(args.run, args.agent, args.seed, args.episodes, args.selection))


if __name__ == "__main__":
    main()
