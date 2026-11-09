import argparse
from pathlib import Path

from garl_lander.genetic_algorithm import GeneticAlgorithm


def main():
    parser = argparse.ArgumentParser(description="Resume GA exactly from a generation boundary")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True, help="Total GA budget including saved steps")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for this continuation")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve the original checkpoint and logs")
    ga = GeneticAlgorithm.load(args.checkpoint)
    if args.budget < ga.optimization_steps:
        parser.error("Budget is smaller than the checkpoint's consumed steps")
    ga.train(args.budget, args.output)


if __name__ == "__main__":
    main()
