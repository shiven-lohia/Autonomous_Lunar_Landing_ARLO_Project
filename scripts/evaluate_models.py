import argparse
from pathlib import Path

from garl_lander.experiment import reevaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved selected models without training")
    parser.add_argument("--run", type=Path, required=True, help="Explicit suite directory prevents mixing runs")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    print(reevaluate(args.run, args.stochastic))


if __name__ == "__main__":
    main()
