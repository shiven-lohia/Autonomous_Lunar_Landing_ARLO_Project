import argparse
from pathlib import Path

import yaml

from garl_lander.experiment import robustness


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved policies under gravity and wind changes")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--wait-for-completion", action="store_true",
                        help="Evaluate newly completed models while the source suite is still training")
    args = parser.parse_args()
    conditions = yaml.safe_load(Path(args.config).read_text())["conditions"]
    print(robustness(args.run, conditions, args.wait_for_completion))


if __name__ == "__main__":
    main()
