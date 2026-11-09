import argparse
from pathlib import Path

from garl_lander.config import load_config
from garl_lander.experiment import run_sweeps


def main():
    parser = argparse.ArgumentParser(description="Run one-factor GA and hybrid sensitivity experiments")
    parser.add_argument("--config", default="configs/ga_ablation.yaml")
    parser.add_argument("--output", type=Path, default=Path("results/ablations"))
    args = parser.parse_args()
    print(run_sweeps(load_config(args.config), args.output))


if __name__ == "__main__":
    main()
