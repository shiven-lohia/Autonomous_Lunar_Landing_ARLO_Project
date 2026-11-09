import argparse
from pathlib import Path

from garl_lander.plotting import generate_plots, plot_supplement


def main():
    parser = argparse.ArgumentParser(description="Regenerate PNG/SVG figures from a single recorded suite")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--supplement", action="store_true", help="Plot an ablation or robustness directory")
    args = parser.parse_args()
    print(plot_supplement(args.run) if args.supplement else generate_plots(args.run))


if __name__ == "__main__":
    main()
