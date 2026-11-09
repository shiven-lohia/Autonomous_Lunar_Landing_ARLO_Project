import argparse
from pathlib import Path

from garl_lander.reporting import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown, LaTeX and PDF from recorded experiments")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("report"))
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(generate_report(args.run, args.output, args.supplement))


if __name__ == "__main__":
    main()
