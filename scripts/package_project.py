"""Create a portable submission archive without machine-specific environments."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

EXCLUDED = {".venv", ".python", ".tools", ".uv-cache", ".git", ".pytest_cache", ".ruff_cache",
            "__pycache__", "tmp", "dist", "htmlcov"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/GA_RL_Course_Project.zip"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
                continue
            if not path.is_file() or path == output or path.name == ".coverage":
                continue
            if "tensorboard" in relative.parts:
                continue
            if "interrupted_attempts" in relative.parts and path.suffix not in (".json", ".csv"):
                continue
            archive.write(path, Path("GA_RL_Gaming_Agent") / relative)
            count += 1
    print(f"Packaged {count} files: {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
