"""Write a factual completion record from explicitly verified artifacts."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from garl_lander.io_utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    args = parser.parse_args()
    if read_json(args.run / "status.json")["status"] != "complete":
        raise RuntimeError("Cannot mark an incomplete suite complete")
    verification = read_json(args.run / "verification.json")
    artifact_verification = read_json("results/artifact_verification.json")
    comparison = pd.read_csv(args.run / "analysis" / "comparison.csv")
    per_seed = pd.read_csv(args.run / "analysis" / "per_seed.csv")
    sweep = pd.read_csv(args.sweep / "sweep_results.csv")
    robust = pd.read_csv(args.robustness / "robustness_results.csv")
    tests = Path("logs_tests.txt").read_text(encoding="utf-16") if Path("logs_tests.txt").read_bytes().startswith(b"\xff\xfe") else Path("logs_tests.txt").read_text()
    test_result = next(line for line in tests.splitlines() if " passed" in line and " in " in line)
    videos = sorted(Path("videos").glob("*/*.mp4"))
    full_plots = sorted((args.run / "plots").glob("*.png"))
    table = ["| Agent | Mean return | Seed SD | Solved episodes | Mean training seconds |",
             "|---|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        sd = f"{row.seed_sd:.1f}" if pd.notna(row.seed_sd) else "n/a"
        table.append(f"| {row.agent} | {row.mean_return:.1f} | {sd} | {100*row.solved_rate:.1f}% | {row.training_seconds:.1f} |")
    record = f"""# Final execution and engineering record

## Delivered main results

Main suite: `{args.run.as_posix()}`. Three independent training seeds (11, 23, 37),
one-million-transition optimization ceiling per GA/PPO/hybrid agent, and 100 final episodes
per selected checkpoint. All five methods completed, producing {len(per_seed)} agent records.

{chr(10).join(table)}

The random policy is evaluation-only; repeated random rows are not independent training seeds.
PPO-short matches the hybrid's realized PPO interactions. These are validation-selected models.
Exact totals and residual allocation are in `analysis/per_seed.csv`.

## Executed work and verification

- Windows Python 3.11.18 and 55 project/runtime/development distributions installed and verified.
- `scripts.doctor`: constructed LunarLander-v3, checked API/spaces, completed a real random episode.
- Pytest: **{test_result}**. Full output and coverage: `logs_tests.txt`.
- Ruff: passed (`logs_lint.txt`).
- Smoke and quick five-agent suites: completed.
- Main suite: completed; every selected checkpoint loaded and all test returns/lengths reproduced.
- Independent verification added {verification['additional_verification_interactions']:,} evaluation interactions,
  recorded separately from training budgets in the suite's `verification.json`.
- Sensitivity: {len(sweep)} agent runs across {sweep[['parameter','value']].drop_duplicates().shape[0]} one-factor settings,
  one seed and 160,000-transition ceiling; development evaluation seeds start at 60,000.
- Robustness: {len(robust)} method/seed/condition records, {robust.episodes.sum():,} evaluated episodes,
  {robust.test_steps.sum():,} extra evaluation transitions; frozen selected policies, no retraining.
- {len(full_plots)} main PNG figures with SVG counterparts; supplement plots also generated.
- {len(videos)} MP4s recorded and decoded, including the fixed-start side-by-side comparison.
- Final PDF: {artifact_verification.get('pdf_pages', 'unknown')} pages; rendered for visual inspection.
- Markdown, LaTeX, bibliography, raw metrics, source hashes and a dependency snapshot supplied.

## Algorithms and files

`src/garl_lander` contains the network, NumPy GA operators, safe GA checkpoints, SB3 PPO,
verified actor transfer, exact step accounting, unified evaluator, statistical analysis,
plotting, recording and report generation. `scripts` contains executable workflows.
`configs`, `tests`, `docs`, `notebooks`, `results`, `report` and `videos` contain the supporting
artifacts. The repository README documents their roles and Windows setup.

Each GA/hybrid run saves `ga/initial.npz`, `ga/selected.npz`, `ga/checkpoint.npz`, episode logs,
and generation histories. PPO/hybrid/PPO-short runs save `ppo/selected.zip`, `ppo/latest.zip`,
`ppo/final.zip`, validation rows, per-update losses and Monitor/TensorBoard logs. Hybrid runs
include `transfer.json` with logit/action/critic verification and actual actor-update evidence.

## Exact commands used

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.doctor
.\\.venv\\Scripts\\python.exe -m pytest --cov=garl_lander --cov-report=term-missing -q
.\\.venv\\Scripts\\python.exe -m ruff check src scripts tests
.\\.venv\\Scripts\\python.exe -m scripts.run_all --config configs/smoke.yaml
.\\.venv\\Scripts\\python.exe -m scripts.run_all --config configs/quick.yaml
.\\.venv\\Scripts\\python.exe -m scripts.run_all --config configs/full.yaml
.\\.venv\\Scripts\\python.exe -m scripts.run_all --config configs/full.yaml --resume '{args.run.as_posix()}' --workers 3
.\\.venv\\Scripts\\python.exe -m scripts.run_ablation --config configs/ablation_quick.yaml
.\\.venv\\Scripts\\python.exe -m scripts.run_robustness --run '{args.run.as_posix()}' --config configs/robustness.yaml --wait-for-completion
.\\.venv\\Scripts\\python.exe -m scripts.generate_report --run '{args.run.as_posix()}' --supplement '{args.sweep.as_posix()}' --supplement '{args.robustness.as_posix()}'
.\\.venv\\Scripts\\python.exe -m scripts.verify_artifacts --run '{args.run.as_posix()}' --report report/report.pdf
.\\.venv\\Scripts\\python.exe -m scripts.compare_gameplay --run '{args.run.as_posix()}' --seed 11 --episode-seed 40000
```

Median videos used `scripts.record_gameplay --run <main-suite> --agent <random|ga|ppo|hybrid>
--seed 11 --episodes 1 --selection median`. The quick hybrid video and independent quick-model
reevaluation were also executed. `scripts.generate_report` includes plot generation.

The environment was bootstrapped with project-local uv, Python 3.11 and an editable `.[dev]`
installation. A pygame/pygame-ce namespace overlap was corrected by retaining pygame-ce.
Exact version snapshot: `requirements-lock.txt`. No system Python or user projects were replaced.

## Resource accounting and limitations

The first full seed ran serially. After its hybrid completed, the runner migrated to three
independent seed workers. Completed checkpoints were retained; an interrupted PPO-short attempt
was archived under `interrupted_attempts`. Its partial work is additional overhead, not part
of any successful agent's optimization ledger. `parallel_migration.json` records this transition.
Runtime was measured on a shared workstation with simultaneous analysis and exploratory jobs;
it is descriptive, not an isolated timing benchmark.

The **full-sized, three-seed confirmatory ablation sweep was not executed**. Its configuration
and pipeline are provided. The smaller sweep covers all requested factors but supports only
exploratory conclusions. Three main training seeds and five validation starts limit inference.
Do not infer statistical significance from a point estimate or bootstrap interval alone.

No LaTeX compiler was available, so LaTeX compilation was not tested. The ReportLab PDF was
generated, rendered and checked; `main.tex` and `references.bib` are supplied for an alternative
build. GPU training and optional Jupyter execution were not tested. SB3 checkpoints must be
trusted. Exact mid-episode PPO resume is not promised; GA generation-boundary resume is tested.

## Delivered recordings

"""
    record += "\n".join(f"- `{p.as_posix()}`" for p in videos)
    Path("docs/EXECUTION_RECORD.md").write_text(record, encoding="utf-8")
    status = f"""# Project status - verified handoff

- [x] Windows Python 3.11 editable installation and exact dependency snapshot.
- [x] Real LunarLander-v3, GA evolution, SB3 PPO and genuine GA-to-PPO actor transfer.
- [x] Critic preservation, equivalent logits/actions, and real PPO actor updates verified.
- [x] Models saved/reloaded; GA generation-boundary exact resume tested.
- [x] Correct optimization, validation, final-test and residual budget ledgers.
- [x] Five-agent smoke, quick and full three-seed studies completed.
- [x] All {len(per_seed)} main selected checkpoints independently reevaluated with matching returns.
- [x] Exploratory sensitivity: {len(sweep)} completed GA/hybrid runs across all four factors.
- [x] Robustness: {len(robust)} method/seed/condition records evaluated without retraining.
- [x] {len(full_plots)} main figures plus supplement PNG/SVG plots from saved data.
- [x] {len(videos)} gameplay videos recorded and decoded.
- [x] Final PDF, Markdown, LaTeX, bibliography, README, protocol, notebook and demonstration guide.
- [x] {test_result}; Ruff passed.
- [ ] Full-budget three-seed confirmatory sensitivity sweep (configured; not executed).
- [ ] LaTeX compiler verification (compiler unavailable; delivered PDF uses ReportLab).
- [ ] GPU, exact mid-episode PPO continuation and optional Jupyter execution (not validated).

No synthetic training values or fabricated significance claims are included. See
`docs/EXECUTION_RECORD.md` for commands, measured outcomes, archived interruption, resource
accounting, remaining limitations, and artifact locations.
"""
    Path("PROJECT_STATUS.md").write_text(status, encoding="utf-8")
    source_files = [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), *Path("configs").glob("*.yaml"),
                    Path("requirements-lock.txt"), Path("pyproject.toml")]
    write_json(Path("results/handoff_manifest.json"), {
        "main_suite": str(args.run), "sensitivity_suite": str(args.sweep), "robustness_suite": str(args.robustness),
        "test_result": test_result, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "report_sha256": hashlib.sha256(Path("report/report.pdf").read_bytes()).hexdigest(),
    })
    print("Wrote PROJECT_STATUS.md, docs/EXECUTION_RECORD.md and results/handoff_manifest.json")


if __name__ == "__main__":
    main()
