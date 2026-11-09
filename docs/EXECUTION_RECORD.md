# Execution record

Exact delivered run paths and measured results. Every number here is copied from the saved
CSV/JSON files named below; nothing is synthetic.

## Delivered runs

| Run | Path | Status |
|---|---|---|
| Quick suite (1 seed, 100k) | `results/raw/quick_c28c75b1b2_20261109T150437Z_3dac13` | complete; all five checkpoints re-evaluated with identical returns (`results/artifact_verification.json`) |
| Full suite (3 seeds, 1M) | `results/raw/full_2f0cf7ab0f_20261109T150901Z_6d56ad` | complete (`status.json`) |
| Robustness (frozen full-suite models) | `results/raw/full_2f0cf7ab0f_20261109T150901Z_6d56ad/robustness/robustness_2f0cf7ab0f_20261109T153100Z_5ebc0a` | complete except `ppo_short_seed23` (not evaluated) |
| Exploratory sweep (1 seed, 160k) | `results/ablations/sweep_eed27f6fcc_20261109T151055Z_a7de6f` | complete (11 variants, 22 agent runs) |
| Confirmatory sweep (`configs/ga_ablation.yaml`) | none | **not executed** |

The full suite was migrated mid-run to three independent seed workers
(`parallel_migration.json`). One interrupted PPO-short seed-11 attempt is archived under
`interrupted_attempts/` and is excluded from the successful-run budget.

## Reports

| Report | Path | Source |
|---|---|---|
| Full study (primary) | `report/full/report.pdf`, `report.md`, `main.tex` | full suite + sweep + robustness |
| Quick study | `report/quick/report.pdf`, `report.md`, `main.tex` | quick suite + sweep |

PDFs are rendered with ReportLab. The LaTeX sources were not compiled.
`provenance.json` in each report folder lists SHA-256 hashes of the source files.

## Full-suite results (100 test episodes per model, test seeds 40000+)

| Agent | Seed 11 | Seed 23 | Seed 37 | Mean | Solved (>=200) | Mean wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random | -178.3 | -178.3 | -178.3 | -178.3 | 0.0% | 0 |
| GA | -50.2 | -86.4 | -11.9 | -49.5 | 1.7% | 149 |
| PPO | 269.8 | 263.8 | 271.6 | **268.4** | 100.0% | 364 |
| GA-PPO | 254.4 | 247.5 | 230.5 | 244.1 | 96.7% | 305 |
| PPO-short | 259.0 | 250.2 | 259.3 | 256.2 | 100.0% | 299 |

Paired training-seed contrasts (seed-bootstrap 95% CI): hybrid - PPO = -24.2 [-41.1, -15.4];
hybrid - PPO-short = -12.0 [-28.9, -2.7]; hybrid - GA = +293.6 [242.4, 333.9].

First validation crossing of 200 return (optimization steps): PPO 692k / 479k / 373k;
GA-PPO 779k / 673k / 563k (seeds 11 / 23 / 37). Source: `analysis/thresholds.csv`.

Actor transfer: max logit error 0.0 on all seeds; post-transfer validation return equals the
GA champion's; PPO changed actor weights on all seeds. Source: `hybrid_seed*/transfer.json`.

**Interpretation.** GA-initialized PPO starts far higher than PPO from scratch (about -100 vs
about -600 validation return), but it learns more slowly afterwards. It finished below
full-budget PPO on all three seeds and below the PPO-short control. With three seeds this
is preliminary evidence against a hybrid advantage in this configuration.

## Robustness (mean return, frozen policies, same test seeds)

| Condition | GA | PPO | GA-PPO | PPO-short* | Random |
|---|---:|---:|---:|---:|---:|
| default | -49.5 | 268.4 | 244.1 | 259.2 | -178.3 |
| lighter gravity (-8.0) | -40.4 | 271.1 | 254.1 | 259.7 | -188.0 |
| heavier gravity (-11.5) | -50.1 | 254.6 | 236.6 | 253.5 | -171.2 |
| wind 10 / turbulence 1 | -136.9 | 215.7 | 162.7 | 188.6 | -245.8 |
| wind 15 / turbulence 1.5 | -191.2 | 124.9 | 93.6 | 110.9 | -300.9 |

*PPO-short averages seeds 11 and 37 only. Source: `robustness_results.csv`.

## Exploratory sweep notes

Settings: one seed, 160k ceiling, population 10, one fitness episode, 20 development-seed
episodes (60000+). Baseline variant = population 10, sigma 0.05, crossover 0.7, budget fraction
0.25. That baseline appears in every factor's table, which is why identical rows repeat.

In 8 of 11 hybrid runs, validation selection kept the initial transferred GA actor
(`selected_ppo_steps = 0`), because PPO never beat it on validation within this small budget.
Hybrid sweep differences mostly reflect the GA champion rather than PPO fine-tuning. Treat the
sweep as a pipeline check plus weak exploratory signal (for example sigma 0.02 and
population 20 scored highest), not as evidence about the full configuration.

## Not in Git

Model checkpoints (`*.zip`, `*.npz`) and MP4 videos are excluded by `.gitignore`. Only their
manifests and metadata are committed. A fresh clone must retrain to obtain checkpoints, about
5-6 minutes per PPO/hybrid run and 2-3 minutes per GA run on the original CPU. It can then
re-record videos with `scripts.record_gameplay` and `scripts.compare_gameplay`.

## Verification of this handoff

Fresh clone on Windows 11, Python 3.12.2 venv, `pip install -e ".[dev]"`:
`scripts.doctor` succeeded, `pytest` 26 passed, `ruff check .` clean.
