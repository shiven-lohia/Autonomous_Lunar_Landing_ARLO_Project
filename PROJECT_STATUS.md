# Project status

Final handoff status. Verified on a fresh clone: Python 3.12.2 venv, `scripts.doctor` succeeds, 26/26 pytest pass, ruff clean.

| Component | Evidence / status |
|---|---|
| Windows Python 3.11 installation | Verified with project-local Python 3.11.18 and editable virtual environment |
| LunarLander-v3 | Doctor completed a real episode on Box2D 2.3.10 |
| GA | Implemented; full runs completed for seeds 11, 23, 37; resume reproduces population and champion exactly |
| PPO | Implemented; full models for seeds 11, 23, 37 evaluated on 100 test episodes each |
| Hybrid | Actor transfer verified (logit error 0.0); optimizer changes actor; full runs completed for seeds 11, 23, 37 |
| Five-agent smoke suite | Completed with reloadable models and evaluation CSVs |
| Quick suite | Completed; all five checkpoints independently re-evaluated with identical returns |
| Main three-seed study | Completed (3 seeds x 5 agents, status `complete`); report in `report/full/` |
| Sensitivity | Reduced-budget exploratory sweep completed (11 variants); larger sweep not executed |
| Robustness | Completed, with one gap: `ppo_short_seed23` was not evaluated in the original run, and checkpoints are not in Git, so it cannot be added without retraining; PPO-short robustness uses seeds 11 and 37 only |
| Plots | Quick-run and full-run PNG/SVG plots generated from saved CSVs; full report figures inspected |
| Report | Quick and full (`report/full/`) PDF, Markdown and LaTeX generated; full PDF rendered and visually checked; student/institution fields left blank |
| Gameplay | Median-selected random, GA, PPO and hybrid videos recorded from full seed 11 on the original machine; MP4s and model checkpoints are gitignored, so only manifests are in the repo |
| Statistical analysis | Seed-level bootstrap, paired differences, thresholds and interaction tables implemented |
| Dependencies | Verified exact snapshot saved in requirements-lock.txt |
| Larger confirmatory ablations | Implemented/configured; not yet executed |
| LaTeX compilation | Not compiled; ReportLab PDF works; LaTeX source supplied (MiKTeX present locally but pdflatex stalled on package install) |

Headline (full study, 3 seeds): PPO 268.4, PPO-short 256.2, GA-PPO 244.1, GA -49.5, random -178.3.
The hybrid did not beat PPO on any seed. Details: `docs/EXECUTION_RECORD.md`.

No reported result is synthetic. See run manifests and raw CSV files for the evidence.
