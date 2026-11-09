# Project status

Final handoff status. Verified on a fresh clone: Python 3.12.2 venv, `scripts.doctor` succeeds, 26/26 pytest pass, ruff clean.

| Component | Evidence / status |
|---|---|
| Windows Python 3.11 installation | Verified with project-local Python 3.11.18 and editable virtual environment |
| LunarLander-v3 | Doctor completed a real episode on Box2D 2.3.10 |
| GA | Implemented; full seed-11 run completed; resume reproduces population and champion exactly |
| PPO | Implemented; full seed-11 model evaluated on 100 episodes |
| Hybrid | Actor transfer verified; optimizer changes actor; full seed-11 run completed |
| Five-agent smoke suite | Completed with reloadable models and evaluation CSVs |
| Quick suite | Completed; all five checkpoints independently re-evaluated with identical returns |
| Main three-seed study | Completed (3 seeds x 5 agents, status `complete`); report in `report/full/` |
| Sensitivity | Reduced-budget exploratory sweep completed (11 variants); larger sweep not executed |
| Robustness | Completed, with one gap: `ppo_short_seed23` was not evaluated (its checkpoint was not in the repo), so PPO-short robustness uses seeds 11 and 37 only |
| Plots | Quick-run and full-run PNG/SVG plots generated (full plots not visually inspected) |
| Report | Quick and full (`report/full/`) PDF, Markdown and LaTeX generated; full PDF not yet visually checked; student/institution fields left blank |
| Gameplay | Median-selected random, GA, PPO and hybrid videos recorded from full seed 11 |
| Statistical analysis | Seed-level bootstrap, paired differences, thresholds and interaction tables implemented |
| Dependencies | Verified exact snapshot saved in requirements-lock.txt |
| Larger confirmatory ablations | Implemented/configured; not yet executed |
| LaTeX compilation | Not compiled; ReportLab PDF works; LaTeX source supplied (MiKTeX present locally but pdflatex stalled on package install) |

No reported result is synthetic. See run manifests and raw CSV files for the evidence.
