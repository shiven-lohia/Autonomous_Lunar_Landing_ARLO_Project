# Project status

Status while experiments execute. This file will be updated with the final verified handoff.

| Component | Evidence / status |
|---|---|
| Windows Python 3.11 installation | Verified with project-local Python 3.11.18 and editable virtual environment |
| LunarLander-v3 | Doctor completed a real episode on Box2D 2.3.10 |
| GA | Implemented; full seed-11 run completed; resume reproduces population and champion exactly |
| PPO | Implemented; full seed-11 model evaluated on 100 episodes |
| Hybrid | Actor transfer verified; optimizer changes actor; full seed-11 run completed |
| Five-agent smoke suite | Completed with reloadable models and evaluation CSVs |
| Quick suite | Completed; all five checkpoints independently re-evaluated with identical returns |
| Main three-seed study | Running; independent seed worker support tested |
| Sensitivity | Reduced-budget mutation/population/allocation/crossover sweep running |
| Robustness | Implemented; full-suite execution follows training |
| Plots | Quick-run PNG and SVG plots generated and inspected |
| Report | Quick PDF, Markdown and LaTeX generated; PDF rendered and visually checked |
| Gameplay | Median-selected random, GA, PPO and hybrid videos recorded from full seed 11 |
| Statistical analysis | Seed-level bootstrap, paired differences, thresholds and interaction tables implemented |
| Dependencies | Verified exact snapshot saved in requirements-lock.txt |
| Larger confirmatory ablations | Implemented/configured; not yet executed |
| LaTeX compilation | Compiler unavailable; ReportLab PDF works; LaTeX source supplied |

No reported result is synthetic. See run manifests and raw CSV files for the evidence.
