# Reproducibility record

## Verified runtime

Windows 64-bit, Python **3.11.18**, CPU-only Torch. `requirements-lock.txt` contains the
exact installed package snapshot. The primary versions are Torch 2.15.0, Gymnasium 1.4.0,
Stable-Baselines3 2.9.2, Box2D 2.3.10 and NumPy 2.4.6. This is a tested snapshot, not a
claim that all future combinations inside the pyproject ranges will behave identically.

Install from the lock with Python 3.11, then editable-install the project with `--no-deps`.
Recreate `.venv` when moving machines.

Handoff re-verification: a fresh Git clone on Windows 11 with Python 3.12.2 and `pip install -e ".[dev]"`
resolved the same primary versions except NumPy 2.5.4. `scripts.doctor` succeeded, 26 tests passed and
ruff was clean. The delivered experiments themselves ran on Python 3.11.18.

The `manifest.json` in each suite includes operating system, processor identifier, logical
CPU count, RAM, package versions, resolved YAML configuration, creation timestamp and
configuration hash. Git revision is null in the delivered manifests because the experiments ran before the
workspace was a Git repository. The project is now tracked at
`github.com/trynottofear/Autonomous_Lunar_Landing_ARLO_Project`.
Source hashes and final artifact hashes are recorded in the handoff metadata.

## Fixed experimental design

| Mode | Training seeds | Main ceiling | Test episodes/model | PPO rollout |
|---|---|---:|---:|---:|
| Smoke | 11 | 24,000 | 3 | 128 x 1 |
| Quick | 11 | 100,000 | 20 | 512 x 2 |
| Full | 11, 23, 37 | 1,000,000 | 100 | 1,024 x 4 |
| Exploratory sweeps | 11 | 160,000 | 20 | 512 x 2 |
| Confirmatory sweeps | 11, 23, 37 | 1,000,000 | 100 | 1,024 x 4 |

The random agent has no optimization allocation. PPO-short uses each hybrid's actual PPO
steps. GA does not begin a generation unless the worst-case complete generation fits;
its unused budget is reported. The hybrid uses actual GA cost before computing PPO allocation.
PPO floors to complete rollouts. Validation and testing are measured outside optimization cost.

Baseline environment: LunarLander-v3, discrete actions, original rewards, raw observations,
gravity -10, wind off, 1,000-step episode limit. Only tiny automated tests shorten the horizon.
All trainable actors have 4,996 parameters and two 64-unit Tanh hidden layers. Critic parameters
are not evolved. Model selection is best validation mean with earliest-tie retention. Saved
`selected` and `final` models may differ.

## Randomness and seed partitions

Python, NumPy and Torch are seeded, Torch deterministic algorithms are requested, and its
thread count is fixed to one. GA uses independent RNG streams for initialization, selection,
crossover and mutation. Training environment seeds use separate stream blocks at or above
one million. Validation seeds begin at 20,000; transfer-reset observations use 30,000-30,007;
final test seeds begin at 40,000; exploratory ablation evaluation begins at 60,000. Final seeds
are never used to choose a checkpoint. No hyperparameters were tuned to main final-test outcomes.

Random-agent actions use an episode-seed-derived RNG, so its repeated rows across suite seeds
are identical and are not independent training replications. Final evaluations are deterministic
for trained policies. Optional stochastic reevaluation is saved in a separate labeled directory.

## Reproduce the pipeline

```powershell
.\.venv\Scripts\python.exe -m scripts.doctor
.\.venv\Scripts\python.exe -m pytest --cov=garl_lander --cov-report=term-missing -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/quick.yaml
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/full.yaml
.\.venv\Scripts\python.exe -m scripts.run_ablation --config configs/ablation_quick.yaml
```

Use the explicit run paths in `docs/EXECUTION_RECORD.md` to regenerate the delivered artifacts
without training. `scripts.verify_artifacts --run <suite> --report report/full/report.pdf` reopens all
selected policies, repeats the complete final evaluation, verifies episode returns and lengths,
checks accounting equations, renders the PDF and decodes every recorded MP4's first frame.
The extra verification interactions are saved separately in `verification.json`. This needs
the model checkpoints, which are not committed to Git; a fresh clone must retrain first.

Regenerate the delivered full-study plots and report from the committed CSVs (no training):

```powershell
$run = 'results/raw/full_2f0cf7ab0f_20261109T150901Z_6d56ad'
$rob = "$run/robustness/robustness_2f0cf7ab0f_20261109T153100Z_5ebc0a"
$sweep = 'results/ablations/sweep_eed27f6fcc_20261109T151055Z_a7de6f'
.\.venv\Scripts\python.exe -m scripts.generate_plots --run $run
.\.venv\Scripts\python.exe -m scripts.generate_plots --run $rob --supplement
.\.venv\Scripts\python.exe -m scripts.generate_report --run $run --supplement $sweep --supplement $rob --output report/full
```

## Compare a reproduced run

Compare resolved configurations before comparing returns. The hash must match. Under the same
runtime and hardware, the selected-policy episode rows should agree to floating-point tolerance.
Compare `analysis/per_seed.csv`, `analysis/paired_contrasts.csv` and raw `evaluation.csv` files.
Do not compare suite directory names or elapsed times for exact equality. GA restart tests
require identical populations, champions and counters after a generation-boundary resume.

## Limitations

Different PyTorch, NumPy, Box2D, CPU instruction sets or devices can change trajectories and
long-run optimization. GPU execution is optional and was not validated. PPO checkpoints save
optimizer state but not a serializable mid-episode Box2D world; the suite resume archives and
restarts an interrupted agent. Exact full-suite continuation inside a PPO episode is not claimed.

Runtime measurements include checkpointing and validation. Experiments shared the workstation
with analysis and exploratory sweep jobs, so wall time is descriptive and not a controlled
microbenchmark. Recordings, repeated verification and robustness evaluations add interactions
beyond each main suite's training ledger; their manifests account for those extra measurements.

Three training seeds support only preliminary uncertainty estimates. Five validation episodes
can be overfit through repeated model selection. GA and PPO perform different numbers of
selection evaluations; both cost and selection-opportunity differences must be considered.
The reduced sensitivity suite has one seed and different population/fitness settings from the
main study. The full confirmatory sweep is a separate, more expensive experiment and was not executed.
In 8 of 11 exploratory hybrid runs, validation kept the initial GA actor (no PPO checkpoint beat it).
The robustness evaluation omits `ppo_short_seed23`.
