# GA-RL LunarLander

**Evolutionary Optimization of Reinforcement Learning Policies for Autonomous Lunar Landing**

An Applied Reinforcement Learning and Optimization course project comparing a random policy,
a genetic algorithm, PPO, GA-initialized PPO, and a PPO-short control. All trainable actors use
the same 8-64-64-4 Tanh network (4,996 parameters). The hybrid copies actual actor tensors into
Stable-Baselines3 PPO, verifies logits and actions, then trains actor and critic.

Start with `report/report.pdf` for the research report, `PROJECT_STATUS.md` for verified execution
status, and `docs/DEMONSTRATION.md` for a presentation walkthrough. Student/institution fields
are deliberately left for you to fill in.

## Windows setup

Python **3.11, 64-bit** and PowerShell are recommended. CPU is sufficient. No WSL, Docker,
CUDA, LaTeX installation or external video encoder is required. The verified installation
uses Gymnasium's Windows Box2D binary wheel and pygame-ce (which exposes the `pygame` API).
Do not install both pygame and pygame-ce into the same environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m scripts.doctor
.\.venv\Scripts\python.exe -m pytest -q
```

The existing workspace already has a working `.venv`; use its Python directly. Its locally
downloaded Python base is in `.python`, so retain that directory on this machine. These folders
are machine-specific and excluded from Git. On another computer create a fresh environment.

To reproduce the tested versions rather than resolve new versions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

## Run experiments

Run commands from the repository root. Rendering is off during training.

```powershell
# All five agents, tiny verification budget, standard environment
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/smoke.yaml

# Preliminary one-seed comparison, 100,000-transition ceiling
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/quick.yaml

# Main study: three seeds, 1,000,000 transitions, 100 test episodes
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/full.yaml

# Optional: run independent seeds on three CPU worker processes
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/full.yaml --workers 3

# A single algorithm (PPO-short requires the full suite)
.\.venv\Scripts\python.exe -m scripts.run_experiment --config configs/quick.yaml --agent hybrid
```

Each command prints a unique suite directory under `results/raw`. Specify it explicitly in
analysis commands; this prevents accidental mixing of different budgets or hyperparameters.

```powershell
$run = 'results/raw/<printed-suite-directory>'
.\.venv\Scripts\python.exe -m scripts.evaluate_models --run $run
.\.venv\Scripts\python.exe -m scripts.generate_plots --run $run
.\.venv\Scripts\python.exe -m scripts.generate_report --run $run
.\.venv\Scripts\python.exe -m scripts.record_gameplay --run $run --agent hybrid --seed 11 --episodes 1 --selection median
```

The report generator writes Markdown, compilable LaTeX, bibliography, PDF and figure copies.
It uses actual data, labels missing experiments and includes source hashes. Use `--output`
to retain alternate reports. All commands have `--help`.

## Sensitivity and robustness

```powershell
# Exploratory one-seed, reduced-budget sweeps for all four factors
.\.venv\Scripts\python.exe -m scripts.run_ablation --config configs/ablation_quick.yaml

# Larger confirmatory sweep; substantially more expensive
.\.venv\Scripts\python.exe -m scripts.run_ablation --config configs/ga_ablation.yaml

# Evaluate frozen selected models under gravity/wind changes
.\.venv\Scripts\python.exe -m scripts.run_robustness --run $run --config configs/robustness.yaml

# Use the printed sweep/robustness directories
.\.venv\Scripts\python.exe -m scripts.generate_plots --run '<supplement-directory>' --supplement
.\.venv\Scripts\python.exe -m scripts.generate_report --run $run --supplement '<sweep-directory>' --supplement '<robustness-directory>'
```

Sweeps vary one factor at a time and evaluate on **development seeds starting at 60,000**.
They do not tune against the main final-test seeds. The reduced sweep uses a different GA
fitness-episode count, population and budget from the main study and is labeled exploratory.
It is not evidence that an apparent effect generalizes to the full configuration.

## Videos and demonstration

`--selection first` uses the first fixed test starts, `median` chooses returns nearest the
evaluation median, and `highest` explicitly selects the best examples. The selection is
written into the video overlay and metadata. Use the same episode seed for comparisons;
do not present the highest-return video as typical behavior. MP4 encoding uses imageio-ffmpeg.

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_environment --seed 11
```

This optional command opens a Pygame window. Headless recording does not need a GUI.

For a side-by-side comparison on one predeclared common start:

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_gameplay --run $run --seed 11 --episode-seed 40000
```

## Scientific protocol

- Default discrete LunarLander-v3, gravity -10, wind off, 1,000-step episode limit,
  original rewards and all eight raw observations. Side engines produce torque and thrust,
  rather than direct left/right movement.
- GA: complete episodes, common training starts within a generation, fresh starts across
  generations, tournament selection, Gaussian mutation, optional crossover and copied elites.
- PPO: SB3 PPO, separate actor/critic, four optimization epochs by default, fixed hyperparameters.
- Hybrid: actor-only initialization, fresh critic and optimizer, verified transfer, then PPO.
- Primary comparison is an **equal budget ceiling**, with actual optimization and unused
  allocation reported. GA reserves enough for a complete worst-case generation. PPO rounds
  down to full `n_steps * n_envs` batches. The hybrid gives unused GA allocation back to PPO.
- GA/PPO model selection uses validation seeds; final testing uses the same held-out starts
  for every method. Random baseline action randomness is also reproducible.
- Validation and testing are counted separately. Learning curves use actual interactions.
- Training seeds are the replication unit. Episode distributions are descriptive. Three seeds
  and small validation sets limit statistical conclusions; no automatic significance claims.
- Return >=200 is a reward threshold, not a physical landing detector. Resting landings,
  crashes, out-of-bounds and time limits are recorded separately. A resting body may be off-pad.

## Checkpoints and interruptions

GA stores a complete next-generation population, champion, counters and four RNG states in
a safe NumPy/JSON checkpoint. Under the same software, generation-boundary continuation is exact:

```powershell
.\.venv\Scripts\python.exe -m scripts.resume_ga --checkpoint '<run>/ga/checkpoint.npz' --budget 1000000 --output results/ga_continuation
```

PPO saves `selected.zip`, `latest.zip`, and `final.zip`, including optimizer state. Validation
selection may choose an earlier policy, including initialization. The public suite resume
retains completed agents and archives/restarts the interrupted agent for auditability:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_all --config configs/full.yaml --resume $run
```

A PPO environment restart is not bitwise continuation of an in-progress physics simulation.
Archived failed attempts are excluded from successful-run budgets; their resource use is
additional. Never load untrusted SB3 ZIP checkpoints.

## Repository map

| Path | Purpose |
|---|---|
| `src/garl_lander` | Agents, transfer, evaluation, accounting, analysis, report and recording |
| `configs` | Smoke, quick, full, exploratory and full ablations, robustness |
| `scripts` | PowerShell-compatible Python module commands |
| `tests` | Unit, budget, resume, transfer, integration and analysis checks |
| `results/raw/<suite>` | Configuration, machine/package metadata, per-agent results and models |
| `<suite>/analysis` | Per-seed metrics, aggregate tables, paired contrasts, threshold observations |
| `<suite>/plots` | PNG and SVG plots regenerated from measured CSV files |
| `report` | PDF, Markdown, LaTeX, bibliography and source hashes |
| `videos` | Labeled gameplay MP4s and recording manifests |
| `notebooks` | Optional read-only results exploration |
| `docs` | Protocol, algorithms, architecture and demonstration notes |

Models are colocated with run metadata rather than duplicated in a global `models/` folder.
This preserves their configuration and selection history. Generated files are created automatically.

## Troubleshooting

If Box2D fails, first run `python --version` and `python -c "import platform; print(platform.architecture())"`.
Inspect the actual pip error and check for a compatible Windows wheel. The verified snapshot
has Box2D 2.3.10 wheels on Python 3.11. If a source build is required, install SWIG and Microsoft
C++ Build Tools before retrying; do not randomly downgrade Gymnasium.

If `py -3.11` is missing, install 64-bit Python 3.11 or use an existing Python 3.11 interpreter
to create the environment. No PowerShell execution-policy change is needed when calling
`.venv\Scripts\python.exe` directly. If packages cannot be downloaded, retain the exact error
and resolve network access before attempting training.

If a GA allocation cannot reserve one generation, increase its budget, reduce population or
fitness episode count, or use the smoke configuration. Shortening episodes changes the task
and is reserved for integration tests. For slow CPU training keep Torch threads at one first;
more threads can make this small network slower. TensorBoard logs are in each PPO run's
`ppo/tensorboard` directory.

See `REPRODUCIBILITY.md`, `PROJECT_STATUS.md`, and `docs/EXECUTION_RECORD.md` for the tested
versions, exact delivered run paths, results, and remaining limitations.
