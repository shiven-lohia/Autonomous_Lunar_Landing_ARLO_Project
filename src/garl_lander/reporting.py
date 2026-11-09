"""Generate matching Markdown, LaTeX and a typeset PDF from measured results."""

import hashlib
import html
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .experiment import suite_runs
from .io_utils import read_json, write_json
from .plotting import LABELS, generate_plots

TITLE = ("Evolutionary Optimization of Reinforcement Learning Policies for Autonomous Lunar Landing: "
         "A Comparative Study of Genetic Algorithms, PPO, and Hybrid GA-PPO")
REFERENCES = [
    ("Schulman et al. (2017). Proximal Policy Optimization Algorithms.", "https://arxiv.org/abs/1707.06347"),
    ("Such et al. (2017). Deep Neuroevolution.", "https://arxiv.org/abs/1712.06567"),
    ("Khadka and Tumer (2018). Evolution-Guided Policy Gradient in Reinforcement Learning.",
     "https://arxiv.org/abs/1805.07917"),
    ("Raffin et al. (2021). Stable-Baselines3. JMLR 22(268), 1-8.", "https://jmlr.org/papers/v22/20-1364.html"),
    ("Farama Foundation. Lunar Lander documentation (accessed 9 November 2026).",
     "https://gymnasium.farama.org/environments/box2d/lunar_lander/"),
    ("Stable-Baselines3. PPO documentation (accessed 9 November 2026).",
     "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"),
]


def md_table(headers: list, rows: list) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def tex_escape(value) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in str(value))


def tex_prose(value: str) -> str:
    keys = {"1": "schulman2017ppo", "2": "such2017deep", "3": "khadka2018erl",
            "4": "raffin2021sb3", "5": "gymnasiumlander", "6": "sb3docs"}
    return re.sub(r"\[([1-6](?:,[1-6])*)\]",
                  lambda match: r"\cite{" + ",".join(keys[n] for n in match.group(1).split(",")) + "}",
                  tex_escape(value))


def content_blocks(sections, tables, figures):
    """Place evidence beside its discussion; reserve detailed logs for the appendix."""
    def table_section(title):
        if title.startswith(("Sensitivity", "Robustness")):
            return "6.1."
        if title.startswith(("Resolved", "Per-seed", "First observed")):
            return "Appendix"
        return "6. Results"

    def figure_section(name):
        if name == "architecture":
            return "4. Methodology"
        if name in ("ga_mutation_sigma", "ga_population_size", "hybrid_ga_budget_fraction",
                    "ga_crossover_probability", "robustness"):
            return "6.1."
        if name in ("ga_diversity", "ppo_only_axis", "total_interactions", "wall_time_learning", "solved_rates"):
            return "Appendix"
        return "6. Results"

    for title, paragraphs in sections:
        yield "section", (title, paragraphs)
        for table in tables:
            if title.startswith(table_section(table[0])):
                yield "table", table
        for figure in figures:
            if title.startswith(figure_section(figure)):
                yield "figure", figure


def build_content(suite: Path):
    manifest = read_json(suite / "manifest.json")
    cfg = manifest["config"]
    aggregate = pd.read_csv(suite / "analysis" / "comparison.csv")
    per_seed = pd.read_csv(suite / "analysis" / "per_seed.csv")
    metrics = aggregate.set_index("agent")
    n = len(cfg["experiment"]["seeds"])
    budget = cfg["experiment"]["total_optimization_steps"]
    eval_n = cfg["experiment"]["final_eval_episodes"]
    mode = "full-budget" if budget >= 1_000_000 and n >= 3 and eval_n >= 100 else "preliminary"
    def result(agent):
        return f"{metrics.loc[agent, 'mean_return']:.1f}" if agent in metrics.index else "not executed"
    comparison = (f"Mean test returns were {result('ga')} for GA, {result('ppo')} for PPO, "
                  f"{result('hybrid')} for GA-PPO, and {result('ppo_short')} for PPO-short.")
    abstract = (
        "This project investigates whether evolutionary initialization improves reinforcement learning for autonomous "
        "landing in the discrete LunarLander-v3 environment. A genetic algorithm optimizes all 4,996 parameters of a "
        "neural actor, after which proximal policy optimization fine-tunes the transferred actor while learning a newly "
        "initialized critic. The comparison includes a uniformly random policy, pure genetic search, PPO from scratch, "
        "the two-stage hybrid, and a PPO-short control that matches the hybrid's gradient-training interactions. "
        "Optimization is constrained by a common interaction ceiling, with complete GA fitness episodes and complete "
        "PPO rollouts. Validation, final evaluation, and unused allocation are measured separately. Policies are selected "
        "using validation starts and then evaluated on a common, disjoint test seed set. "
        f"The reported {mode} experiment uses {n} independent training seeds, an allocation of {budget:,} transitions "
        f"per main trained agent, and {eval_n} test episodes per checkpoint. {comparison} "
        "These measurements describe the implemented configuration rather than a general ranking of optimization methods. "
        "The repository supplies unit and integration tests, numerical actor-transfer verification, resumable genetic "
        "checkpoints, machine-readable interaction ledgers, learning curves, gameplay recordings, and generated report sources. "
        "Uncertainty is summarized at the training-seed level to avoid treating many episodes from one policy as independent "
        "training replications. Limitations include the small number of training seeds, validation selection effects, "
        "weight-space crossover disruption, and the cost of population evaluation. The study tests the hybrid hypothesis "
        "without assuming that combining optimization methods guarantees better performance."
    )
    sections = [
        ("1. Introduction", [
            "Reinforcement learning optimizes decisions whose consequences unfold over time. Autonomous landing is a useful "
            "course benchmark because attitude, velocity, contact, and engine control interact through nonlinear dynamics. "
            "An actor must trade immediate thrust costs against the delayed consequences of a stable touchdown.",
            "Gradient optimization exploits differentiable policy models but can be sensitive to initialization and sampled "
            "trajectories. Evolution evaluates complete behaviors without differentiating through the environment, at the cost "
            "of many interactions. This project examines whether evolutionary search provides an initialization worth its cost.",
            "Contributions are an auditable two-stage implementation, a matched architecture across methods, a PPO-short "
            "control, explicit interaction accounting, and an experiment-to-report pipeline. The null position is that GA "
            "initialization offers no consistent benefit once its interactions and wall time are included.",
        ]),
        ("2. Literature review", [
            "Such et al. [2] study genetic optimization of deep neural policies and demonstrate that direct weight search can "
            "scale beyond small networks. Their findings motivate testing neuroevolution here, but do not establish its "
            "effectiveness for this architecture, budget, or LunarLander version.",
            "Schulman et al. [1] introduce PPO, alternating trajectory collection with minibatch optimization of a surrogate "
            "policy objective. This project uses the tested Stable-Baselines3 implementation described by Raffin et al. [4], "
            "with explicit settings and reproducible package versions.",
            "Khadka and Tumer [3] combine evolutionary populations and policy gradients through shared experience and periodic "
            "reinsertion of a gradient-trained policy. Our method has a single actor transfer followed by PPO training. It "
            "does not reproduce that interleaved ERL algorithm. Its narrower question concerns the value of evolved initialization.",
        ]),
        ("3. Problem formulation and hypotheses", [
            "The task is an episodic Markov decision process M=(S,A,P,R,gamma). The eight observations encode position, "
            "linear velocity, angle, angular velocity, and two leg contacts. Actions are no thrust, left orientation engine, "
            "main engine, and right orientation engine [5]. Orientation engines apply forces and torques; they are not "
            "direct horizontal displacement commands.",
            "The objective is expected discounted return J(theta)=E[sum_t gamma^t r_t], while evaluation reports the "
            "undiscounted episode return. Default rewards combine state-dependent shaping and engine-use penalties. "
            "Episodes end through crash, leaving bounds, resting, or the time limit. The benchmark keeps the original "
            "reward function, raw observations, discrete actions, default gravity and no wind.",
            "H1: GA produces policies better than random under the allocated budget. H2: evolved initialization improves "
            "validation performance before PPO updates. H3: the hybrid reaches observed validation thresholds with fewer "
            "total optimization interactions than PPO. H4: the hybrid improves final return relative to both PPO and "
            "PPO-short. H5: mutation, population size, crossover, and budget fraction materially affect results. H6: any "
            "interaction advantage remains after wall-clock accounting. H7: any advantage persists under held-out disturbances. "
            "Each hypothesis can fail; evidence is reported descriptively with paired training-seed comparisons.",
        ]),
        ("4. Methodology", [
            "All trainable actors use Linear(8,64), Tanh, Linear(64,64), Tanh, Linear(64,4). The parameter count is "
            "8*64+64+64*64+64+64*4+4=4,996. GA selects argmax logits; PPO trains a categorical policy. The PPO critic "
            "has separate 64-unit hidden layers and a scalar output. Observations and rewards are not normalized.",
            "Each GA chromosome concatenates weights and biases in layer order. Initialization uses independent uniform "
            "samples with the ordinary linear-layer fan-in bounds. Candidate fitness is mean return over common seeds "
            "within a generation; training seeds change between generations. Tournament selection, configurable uniform "
            "or blend crossover, per-gene Gaussian mutation, and copied elites generate the next population. Diversity "
            "is mean standard deviation across chromosome coordinates. Non-finite parameters and fitness raise errors.",
            "GA pseudocode: initialize population; reserve a complete worst-case generation; evaluate every candidate on "
            "common episode seeds; validate the generation winner; retain the best validation checkpoint; log statistics; "
            "copy elites and breed offspring; atomically checkpoint population, counters, champion and random-generator "
            "states; repeat while another generation fits. The first actor is also a validation candidate. Selection "
            "ties retain the earlier policy. This prevents discarding a useful initial actor.",
            "PPO optimizes Lclip=E[min(r_t*A_t, clip(r_t,1-epsilon,1+epsilon)*A_t)], where r_t is the current-to-old "
            "action-probability ratio and A_t is a generalized-advantage estimate. Clipping limits the incentive for "
            "large probability-ratio changes. The implementation also trains a value function and includes an entropy "
            "term. These are SB3 updates; no alternate PPO optimizer is substituted [1,6].",
            "Hybrid pseudocode: allocate a fraction of the budget to GA; select its actor on validation seeds; construct "
            "a fresh PPO actor-critic; copy three actor weight matrices and bias vectors; verify raw logits and actions "
            "on synthetic and actual environment-reset observations; verify the critic is unchanged; train PPO for the "
            "remaining actual budget rounded down to complete rollouts. Optimizer state is fresh. PPO-short starts "
            "from scratch for exactly the number of transitions actually used by the hybrid's PPO stage.",
        ]),
        ("5. Experimental setup", [
            f"Recorded mode: {cfg['name']}. Training seeds: {cfg['experiment']['seeds']}. Allocation: {budget:,} "
            f"transitions per GA, PPO and hybrid run. Final evaluation: {eval_n} episodes per saved selected model. "
            f"Validation uses {cfg['experiment']['validation_episodes']} episodes from seed "
            f"{cfg['evaluation']['validation_seed_start']}; final testing starts at {cfg['evaluation']['final_seed_start']}. "
            "Optimization seeds occupy separate integer domains at or above one million. Initialization, selection, "
            "crossover and mutation use separate reproducible random streams.",
            "Primary fairness uses an equal allocation ceiling, not artificially equal actual steps. GA reserves "
            "population_size * fitness_episodes * episode_limit transitions before a generation. This conservative "
            "rule leaves some unused allocation. Hybrid PPO receives the remaining actual allocation. SB3's usual "
            "rollout rounding is explicitly handled by rounding down; the transition counter verifies the result. "
            "Selection and final-test interactions are outside the primary optimization ceiling and reported separately.",
            "PPO validation occurs after optimizer updates at the first rollout boundary meeting the checkpoint interval "
            "and at the final update. Initial, selected, latest and final policies are distinguishable. Final evaluation "
            "uses the validation-selected model, which can be an earlier checkpoint than the last update. GA validation "
            "occurs once per generation; unequal selection opportunity is a limitation and its cost is disclosed.",
            f"Hardware: {manifest['processor']}; {manifest['logical_cpus']} logical CPUs; {manifest['ram_gib']} GiB RAM. "
            f"Software: Python {manifest['python']}, {manifest['platform']}. CPU training uses "
            f"{cfg['experiment']['threads']} Torch thread(s). Package versions and resolved configuration are stored "
            "with each run. Wall time includes optimization and model selection, and excludes final testing. "
            "The delivered experiments ran on a shared workstation while report generation and exploratory sweeps "
            "also ran. These timings describe observed elapsed cost, not an isolated-system speed benchmark.",
            "Uncertainty: compute each policy's mean across common test starts, then aggregate those means across "
            "independent training seeds. Percentile bootstrap intervals resample training-seed means (10,000 replicates). "
            "Intervals are omitted for fewer than three training seeds. With three seeds they are preliminary, not strong "
            "population evidence. Random-agent repetitions use the same fixed action RNG and are not independent training "
            "replications. Paired episode differences are descriptive; inference does not treat episodes as training seeds.",
        ]),
        ("6. Results", [comparison,
            "Tables report actual saved measurements. Dots in performance plots show training-seed means; error bars "
            "are seed-bootstrap intervals when available. Learning curves use validation observations and include the "
            "hybrid's GA-stage selected-champion measurements and GA cost on the primary x-axis. Separate plots show "
            "PPO-stage observations against PPO-only cost, total cost including validation, "
            "and wall time. Curves interpolate only within the range jointly observed across seeds; bands show one "
            "standard deviation, not confidence intervals.",
            "GA fitness curves need not increase monotonically because generation seeds change. Validation-selected "
            "champion scores can increase even when current-generation scores fall. Solved-episode rate means return "
            "at least 200. A resting landing is separately identified from the simulator's sleep and crash signals; "
            "it does not guarantee the landing pad was reached. Engine action counts are a usage proxy, not fuel consumed.",
        ]),
    ]
    if all(a in metrics.index for a in ("ga", "random")):
        ga_difference = metrics.loc["ga", "mean_return"] - metrics.loc["random", "mean_return"]
        sections[-1][1].append(
            f"The GA-minus-random mean-return difference is {ga_difference:+.1f} points. "
            f"GA's mean solved-episode rate is {100 * metrics.loc['ga', 'solved_rate']:.1f}%. "
            "Improvement over random behavior and solving the landing task are distinct outcomes.")
    if all(a in metrics.index for a in ("hybrid", "ppo", "ppo_short")):
        diff = metrics.loc["hybrid", "mean_return"] - metrics.loc["ppo", "mean_return"]
        short_diff = metrics.loc["hybrid", "mean_return"] - metrics.loc["ppo_short", "mean_return"]
        seed_pivot = per_seed.pivot(index="seed", columns="agent", values="mean_return")
        wins = int((seed_pivot.hybrid > seed_pivot.ppo).sum())
        time_diff = metrics.loc["hybrid", "training_seconds"] - metrics.loc["ppo", "training_seconds"]
        discussion = [
            f"The measured hybrid-minus-PPO difference is {diff:+.1f} return points, and the hybrid-minus-PPO-short "
            f"difference is {short_diff:+.1f}. These comparisons must be read alongside seed variation, unequal "
            "unused residuals, checkpoint selection and computational overhead. A positive point estimate alone is "
            "insufficient to establish a robust benefit; a negative estimate is evidence against the proposed advantage "
            "for this configuration.",
            f"The hybrid exceeds PPO in {wins} of {n} matched training seeds. Its mean training-and-selection "
            f"wall time differs from PPO by {time_diff:+.1f} seconds. "
            + ("The point estimate favors the hybrid, but the small seed count and validation-selection effects "
               "limit the strength of that finding." if diff > 0 else
               "The point estimate does not support a hybrid advantage over full-budget PPO in this experiment. "
               "Exact initialization transfer therefore should not be confused with improved sample efficiency."),
        ]
    else:
        discussion = ["The baseline suite is incomplete; absent comparisons are not inferred."]
    discussion += [
        "The transfer test answers an engineering question: whether PPO starts from the actual evolved actor. It does "
        "not demonstrate that this actor is easy to fine-tune. Deterministic GA behavior and stochastic PPO exploration "
        "can differ sharply even when their logits match. A fresh critic may initially provide noisy advantages. "
        "Useful GA behavior may be lost during early policy-gradient updates.",
        "The selected checkpoint can precede the final optimizer update. Reported final-test performance therefore "
        "means performance after the complete training-and-selection procedure, not necessarily the last policy. "
        "Sensitivity findings use a distinct development evaluation set; they should not be used to tune against the "
        "main test set. Repeated validation across many generations can overfit the small validation set.",
        "The return-per-interaction ratio is provided for completeness but is not a stable efficiency ranking when "
        "returns can be negative or shifted by constants. Observed threshold crossings and learning curves are more "
        "interpretable. Thresholds that were never observed are marked not reached within budget.",
    ]
    sections.extend([
        ("7. Discussion and limitations", discussion),
        ("8. Conclusion", [
            f"The project implements and measures a reproducible {mode} comparison of evolutionary search, PPO and "
            "GA-initialized PPO. It establishes exact actor transfer, functioning gradient refinement, saved-model "
            "evaluation and transparent budget accounting. The recorded results above, rather than an assumed hybrid "
            "advantage, determine the comparative conclusion.",
            "RQ1 and RQ2 are assessed through GA validation and transfer records. RQ3 and RQ4 are assessed through "
            "observed learning curves and paired final performance. RQ5 requires the sensitivity runs reported in the "
            "supplement status below. RQ6 is addressed by measured runtime and total interactions. RQ7 remains limited "
            "by the number of independent training seeds and the robustness conditions actually executed. No unexecuted "
            "experiment is treated as supporting evidence.",
        ]),
        ("9. Future work", [
            "Extend to five or more training seeds, larger held-out validation sets, matched selection overhead, "
            "adaptive mutation, alternating evolution and PPO, population-based hyperparameter training, structured "
            "crossover, and continuous-control tasks. Consider critic warm-up and actor entropy diagnostics to explain "
            "why an evolved deterministic policy may be unstable during stochastic fine-tuning.",
        ]),
    ])
    tables = []
    rows = []
    for r in aggregate.itertuples():
        ci = f"[{r.ci_low:.1f}, {r.ci_high:.1f}]" if pd.notna(r.ci_low) else "not estimated"
        rows.append([LABELS[r.agent], str(r.training_seeds), f"{r.mean_return:.1f}",
                     f"{r.seed_sd:.1f}" if pd.notna(r.seed_sd) else "n/a", ci, f"{100*r.solved_rate:.1f}%"])
    tables.append(("Performance across training seeds", ["Agent", "Seeds", "Return", "Seed SD", "95% bootstrap CI", "Solved"], rows))
    tables.append(("Mean resource use per independent run", ["Agent", "Optimize", "Select", "Test", "Unused", "Seconds"],
                   [[LABELS[r.agent], f"{r.optimization_steps:,.0f}", f"{r.selection_steps:,.0f}",
                     f"{r.test_steps:,.0f}", f"{r.unused_steps:,.0f}", f"{r.training_seconds:.1f}"]
                    for r in aggregate.itertuples()]))
    paired_path = suite / "analysis" / "paired_contrasts.csv"
    paired = pd.read_csv(paired_path) if paired_path.stat().st_size > 5 else pd.DataFrame()
    if not paired.empty:
        tables.append(("Paired training-seed contrasts", ["Contrast", "Pairs", "Difference", "95% bootstrap CI"],
                       [[r.contrast, str(r.training_seed_pairs), f"{r.mean_difference:+.1f}",
                         f"[{r.ci_low:.1f}, {r.ci_high:.1f}]" if pd.notna(r.ci_low) else "not estimated"]
                        for r in paired.itertuples()]))
    transfer_rows, stage_rows = [], []
    for directory, summary in suite_runs(suite):
        if summary["agent"] == "hybrid":
            t = read_json(directory / "transfer.json")
            transfer_rows.append([str(summary["seed"]), f"{t['ga_initial_validation']:.1f}",
                                  f"{t['ga_selected_validation']:.1f}", f"{t['post_transfer_validation']:.1f}",
                                  f"{t['max_logit_error']:.2e}", str(t["actor_changed_after_ppo"])])
            validation = pd.read_csv(directory / "ppo" / "validation.csv")
            stage_rows.append([str(summary["seed"]), f"{t['ga_initial_validation']:.1f}",
                               f"{t['ga_selected_validation']:.1f}", f"{validation.mean_return.iloc[-1]:.1f}",
                               f"{validation.mean_return.max():.1f}",
                               f"{validation.mean_return.iloc[-1] - t['ga_selected_validation']:+.1f}"])
    if transfer_rows:
        tables.append(("Actor transfer audit (validation returns)", ["Seed", "Initial", "After GA", "Transfer", "Logit error", "Updated"], transfer_rows))
        tables.append(("Hybrid stage trajectory (same validation starts)",
                       ["Seed", "Before GA", "After GA", "Final PPO", "Best PPO", "Final gain"], stage_rows))
    threshold_path = suite / "analysis" / "thresholds.csv"
    if threshold_path.exists() and threshold_path.stat().st_size > 5:
        thresholds = pd.read_csv(threshold_path)
        tables.append(("First observed validation threshold (optimization steps)", ["Agent", "Seed", "Threshold", "Observed crossing"],
                       [[LABELS[r.agent], str(r.seed), str(r.threshold),
                         f"{r.first_observed_steps:,.0f}" if pd.notna(r.first_observed_steps) else "not reached"]
                        for r in thresholds.itertuples()]))
    tables.append(("Per-seed final-test means", ["Agent", "Seed", "Return", "Solved", "Selected PPO step"],
                   [[LABELS[r.agent], str(r.seed), f"{r.mean_return:.1f}", f"{100*r.solved_rate:.1f}%",
                     f"{r.selected_ppo_steps:,.0f}" if pd.notna(r.selected_ppo_steps) else "n/a"]
                    for r in per_seed.itertuples()]))
    return manifest, abstract, sections, tables


def generate_report(suite: Path, output: Path = Path("report"), supplements: list[Path] | None = None) -> Path:
    figures = generate_plots(suite)
    output.mkdir(parents=True, exist_ok=True)
    bibliography = Path("report/references.bib")
    if bibliography.exists() and bibliography.resolve() != (output / "references.bib").resolve():
        shutil.copy2(bibliography, output / "references.bib")
    target_figures = output / "figures"
    target_figures.mkdir(exist_ok=True)
    for source in figures.glob("*.png"):
        shutil.copy2(source, target_figures / source.name)
    manifest, abstract, sections, tables = build_content(suite)
    supplement_paragraphs, supplement_figures = [], []
    for path in supplements or []:
        from .plotting import plot_supplement
        plot_dir = plot_supplement(path)
        for source in plot_dir.glob("*.png"):
            shutil.copy2(source, target_figures / source.name)
            supplement_figures.append(source.stem)
        if (path / "sweep_results.csv").exists():
            data = pd.read_csv(path / "sweep_results.csv")
            supplement_paragraphs.append(f"Executed sensitivity source: {path.as_posix()}. "
                                         f"{len(data)} measured agent runs; development seeds start at 60,000. "
                                         "The sensitivity budget can differ from the main study, so these results are exploratory.")
            grouped = data.groupby(["parameter", "value", "agent"]).agg(
                mean_return=("mean_return", "mean"), seeds=("seed", "nunique"),
                allocated=("allocated_steps", "mean"), generations=("generations", "mean")).reset_index()
            if grouped.seeds.min() < 3 or grouped.allocated.min() < 1_000_000:
                supplement_paragraphs.append("The full three-seed, one-million-transition-per-agent sensitivity "
                                             "suite was not executed. The delivered smaller sweep tests the pipeline "
                                             "and provides exploratory measurements; it does not replace confirmatory ablations.")
            for parameter, variants in grouped[grouped.agent == "hybrid"].groupby("parameter"):
                winner = variants.loc[variants.mean_return.idxmax()]
                supplement_paragraphs.append(
                    f"Within the tested hybrid variants for {parameter}, the highest development mean was "
                    f"{winner.mean_return:.1f} at {winner.value:g}. This is an exploratory selection across "
                    f"{len(variants)} settings, not an independently confirmed optimum. The baseline "
                    "configuration was not changed in response to these results.")
            tables.append(("Sensitivity results (development hold-out)", ["Parameter", "Value", "Agent", "Seeds", "Budget", "Gen.", "Return"],
                           [[{"ga.mutation_sigma": "Mutation SD", "ga.population_size": "Population",
                              "hybrid.ga_budget_fraction": "GA allocation",
                              "ga.crossover_probability": "Crossover prob."}.get(r.parameter, r.parameter),
                             str(r.value), LABELS[r.agent],
                             str(r.seeds), f"{r.allocated:,.0f}", f"{r.generations:.0f}",
                             f"{r.mean_return:.1f}"] for r in grouped.itertuples()]))
        elif (path / "robustness_results.csv").exists():
            data = pd.read_csv(path / "robustness_results.csv")
            supplement_paragraphs.append(f"Executed robustness source: {path.as_posix()}. "
                                         "Selected policies were evaluated without adaptation using identical perturbations "
                                         "and episode seeds across methods. This is a separate out-of-distribution evaluation.")
            grouped = data.groupby(["condition", "agent"]).mean_return.mean().reset_index()
            pivot = grouped.pivot(index="condition", columns="agent", values="mean_return")
            if "default" in pivot.index:
                for agent in ("ppo", "hybrid"):
                    if agent in pivot:
                        worst = pivot[agent].idxmin()
                        supplement_paragraphs.append(
                            f"For {LABELS[agent]}, the lowest mean among the recorded robustness conditions "
                            f"was {pivot.loc[worst, agent]:.1f} under {worst}, compared with "
                            f"{pivot.loc['default', agent]:.1f} under the default condition. These are frozen-policy "
                            "evaluations, with the same test starts for every method.")
            tables.append(("Robustness: mean return", ["Condition", "Agent", "Return"],
                           [[r.condition, LABELS[r.agent], f"{r.mean_return:.1f}"] for r in grouped.itertuples()]))
    if not supplements:
        supplement_paragraphs = ["Mutation, population-size, GA-budget, crossover, and robustness experiments: not executed "
                                 "for this report. Configuration and executable pipelines are provided."]
    sections.insert(6, ("6.1. Sensitivity and robustness execution status", supplement_paragraphs))
    cfg = manifest["config"]
    hyper_rows = [[f"{section}.{key}", str(value)] for section in
                  ("environment", "experiment", "network", "ga", "ppo", "hybrid", "evaluation")
                  for key, value in cfg[section].items()]
    tables.append(("Resolved hyperparameters", ["Setting", "Value"], hyper_rows))
    date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")
    sections.append(("Appendix: reproducibility and artifact lineage", [
        f"Primary data source: {suite.as_posix()}. Configuration hash: {manifest['config_hash']}. "
        "The report was generated from that explicit suite; no implicit latest-run selection is used.",
        "Install using Python 3.11 and the saved version snapshot, run scripts.doctor, then pytest. Execute "
        "scripts.run_all with the appropriate YAML. Regenerate using scripts.generate_report --run <suite>. "
        "See README.md and REPRODUCIBILITY.md for exact PowerShell commands and the execution record.",
        "GA checkpoints store NumPy arrays plus JSON metadata, without object pickle. SB3 model files use its native "
        "serialization and should only be loaded from trusted sources. GA resumes exactly at a generation boundary "
        "under the same software. PPO models save optimizer state, but restarting the environment is not bitwise "
        "equivalent to an uninterrupted run. The suite safely archives interrupted attempts and restarts that agent.",
        "The PDF is generated with ReportLab because no LaTeX compiler was found during implementation. The supplied "
        "main.tex and references.bib provide an alternative source build: pdflatex main; bibtex main; pdflatex main; "
        "pdflatex main. Numerical results are shared across Markdown, LaTeX and PDF outputs.",
    ]))
    md = [f"# {TITLE}", "Course: Applied Reinforcement Learning and Optimization",
          "Student: [Your name] | Institution: [Your institution]", f"Date: {date}", "## Abstract", abstract]
    figure_names = ["architecture", "final_performance", "learning_curves", "ga_evolution", "ga_diversity",
                    "return_distribution", "runtime", "interaction_accounting", "ppo_only_axis",
                    "total_interactions", "wall_time_learning", "solved_rates"] + supplement_figures
    figure_names = [name for name in figure_names if (figures / f"{name}.png").exists() or name in supplement_figures]
    for kind, block in content_blocks(sections, tables, figure_names):
        if kind == "section":
            title, paragraphs = block
            md.extend([f"## {title}", *paragraphs])
        elif kind == "table":
            title, headers, rows = block
            md.extend([f"### {title}", md_table(headers, rows)])
        else:
            md.append(f"![{block.replace('_', ' ')}](figures/{block}.png)")
    md.append("## References")
    md.extend(f"{i}. [{text}]({url})" for i, (text, url) in enumerate(REFERENCES, 1))
    (output / "report.md").write_text("\n\n".join(md), encoding="utf-8")
    write_tex(output, date, abstract, sections, tables, figure_names)
    write_pdf(output, date, abstract, sections, tables, figure_names)
    files = [suite / "manifest.json", *suite.glob("*_seed*/summary.json"), *suite.glob("*_seed*/evaluation.csv")]
    for supplement in supplements or []:
        files.extend(supplement.glob("*.csv"))
        files.extend(supplement.glob("manifest.json"))
    write_json(output / "provenance.json", {"suite": str(suite), "supplements": [str(p) for p in supplements or []],
                                            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    return output / "report.pdf"


def write_tex(output, date, abstract, sections, tables, figures):
    lines = [r"\documentclass[11pt,a4paper]{article}", r"\usepackage[margin=24mm]{geometry}",
             r"\usepackage[T1]{fontenc}", r"\usepackage{graphicx,longtable,booktabs,hyperref}",
             r"\usepackage{float}", r"\hypersetup{colorlinks=true,urlcolor=blue}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\setlength{\parskip}{0.5em}", r"\setlength{\parindent}{0pt}",
             r"\title{" + tex_escape(TITLE) + "}",
             r"\author{[Your name] \\ [Your institution] \\ Applied Reinforcement Learning and Optimization}",
             r"\date{" + date + "}", r"\begin{document}", r"\maketitle", r"\begin{abstract}",
             tex_escape(abstract), r"\end{abstract}", r"\newpage", r"\tableofcontents", r"\newpage"]
    def add_section(title, paragraphs):
        lines.append(r"\section*{" + tex_escape(title) + "}")
        lines.append(r"\addcontentsline{toc}{section}{" + tex_escape(title) + "}")
        lines.extend(tex_prose(p) + "\n" for p in paragraphs)
    def add_table(title, headers, rows):
        cols = len(headers)
        widths = [0.85 / cols] * cols
        if cols == 2:
            widths = [0.50, 0.35]
        lines.extend([r"\subsection*{" + tex_escape(title) + "}", r"\begingroup\small",
                      r"\begin{longtable}{" + "".join(f"p{{{w:.3f}\\textwidth}}" for w in widths) + "}",
                      r"\toprule", " & ".join(tex_escape(h) for h in headers) + r" \\", r"\midrule\endhead"])
        lines.extend(" & ".join(tex_escape(cell) for cell in row) + r" \\" for row in rows)
        lines.extend([r"\bottomrule\end{longtable}", r"\endgroup"])
    def add_figure(name):
        lines.extend([r"\begin{figure}[htbp]\centering",
                      r"\includegraphics[width=0.96\textwidth]{figures/" + name + ".png}",
                      r"\caption{" + tex_escape(name.replace("_", " ").capitalize()) + "}", r"\end{figure}"])
    for kind, block in content_blocks(sections, tables, figures):
        if kind == "section":
            add_section(*block)
        elif kind == "table":
            add_table(*block)
        else:
            add_figure(block)
    lines.extend([r"\clearpage", r"\nocite{*}", r"\bibliographystyle{plain}", r"\bibliography{references}",
                  r"\end{document}"])
    (output / "main.tex").write_text("\n".join(lines), encoding="utf-8")


def write_pdf(output, date, abstract, sections, tables, figures):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=24, leading=29,
                             textColor=colors.HexColor("#12384A"), spaceAfter=24))
    styles.add(ParagraphStyle("Deck", fontSize=12, leading=18, textColor=colors.HexColor("#436373"), spaceAfter=14))
    styles.add(ParagraphStyle("ReportBody", fontSize=10, leading=14, spaceAfter=10))
    styles.add(ParagraphStyle("TableCell", fontSize=8, leading=10))
    styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=12))
    story = [Spacer(1, 0.5 * inch), Paragraph("APPLIED REINFORCEMENT LEARNING &amp; OPTIMIZATION", styles["Deck"]),
             Paragraph(html.escape(TITLE), styles["ReportTitle"]),
             Paragraph("Course project | Experimental research report", styles["Deck"]),
             Paragraph("Student: [Your name]<br/>Institution: [Your institution]<br/>" + date, styles["ReportBody"]),
             Spacer(1, 0.25 * inch), Paragraph("Abstract", styles["Heading2"]),
             Paragraph(html.escape(abstract), styles["ReportBody"]), PageBreak()]
    def add_section(title, paragraphs):
        story.append(Paragraph(html.escape(title), styles["Heading1"]))
        story.extend(Paragraph(html.escape(p), styles["ReportBody"]) for p in paragraphs)
    def add_table(title, headers, rows):
        story.append(Paragraph(html.escape(title), styles["Heading2"]))
        data = [[Paragraph(html.escape(str(c)), styles["TableCell"]) for c in row] for row in [headers] + rows]
        width = 7.0 * inch
        widths = [width / len(headers)] * len(headers)
        if len(headers) == 2:
            widths = [width * 0.58, width * 0.42]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCECEF")),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                                   ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#00866A")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.extend([table, Spacer(1, 0.18 * inch)])
    def add_figure(index, name):
        path = output / "figures" / f"{name}.png"
        image = Image(str(path))
        image.drawHeight *= (7 * inch / image.drawWidth)
        image.drawWidth = 7 * inch
        story.append(KeepTogether([image, Paragraph(f"Figure {index}. {name.replace('_', ' ').capitalize()}.",
                                                   styles["Caption"])]))
    figure_number = 0
    for kind, block in content_blocks(sections, tables, figures):
        if kind == "section":
            add_section(*block)
        elif kind == "table":
            add_table(*block)
        else:
            figure_number += 1
            add_figure(figure_number, block)
    story.append(PageBreak())
    story.append(Paragraph("References", styles["Heading1"]))
    for index, (text, url) in enumerate(REFERENCES, 1):
        story.append(Paragraph(f'{index}. {html.escape(text)} <link href="{url}">Source</link>.', styles["ReportBody"]))

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#BCD1D8"))
        canvas.line(0.65 * inch, 0.56 * inch, 7.65 * inch, 0.56 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#496674"))
        canvas.drawString(0.65 * inch, 0.38 * inch, "GA-RL LunarLander | Measured, reproducible experiments")
        canvas.drawRightString(7.65 * inch, 0.38 * inch, str(document.page))

    document = SimpleDocTemplate(str(output / "report.pdf"), pagesize=(8.3 * inch, 11.7 * inch),
                                 leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                                 topMargin=0.65 * inch, bottomMargin=0.75 * inch,
                                 title=TITLE, author="Course project")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
