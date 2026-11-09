"""Figures from recorded measurements; no extrapolation or synthetic learning data."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .experiment import suite_runs
from .statistics import analyze

COLORS = {"random": "#8D99AE", "ga": "#E09F3E", "ppo": "#277DA8", "hybrid": "#00866A",
          "ppo_short": "#9155A7"}
LABELS = {"random": "Random", "ga": "GA", "ppo": "PPO", "hybrid": "GA-PPO", "ppo_short": "PPO-short"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def architecture(directory: Path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    boxes = [(0.1, "Population\n4,996 weights / actor"), (3.5, "Genetic search\nValidation-selected actor"),
             (6.9, "PPO fine-tuning\nFresh, separate critic")]
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.1), 2.8, 1.1, boxstyle="round,pad=0.10",
                                    facecolor="#EDF5F5", edgecolor="#00866A"))
        ax.text(x + 1.4, 1.65, label, ha="center", va="center", fontsize=11)
    for x in (2.95, 6.35):
        ax.add_patch(FancyArrowPatch((x, 1.65), (x + 0.45, 1.65), arrowstyle="-|>", mutation_scale=14))
    ax.text(5, 0.4, "Shared actor: 8 → 64 → 64 → 4  |  Tanh hidden layers  |  Raw observations",
            ha="center", fontsize=11)
    ax.set_title("Two-stage GA-initialized PPO", fontsize=15)
    save(fig, directory, "architecture")


def curve_plot(curves: pd.DataFrame, x: str, xlabel: str, directory: Path, filename: str, agents=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for agent, group in curves.groupby("agent", sort=False):
        if agents and agent not in agents:
            continue
        runs = [r.sort_values(x).drop_duplicates(x) for _, r in group.groupby("seed")]
        for run in runs:
            ax.plot(run[x], run.mean_return, color=COLORS[agent], alpha=0.24, linewidth=0.8)
        left, right = max(r[x].min() for r in runs), min(r[x].max() for r in runs)
        if right > left:
            grid = np.linspace(left, right, 120)
            values = np.stack([np.interp(grid, r[x], r.mean_return) for r in runs])
            ax.plot(grid, values.mean(0), color=COLORS[agent], label=LABELS[agent], linewidth=2)
            if len(runs) > 1:
                deviation = values.std(0, ddof=1)
                ax.fill_between(grid, values.mean(0) - deviation, values.mean(0) + deviation,
                                color=COLORS[agent], alpha=0.12)
        else:
            ax.plot(runs[0][x], runs[0].mean_return, color=COLORS[agent], label=LABELS[agent])
    ax.axhline(200, color="#666666", linestyle="--", linewidth=0.7, label="200-point threshold")
    ax.set(xlabel=xlabel, ylabel="Mean validation return", title="Learning progress on held-out validation starts")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    fig.text(0.12, -0.015, "Bands: ±1 SD across training seeds; interpolation only within shared observed ranges.",
             fontsize=8, color="#555555")
    save(fig, directory, filename)


def generate_plots(suite: Path) -> Path:
    style()
    per_seed, aggregate = analyze(suite)
    directory = suite / "plots"
    architecture(directory)
    order = [a for a in COLORS if a in set(aggregate.agent)]
    aggregate = aggregate.set_index("agent").loc[order].reset_index()
    for column, ylabel, filename, title in (
        ("mean_return", "Mean final-test return", "final_performance", "Selected-policy performance"),
        ("solved_rate", "Fraction with return ≥ 200", "solved_rates", "Solved episodes (reward criterion)"),
        ("training_seconds", "Training and selection time (seconds)", "runtime", "Computational cost"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.bar(range(len(order)), aggregate[column], color=[COLORS[a] for a in order], alpha=0.8)
        if column == "mean_return":
            finite = aggregate.ci_low.notna()
            subset = aggregate.loc[finite]
            if len(subset):
                ax.errorbar(np.flatnonzero(finite), subset[column],
                            yerr=np.array([subset[column] - subset.ci_low,
                                           subset.ci_high - subset[column]]), fmt="none", color="#333333", capsize=4)
        for i, agent in enumerate(order):
            values = per_seed.loc[per_seed.agent == agent, column]
            ax.scatter(np.linspace(i - 0.10, i + 0.10, len(values)), values, color="#333333", s=20, zorder=3)
        ax.set(xticks=range(len(order)), xticklabels=[LABELS[a] for a in order], ylabel=ylabel, title=title)
        if column == "solved_rate":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.15)
        save(fig, directory, filename)
    episodes = pd.read_csv(suite / "analysis" / "episodes.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bp = ax.boxplot([episodes.loc[episodes.agent == a, "return"] for a in order],
                    tick_labels=[LABELS[a] for a in order], patch_artist=True, showfliers=False)
    for patch, agent in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(COLORS[agent])
        patch.set_alpha(0.65)
    ax.set(ylabel="Episode return", title="Test episode distributions (pooled descriptively)")
    save(fig, directory, "return_distribution")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(order))
    for key, label, color in [("optimization_steps", "Optimization", "#277DA8"),
                               ("selection_steps", "Validation / selection", "#E09F3E"),
                               ("test_steps", "Final testing", "#8D99AE")]:
        ax.bar(range(len(order)), aggregate[key], bottom=bottom, label=label, color=color)
        bottom += aggregate[key].to_numpy()
    ax.set(xticks=range(len(order)), xticklabels=[LABELS[a] for a in order], ylabel="Mean interactions",
           title="Every interaction has an attributed cost")
    ax.legend(fontsize=8)
    save(fig, directory, "interaction_accounting")
    curves_path = suite / "analysis" / "learning_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        for key, label, name in (("optimization_steps", "Optimization transitions (GA + PPO)", "learning_curves"),
                                  ("ppo_steps", "PPO transitions only (excludes GA cost)", "ppo_only_axis"),
                                  ("total_steps", "Optimization + accumulated validation transitions", "total_interactions"),
                                  ("elapsed_seconds", "Training and validation wall time (seconds)", "wall_time_learning")):
            subset = curves[curves.stage == "ppo"] if key == "ppo_steps" else curves
            curve_plot(subset, key, label, directory, name)
        for agent in ("ppo", "hybrid"):
            curve_plot(curves, "optimization_steps", "Optimization transitions", directory,
                       f"{agent}_learning", [agent])
    histories = []
    for run, summary in suite_runs(suite):
        path = run / "ga" / "generations.csv"
        if path.exists():
            history = pd.read_csv(path)
            history["agent"], history["seed"] = summary["agent"], summary["seed"]
            histories.append(history)
    if histories:
        history = pd.concat(histories)
        history.to_csv(suite / "analysis" / "ga_history.csv", index=False)
        for x, ys, name in (("generation", ["best", "mean"], "ga_evolution"),
                             ("optimization_steps", ["best", "mean"], "ga_interactions"),
                             ("generation", ["diversity"], "ga_diversity")):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for (agent, seed), group in history.groupby(["agent", "seed"]):
                for y in ys:
                    ax.plot(group[x], group[y], color=COLORS[agent], alpha=0.7,
                            linestyle="--" if y == "mean" else "-", label=f"{LABELS[agent]} seed {seed}: {y}")
            ax.set(xlabel="Generation" if x == "generation" else "GA optimization transitions",
                   ylabel="Mean per-gene population SD" if name == "ga_diversity" else "Training fitness (return)",
                   title="Genetic search: measured generation statistics")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.15)
            save(fig, directory, name)
    return directory


def plot_supplement(path: Path) -> Path:
    style()
    directory = path / "plots"
    sweep = path / "sweep_results.csv"
    robust = path / "robustness_results.csv"
    if sweep.exists():
        frame = pd.read_csv(sweep)
        for parameter, group in frame.groupby("parameter"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for agent, runs in group.groupby("agent"):
                stats = runs.groupby("value").mean_return.agg(["mean", "std"])
                if stats["std"].notna().any():
                    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o",
                                color=COLORS[agent], label=LABELS[agent], capsize=3)
                else:
                    ax.plot(stats.index, stats["mean"], marker="o", color=COLORS[agent], label=LABELS[agent])
            labels = {"ga.mutation_sigma": "Mutation standard deviation", "ga.population_size": "Population size",
                      "hybrid.ga_budget_fraction": "Fraction allocated to GA",
                      "ga.crossover_probability": "Crossover probability"}
            seeds = group.seed.nunique()
            ax.set(xlabel=labels.get(parameter, parameter), ylabel="Development evaluation return",
                   title=f"Exploratory sensitivity ({seeds} training seed{'s' if seeds != 1 else ''})",
                   xticks=sorted(group.value.unique()))
            ax.legend()
            save(fig, directory, parameter.replace(".", "_"))
    elif robust.exists():
        frame = pd.read_csv(robust)
        table = frame.pivot_table(index="condition", columns="agent", values="mean_return")
        ax = table.plot.bar(figsize=(10, 5), color=[COLORS[a] for a in table.columns], rot=15)
        ax.set(ylabel="Mean test return", title="Robustness: fixed policies, changed environment")
        save(ax.figure, directory, "robustness")
    else:
        raise ValueError("Expected sweep_results.csv or robustness_results.csv")
    return directory
