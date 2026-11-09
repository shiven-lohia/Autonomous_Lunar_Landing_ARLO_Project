"""Training-seed uncertainty, paired contrasts and observed threshold crossings."""

from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import suite_runs


def bootstrap_mean(values, seed: int = 812, samples: int = 10000) -> tuple:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def analyze(suite: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = suite / "analysis"
    destination.mkdir(exist_ok=True)
    rows, episodes, curves, thresholds = [], [], [], []
    for directory, summary in suite_runs(suite):
        rows.append(summary)
        ep = pd.read_csv(directory / "evaluation.csv")
        ep["agent"], ep["seed"] = summary["agent"], summary["seed"]
        episodes.append(ep)
        path = directory / "ppo" / "validation.csv"
        if path.exists():
            curve = pd.read_csv(path)
            curve["stage"] = "ppo"
            history_path = directory / "ga" / "generations.csv"
            if history_path.exists():
                history = pd.read_csv(history_path)
                evolutionary = pd.DataFrame({
                    "ppo_steps": 0, "optimization_steps": history.optimization_steps,
                    "selection_steps": history.selection_steps,
                    "total_steps": history.optimization_steps + history.selection_steps,
                    "mean_return": history.selected_validation_return,
                    "elapsed_seconds": history.elapsed_seconds, "stage": "ga"})
                curve = pd.concat([evolutionary, curve], ignore_index=True)
            curve["agent"], curve["seed"] = summary["agent"], summary["seed"]
            curves.append(curve)
            for threshold in (0, 100, 200):
                crossed = curve[curve.mean_return >= threshold]
                thresholds.append({"agent": summary["agent"], "seed": summary["seed"],
                                   "threshold": threshold,
                                   "first_observed_steps": int(crossed.optimization_steps.iloc[0]) if len(crossed) else None,
                                   "status": "reached" if len(crossed) else "not reached within budget"})
    if not rows:
        raise ValueError("No completed agent runs in this suite")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(destination / "per_seed.csv", index=False)
    all_episodes = pd.concat(episodes, ignore_index=True)
    all_episodes.to_csv(destination / "episodes.csv", index=False)
    aggregates = []
    for agent, group in per_seed.groupby("agent", sort=False):
        low, high = bootstrap_mean(group.mean_return)
        if agent == "random":
            low = high = np.nan
        aggregates.append({"agent": agent, "training_seeds": len(group) if agent != "random" else 0,
                           "mean_return": group.mean_return.mean(),
                           "seed_sd": group.mean_return.std(ddof=1) if agent != "random" else np.nan,
                           "ci_low": low, "ci_high": high, "solved_rate": group.solved_rate.mean(),
                           "resting_landing_rate": group.resting_landing_rate.mean(),
                           "training_seconds": group.training_seconds.mean(),
                           "optimization_steps": group.optimization_steps.mean(),
                           "selection_steps": group.selection_steps.mean(), "test_steps": group.test_steps.mean(),
                           "unused_steps": group.unused_steps.mean()})
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(destination / "comparison.csv", index=False)
    pd.DataFrame(thresholds).to_csv(destination / "thresholds.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(destination / "learning_curves.csv", index=False)
    paired = []
    pivot = per_seed.pivot(index="seed", columns="agent", values="mean_return")
    if "hybrid" in pivot:
        for comparator in ("ppo", "ppo_short", "ga"):
            if comparator not in pivot:
                continue
            differences = (pivot.hybrid - pivot[comparator]).dropna()
            low, high = bootstrap_mean(differences)
            paired.append({"contrast": f"hybrid - {comparator}", "training_seed_pairs": len(differences),
                           "mean_difference": differences.mean(), "ci_low": low, "ci_high": high})
    pd.DataFrame(paired).to_csv(destination / "paired_contrasts.csv", index=False)
    episode_pivot = all_episodes.pivot(index=["seed", "episode_seed"], columns="agent", values="return")
    if "hybrid" in episode_pivot and "ppo" in episode_pivot:
        (episode_pivot.hybrid - episode_pivot.ppo).rename("hybrid_minus_ppo").to_csv(
            destination / "paired_episode_differences.csv")
    return per_seed, aggregate
