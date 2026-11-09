"""Configuration loading and explicit scientific constraints."""

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT = {
    "name": "full",
    "environment": {"id": "LunarLander-v3", "continuous": False, "enable_wind": False,
                    "gravity": -10.0, "max_episode_steps": 1000},
    "experiment": {"seeds": [11, 23, 37], "total_optimization_steps": 1_000_000,
                   "final_eval_episodes": 100, "checkpoint_interval": 50000,
                   "validation_episodes": 5, "threads": 1},
    "network": {"actor_hidden": [64, 64], "critic_hidden": [64, 64], "activation": "tanh"},
    "ga": {"population_size": 20, "elite_count": 2, "tournament_size": 3,
           "mutation_probability": 0.05, "mutation_sigma": 0.05,
           "crossover_probability": 0.7, "crossover_type": "uniform",
           "fitness_episodes": 3, "immigrant_rate": 0.0, "max_generations": None},
    "ppo": {"learning_rate": 0.0003, "gamma": 0.999, "gae_lambda": 0.98,
            "clip_range": 0.2, "n_steps": 1024, "batch_size": 64, "n_epochs": 4,
            "ent_coef": 0.01, "n_envs": 4, "device": "cpu"},
    "hybrid": {"ga_budget_fraction": 0.25, "transfer_actor_only": True},
    "evaluation": {"deterministic": True, "solved_reward_threshold": 200,
                   "validation_seed_start": 20000, "final_seed_start": 40000},
}


def merge(base: dict, overrides: dict) -> dict:
    """Deep merge without modifying inputs."""
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and key in result else value
    return result


def validate(cfg: dict) -> dict:
    """Reject unsupported architectures, seed leakage, and invalid budgets."""
    g, p, e = cfg["ga"], cfg["ppo"], cfg["experiment"]
    if not e["seeds"] or len(set(e["seeds"])) != len(e["seeds"]):
        raise ValueError("Provide nonempty, unique experiment seeds")
    if any(not isinstance(s, int) or not 0 <= s < 10000 for s in e["seeds"]):
        raise ValueError("Experiment seeds must be integers in [0, 10000)")
    if not 1 <= g["elite_count"] < g["population_size"]:
        raise ValueError("Population must exceed positive elite count")
    if not 1 <= g["tournament_size"] <= g["population_size"]:
        raise ValueError("Invalid tournament size")
    for key in ("mutation_probability", "crossover_probability", "immigrant_rate"):
        if not 0 <= g[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if g["mutation_sigma"] < 0 or g["fitness_episodes"] < 1:
        raise ValueError("Invalid mutation sigma or fitness episode count")
    if g["crossover_type"] not in ("uniform", "blend"):
        raise ValueError("Crossover must be uniform or blend")
    if g["max_generations"] is not None and g["max_generations"] < 1:
        raise ValueError("max_generations must be positive or null")
    if not 0 < cfg["hybrid"]["ga_budget_fraction"] < 1:
        raise ValueError("GA budget fraction must lie strictly between 0 and 1")
    if not cfg["hybrid"]["transfer_actor_only"]:
        raise ValueError("Primary protocol transfers actor only")
    if not cfg["evaluation"]["deterministic"]:
        raise ValueError("Primary evaluation is deterministic; use evaluate_models --stochastic separately")
    if p["n_steps"] < 2 or p["n_envs"] < 1 or p["batch_size"] < 2:
        raise ValueError("Invalid PPO rollout/batch dimensions")
    if (p["n_steps"] * p["n_envs"]) % p["batch_size"]:
        raise ValueError("PPO batch_size must divide rollout size")
    if e["total_optimization_steps"] < p["n_steps"] * p["n_envs"]:
        raise ValueError("Budget cannot fund one PPO rollout")
    if min(e["validation_episodes"], e["final_eval_episodes"], e["checkpoint_interval"], e["threads"]) < 1:
        raise ValueError("Episode counts, intervals and thread count must be positive")
    if cfg["network"] != DEFAULT["network"]:
        raise ValueError("This study fixes 8-64-64-4 Tanh actors and 8-64-64-1 critics")
    env = cfg["environment"]
    if env["id"] != "LunarLander-v3" or env["continuous"] or env["max_episode_steps"] < 1:
        raise ValueError("Expected discrete LunarLander-v3 with a positive time limit")
    if not -12 < env["gravity"] < 0:
        raise ValueError("Gravity must lie in (-12,0)")
    val = set(range(cfg["evaluation"]["validation_seed_start"],
                    cfg["evaluation"]["validation_seed_start"] + e["validation_episodes"]))
    test = set(range(cfg["evaluation"]["final_seed_start"],
                     cfg["evaluation"]["final_seed_start"] + e["final_eval_episodes"]))
    if val & test or min(val | test) < 10000 or max(val | test) >= 1_000_000:
        raise ValueError("Validation/test seeds must be disjoint and in [10000,1000000)")
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate(merge(DEFAULT, yaml.safe_load(stream) or {}))
