"""SB3 PPO with exact rollout budgets and post-update validation."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import make_env
from .evaluation import evaluate
from .io_utils import append_csv, read_json, write_json
from .seeding import episode_seeds, training_seed


class CountSteps(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transition_count = 0

    def step(self, action):
        self.transition_count += 1
        return self.env.step(action)


class LoggedPPO(PPO):
    """Call instrumentation after the optimizer update, never before it."""

    def train(self) -> None:
        super().train()
        hook = getattr(self, "_post_update", None)
        if hook is not None:
            hook(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_post_update"]


def rollout_budget(budget: int, cfg: dict) -> int:
    size = cfg["ppo"]["n_steps"] * cfg["ppo"]["n_envs"]
    return budget // size * size


def make_ppo(cfg: dict, seed: int, directory: Path | None = None) -> LoggedPPO:
    p = dict(cfg["ppo"])
    n_envs = p.pop("n_envs")
    factories = []
    for index in range(n_envs):
        filename = str(directory / f"env_{index}") if directory else None
        factories.append(lambda filename=filename: CountSteps(Monitor(make_env(cfg), filename=filename)))
    env = DummyVecEnv(factories)
    model = LoggedPPO("MlpPolicy", env, seed=seed, verbose=0,
                      policy_kwargs={"activation_fn": torch.nn.Tanh,
                                     "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                                     "ortho_init": False}, **p)
    env.seed(training_seed(seed, 1))
    return model


def train_ppo(model: LoggedPPO, cfg: dict, budget: int, directory: Path,
              ga_steps: int = 0, ga_selection_steps: int = 0, ga_elapsed: float = 0.0,
              resume: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    target = rollout_budget(budget, cfg)
    if target < cfg["ppo"]["n_steps"] * cfg["ppo"]["n_envs"]:
        raise ValueError("PPO allocation cannot fund one complete rollout")
    started = time.perf_counter()
    metadata_path = directory / "training_state.json"
    state = read_json(metadata_path) if resume and metadata_path.exists() else {
        "selection_steps": 0, "best_validation": None, "last_checkpoint": 0, "elapsed_seconds": 0.0}
    prior_elapsed = state["elapsed_seconds"]
    initial_steps = model.num_timesteps
    model.set_logger(configure(str(directory / "tensorboard"), ["tensorboard", "csv"]))

    def validate_checkpoint(current, initial=False):
        rows = evaluate(current, cfg, episode_seeds(cfg, "validation"))
        score = float(rows["return"].mean())
        state["selection_steps"] += int(rows.length.sum())
        state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        append_csv(directory / "validation.csv", [{
            "ppo_steps": current.num_timesteps, "optimization_steps": ga_steps + current.num_timesteps,
            "selection_steps": ga_selection_steps + state["selection_steps"],
            "total_steps": ga_steps + current.num_timesteps + ga_selection_steps + state["selection_steps"],
            "mean_return": score, "elapsed_seconds": ga_elapsed + state["elapsed_seconds"]}])
        if state["best_validation"] is None or score > state["best_validation"]:
            state["best_validation"] = score
            state["selected_ppo_steps"] = current.num_timesteps
            current.save(directory / "selected.zip")
        if initial:
            state["initial_return"] = score
        state["last_checkpoint"] = current.num_timesteps
        current.save(directory / "latest.zip")
        write_json(metadata_path, state)
        print(f"  PPO steps={current.num_timesteps}/{target} validation={score:.1f}", flush=True)

    def after_update(current):
        values = current.logger.name_to_value
        row = {"ppo_steps": current.num_timesteps, "optimization_steps": ga_steps + current.num_timesteps,
               "elapsed_seconds": ga_elapsed + prior_elapsed + time.perf_counter() - started}
        for key in ("policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction",
                    "explained_variance", "learning_rate", "loss"):
            value = float(values.get("train/" + key, np.nan))
            row[key] = value if np.isfinite(value) else None
        append_csv(directory / "updates.csv", [row])
        if (current.num_timesteps - state["last_checkpoint"] >= cfg["experiment"]["checkpoint_interval"]
                or current.num_timesteps == target):
            validate_checkpoint(current)

    if not resume:
        validate_checkpoint(model, initial=True)
    model._post_update = after_update
    remaining = target - model.num_timesteps
    if remaining < 0:
        raise ValueError("Checkpoint exceeds requested budget")
    if remaining:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, tb_log_name="PPO")
    actual = sum(model.get_env().get_attr("transition_count"))
    if actual != model.num_timesteps - initial_steps or model.num_timesteps != target:
        raise AssertionError("PPO transition counter disagrees with rollout accounting")
    model.save(directory / "final.zip")
    state.update({"ppo_steps": model.num_timesteps, "unused_ppo_budget": budget - target,
                  "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                  "resumed_with_fresh_environment": resume})
    write_json(metadata_path, state)
    model._post_update = None
    return state
