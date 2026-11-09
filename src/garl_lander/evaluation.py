"""One episode loop for all agents; independent test observations and raw rows."""

import numpy as np
import pandas as pd

from .environment import classify_outcome, make_env
from .policy import GAPolicyNetwork


def action_for(agent, observation, rng: np.random.Generator, deterministic: bool = True) -> int:
    if agent is None:
        return int(rng.integers(4))
    if isinstance(agent, GAPolicyNetwork):
        return agent.predict_action(observation)
    action, _ = agent.predict(observation, deterministic=deterministic)
    return int(action)


def evaluate(agent, cfg: dict, seeds: list[int], deterministic: bool = True) -> pd.DataFrame:
    records = []
    env = make_env(cfg)
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed) + 700000)
            total, length, engine = 0.0, 0, 0
            while True:
                action = action_for(agent, obs, rng, deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                engine += int(action != 0)
                if terminated or truncated:
                    break
            if not np.isfinite(total):
                raise FloatingPointError("Non-finite episode return")
            records.append({"episode_seed": int(seed), "return": total, "length": length,
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "outcome": classify_outcome(env, obs, terminated, truncated),
                            "solved": total >= cfg["evaluation"]["solved_reward_threshold"],
                            "engine_actions": engine, "deterministic": deterministic})
    finally:
        env.close()
    return pd.DataFrame(records)


def summarize(episodes: pd.DataFrame) -> dict:
    r = episodes["return"]
    return {"episodes": len(r), "mean_return": float(r.mean()),
            "std_return": float(r.std(ddof=1)) if len(r) > 1 else None,
            "median_return": float(r.median()), "min_return": float(r.min()), "max_return": float(r.max()),
            "mean_length": float(episodes.length.mean()), "solved_rate": float(episodes.solved.mean()),
            "resting_landing_rate": float((episodes.outcome == "landed_resting").mean()),
            "crash_rate": float((episodes.outcome == "crash").mean()),
            "out_of_bounds_rate": float((episodes.outcome == "out_of_bounds").mean()),
            "test_steps": int(episodes.length.sum())}
