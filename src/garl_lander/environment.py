"""Unmodified Gymnasium rewards and explicit episode outcomes."""

import gymnasium as gym


def make_env(cfg: dict, render_mode: str | None = None) -> gym.Env:
    args = dict(cfg["environment"])
    env_id = args.pop("id")
    env = gym.make(env_id, render_mode=render_mode, **args)
    if env.observation_space.shape != (8,) or not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError("Expected eight observations and discrete actions")
    if env.action_space.n != 4:
        env.close()
        raise ValueError("Expected four actions")
    return env


def classify_outcome(env: gym.Env, obs, terminated: bool, truncated: bool) -> str:
    """Use LunarLander internal terminal conditions; reward threshold is separate.

    A sleeping, non-crashed body is a resting landing, not necessarily on the pad.
    game_over is the body-ground contact flag in Gymnasium's ContactDetector.
    """
    lander = env.unwrapped
    if terminated:
        if lander.game_over:
            return "crash"
        if abs(float(obs[0])) >= 1:
            return "out_of_bounds"
        if not lander.lander.awake:
            return "landed_resting"
        return "other_terminal"
    return "time_limit" if truncated else "ongoing"
