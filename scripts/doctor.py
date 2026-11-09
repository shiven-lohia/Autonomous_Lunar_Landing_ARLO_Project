"""Diagnose the runtime and run a complete random LunarLander episode."""

import platform
from importlib.metadata import version

import torch

from garl_lander.config import DEFAULT
from garl_lander.environment import make_env
from garl_lander.evaluation import evaluate, summarize


def main():
    print(f"Python {platform.python_version()} ({platform.architecture()[0]})")
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "Box2D", "pygame-ce"):
        print(f"{name}: {version(name)}")
    print(f"CUDA available: {torch.cuda.is_available()}; project default: CPU")
    env = make_env(DEFAULT)
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (8,) and env.action_space.n == 4 and isinstance(info, dict)
        assert len(env.step(0)) == 5
    finally:
        env.close()
    print(summarize(evaluate(None, DEFAULT, [11])))
    print("SUCCESS: LunarLander construction, reset, step, and complete episode verified.")


if __name__ == "__main__":
    main()
