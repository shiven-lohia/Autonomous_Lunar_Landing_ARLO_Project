from types import SimpleNamespace

import numpy as np

from garl_lander.environment import classify_outcome, make_env


def test_api_and_truncation(tiny):
    env = make_env(tiny)
    try:
        obs, info = env.reset(seed=17)
        assert obs.shape == (8,) and isinstance(info, dict)
        assert env.action_space.n == 4
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert terminated or truncated
    finally:
        env.close()


def test_terminal_classification():
    body = SimpleNamespace(awake=False)
    base = SimpleNamespace(game_over=False, lander=body)
    env = SimpleNamespace(unwrapped=base)
    assert classify_outcome(env, np.zeros(8), True, False) == "landed_resting"
    base.game_over = True
    assert classify_outcome(env, np.zeros(8), True, False) == "crash"
    base.game_over = False
    assert classify_outcome(env, [1.1] + [0] * 7, True, False) == "out_of_bounds"
    assert classify_outcome(env, np.zeros(8), False, True) == "time_limit"
