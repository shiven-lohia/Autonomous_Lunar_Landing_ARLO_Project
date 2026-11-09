import numpy as np
import torch
from stable_baselines3 import PPO

from garl_lander.policy import GAPolicyNetwork
from garl_lander.ppo_agent import make_ppo, train_ppo
from garl_lander.weight_transfer import actor_logits, transfer_actor


def test_transfer_update_and_reload(tiny, tmp_path):
    ga = GAPolicyNetwork()
    model = make_ppo(tiny, 11)
    try:
        report = transfer_actor(ga, model, tiny)
        assert report["critic_unchanged"] and report["actions_match"]
        observations = np.random.default_rng(42).normal(size=(5, 8)).astype(np.float32)
        before = actor_logits(model, observations).detach().clone()
        train_ppo(model, tiny, 128, tmp_path)
        after = actor_logits(model, observations).detach()
        assert not torch.equal(before, after)
        reloaded = PPO.load(tmp_path / "final.zip")
        torch.testing.assert_close(after, actor_logits(reloaded, observations))
    finally:
        model.get_env().close()
