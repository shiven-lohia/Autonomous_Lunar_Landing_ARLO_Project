"""Exact actor mapping with raw-logit, action and untouched-critic checks."""

import numpy as np
import torch

from .environment import make_env
from .policy import GAPolicyNetwork


def actor_logits(model, observations: np.ndarray) -> torch.Tensor:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    features = model.policy.extract_features(obs)
    latent = model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent)


def transfer_actor(source: GAPolicyNetwork, target, cfg: dict) -> dict:
    targets = [target.policy.mlp_extractor.policy_net[0], target.policy.mlp_extractor.policy_net[2],
               target.policy.action_net]
    sources = [source.net[0], source.net[2], source.net[4]]
    critic_before = {k: v.detach().clone() for k, v in target.policy.state_dict().items()
                     if "value_net" in k}
    with torch.no_grad():
        for src, dst in zip(sources, targets, strict=True):
            if src.weight.shape != dst.weight.shape or src.bias.shape != dst.bias.shape:
                raise ValueError("GA and PPO actor shapes do not match")
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
            torch.testing.assert_close(src.weight, dst.weight.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(src.bias, dst.bias.cpu(), rtol=0, atol=0)
    rng = np.random.default_rng(991)
    observations = list(rng.normal(size=(32, 8)).astype(np.float32))
    env = make_env(cfg)
    try:
        # Resets provide genuine states without consuming unaccounted transitions.
        for seed in range(30000, 30008):
            observations.append(env.reset(seed=seed)[0])
    finally:
        env.close()
    observations = np.asarray(observations)
    with torch.no_grad():
        expected = source(torch.from_numpy(observations))
        actual = actor_logits(target, observations).cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    if not torch.equal(expected.argmax(-1), actual.argmax(-1)):
        raise AssertionError("Actions changed during actor transfer")
    for key, before in critic_before.items():
        torch.testing.assert_close(before, target.policy.state_dict()[key], rtol=0, atol=0)
    if target.policy.optimizer.state:
        raise ValueError("Transfer requires a freshly initialized PPO optimizer")
    return {"max_logit_error": float((expected - actual).abs().max()), "observations_checked": len(observations),
            "actor_tensors_copied": 6, "critic_unchanged": True, "actions_match": True,
            "verification_steps": 0}
