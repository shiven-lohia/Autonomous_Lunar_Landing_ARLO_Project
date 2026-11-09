"""A 4,996-parameter actor with safe NumPy checkpoint serialization."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GAPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, 4))

    def forward(self, observations):
        return self.net(observations)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_parameters(self) -> np.ndarray:
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy().copy()

    def set_flat_parameters(self, chromosome: np.ndarray) -> None:
        chromosome = np.asarray(chromosome, dtype=np.float32)
        if chromosome.shape != (self.parameter_count(),) or not np.isfinite(chromosome).all():
            raise ValueError("Chromosome must contain exactly 4996 finite parameters")
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters():
                count = parameter.numel()
                parameter.copy_(torch.from_numpy(chromosome[offset:offset + count].copy()).reshape(parameter.shape))
                offset += count

    @torch.no_grad()
    def predict_action(self, observation) -> int:
        return int(self(torch.as_tensor(observation, dtype=torch.float32)).argmax().item())


def initialize_chromosome(rng: np.random.Generator) -> np.ndarray:
    """Same uniform bounds as nn.Linear.reset_parameters; independent NumPy RNG."""
    pieces = []
    for fan_in, fan_out in ((8, 64), (64, 64), (64, 4)):
        bound = 1 / np.sqrt(fan_in)
        pieces.extend([rng.uniform(-bound, bound, fan_out * fan_in), rng.uniform(-bound, bound, fan_out)])
    return np.concatenate(pieces).astype(np.float32)


def save_policy(policy: GAPolicyNetwork, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, parameters=policy.get_flat_parameters(), architecture=np.array([8, 64, 64, 4]))


def load_policy(path: str | Path) -> GAPolicyNetwork:
    policy = GAPolicyNetwork()
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["architecture"], [8, 64, 64, 4]):
            raise ValueError("Checkpoint architecture mismatch")
        policy.set_flat_parameters(data["parameters"])
    return policy
