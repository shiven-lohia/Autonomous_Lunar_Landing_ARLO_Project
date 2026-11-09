import numpy as np
import pytest
import torch

from garl_lander.policy import GAPolicyNetwork, initialize_chromosome, load_policy, save_policy


def test_shapes_and_roundtrip(tmp_path):
    policy = GAPolicyNetwork()
    assert policy.parameter_count() == 4996
    assert policy(torch.zeros(8)).shape == (4,)
    assert policy(torch.zeros(5, 8)).shape == (5, 4)
    flat = initialize_chromosome(np.random.default_rng(11))
    policy.set_flat_parameters(flat)
    np.testing.assert_array_equal(flat, policy.get_flat_parameters())
    save_policy(policy, tmp_path / "actor.npz")
    loaded = load_policy(tmp_path / "actor.npz")
    torch.testing.assert_close(policy(torch.zeros(8)), loaded(torch.zeros(8)))
    assert torch.isfinite(loaded(torch.zeros(8))).all()


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(4996, np.nan), np.full(4996, np.inf)])
def test_invalid_chromosome(bad):
    with pytest.raises(ValueError):
        GAPolicyNetwork().set_flat_parameters(bad)
