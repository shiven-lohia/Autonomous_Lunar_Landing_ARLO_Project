import pytest

from garl_lander.config import validate
from garl_lander.genetic_algorithm import GeneticAlgorithm
from garl_lander.ppo_agent import make_ppo, rollout_budget, train_ppo
from garl_lander.seeding import episode_seeds, training_seed


def test_budget_rounding_and_vec_counts(tiny, tmp_path):
    assert rollout_budget(130, tiny) == 128
    model = make_ppo(tiny, 11)
    try:
        state = train_ppo(model, tiny, 130, tmp_path)
        assert state["ppo_steps"] == 128 and state["unused_ppo_budget"] == 2
        assert sum(model.get_env().get_attr("transition_count")) == 128
        assert state["selection_steps"] == 80  # initial and final, 2 x 20 steps each
    finally:
        model.get_env().close()


def test_ga_reservation(tiny, tmp_path):
    ga = GeneticAlgorithm(tiny, 11)
    with pytest.raises(ValueError, match="reserve"):
        ga.train(79, tmp_path)


def test_seeds_separated(tiny):
    val, test = set(episode_seeds(tiny, "validation")), set(episode_seeds(tiny, "final"))
    assert not val & test
    train = {training_seed(11, stream, i) for stream in range(5) for i in range(10)}
    assert len(train) == 50 and not train & (val | test)


@pytest.mark.parametrize("section,key,value", [
    ("ga", "elite_count", 4), ("ga", "mutation_probability", 1.1),
    ("ga", "mutation_sigma", -1), ("hybrid", "ga_budget_fraction", 1),
    ("ppo", "batch_size", 13), ("experiment", "seeds", []),
    ("evaluation", "final_seed_start", 20000),
])
def test_invalid_configs(tiny, section, key, value):
    tiny[section][key] = value
    with pytest.raises(ValueError):
        validate(tiny)
