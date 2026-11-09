from copy import deepcopy

import pytest

from garl_lander.config import DEFAULT, validate
from garl_lander.seeding import seed_everything


@pytest.fixture
def tiny():
    cfg = deepcopy(DEFAULT)
    cfg["name"] = "test"
    cfg["environment"]["max_episode_steps"] = 20
    cfg["experiment"].update(seeds=[11], total_optimization_steps=1024,
                             final_eval_episodes=2, validation_episodes=2, checkpoint_interval=128)
    cfg["ga"].update(population_size=4, elite_count=1, tournament_size=2, fitness_episodes=1,
                     max_generations=2)
    cfg["ppo"].update(n_steps=32, n_envs=2, batch_size=32, n_epochs=2)
    seed_everything(11)
    return validate(cfg)
