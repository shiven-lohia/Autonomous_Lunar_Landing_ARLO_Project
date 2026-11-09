import numpy as np

from garl_lander.genetic_algorithm import GeneticAlgorithm, crossover, mutate, tournament


def test_population_and_operators(tiny):
    ga = GeneticAlgorithm(tiny, 11)
    again = GeneticAlgorithm(tiny, 11)
    np.testing.assert_array_equal(ga.population, again.population)
    assert ga.population.shape == (4, 4996)
    assert not np.array_equal(ga.population[0], ga.population[1])
    rng = np.random.default_rng(10)
    a, b = np.zeros(50), np.ones(50)
    child = crossover(a, b, rng)
    assert set(np.unique(child)) == {0, 1}
    assert not np.shares_memory(child, a)
    assert 0 <= tournament(np.arange(4), rng, 2) < 4
    assert tournament(np.arange(4), rng, 4) == 3
    np.testing.assert_array_equal(mutate(a, rng, 0, 1), a)
    assert np.any(mutate(a, rng, 1, 1) != a)
    offspring = ga.breed(np.arange(4))
    np.testing.assert_array_equal(offspring[0], ga.population[3])
    assert offspring.shape == ga.population.shape
    assert not np.shares_memory(offspring, ga.population)


def test_checkpoint_exact_resume(tiny, tmp_path):
    full = GeneticAlgorithm(tiny, 11)
    full.train(500, tmp_path / "full")
    split = GeneticAlgorithm(tiny, 11)
    split.train(500, tmp_path / "split", stop_after=1)
    resumed = GeneticAlgorithm.load(tmp_path / "split" / "checkpoint.npz")
    resumed.train(500, tmp_path / "split")
    np.testing.assert_array_equal(full.population, resumed.population)
    np.testing.assert_array_equal(full.champion, resumed.champion)
    assert full.optimization_steps == resumed.optimization_steps == 160
    assert full.selection_steps == resumed.selection_steps == 120
    assert all(np.isfinite(h["best"]) for h in full.history)
