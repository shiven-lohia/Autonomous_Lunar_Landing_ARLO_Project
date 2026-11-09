import numpy as np
import pandas as pd

from garl_lander.statistics import bootstrap_mean


def test_seed_uncertainty_not_episode_pseudoreplication():
    assert np.isnan(bootstrap_mean([5.0])[0])
    low, high = bootstrap_mean([1.0, 2.0, 3.0])
    assert 1 <= low <= 2 <= high <= 3
    assert bootstrap_mean([1.0, 2.0, 3.0]) == (low, high)


def test_artifacts_from_saved_suite(tiny, tmp_path):
    from garl_lander.experiment import run_suite
    from garl_lander.plotting import generate_plots
    from garl_lander.statistics import analyze
    suite = run_suite(tiny, tmp_path, agents=("ga", "hybrid", "ppo_short"))
    per_seed, aggregate = analyze(suite)
    assert len(per_seed) == 3 and len(aggregate) == 3
    assert aggregate.ci_low.isna().all()
    plots = generate_plots(suite)
    assert (plots / "architecture.png").stat().st_size > 1000
    assert (plots / "learning_curves.svg").stat().st_size > 1000
    curves = pd.read_csv(suite / "analysis" / "learning_curves.csv")
    assert curves.optimization_steps.max() <= tiny["experiment"]["total_optimization_steps"]
