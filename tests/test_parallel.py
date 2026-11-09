import pandas as pd

from garl_lander.experiment import run_suite


def test_parallel_seed_ownership_and_reproducibility(tiny, tmp_path):
    tiny["experiment"]["seeds"] = [11, 23]
    serial = run_suite(tiny, tmp_path / "serial", agents=("random", "ga"))
    parallel = run_suite(tiny, tmp_path / "parallel", agents=("random", "ga"), workers=2)
    for agent in ("random", "ga"):
        for seed in (11, 23):
            relative = f"{agent}_seed{seed}/evaluation.csv"
            pd.testing.assert_frame_equal(pd.read_csv(serial / relative), pd.read_csv(parallel / relative))
