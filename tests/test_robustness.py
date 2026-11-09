import pandas as pd

from garl_lander.experiment import robustness, run_suite
from garl_lander.io_utils import read_json


def test_robustness_preserves_models_and_counts_extra_steps(tiny, tmp_path):
    suite = run_suite(tiny, tmp_path, agents=("ga",))
    checkpoint = suite / "ga_seed11" / "ga" / "selected.npz"
    before = checkpoint.read_bytes()
    destination = robustness(suite, {"default": {}, "wind": {"enable_wind": True}})
    assert checkpoint.read_bytes() == before
    rows = pd.read_csv(destination / "robustness_results.csv")
    source = read_json(suite / "ga_seed11" / "summary.json")
    assert abs(rows.loc[rows.condition == "default", "mean_return"].iloc[0] - source["mean_return"]) < 1e-8
    assert rows.test_steps.sum() == 80
    assert read_json(destination / "manifest.json")["extra_evaluation_steps"] == 80
