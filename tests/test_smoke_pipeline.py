from garl_lander.experiment import reevaluate, run_suite, suite_runs
from garl_lander.io_utils import read_json


def test_full_five_agent_pipeline(tiny, tmp_path):
    suite = run_suite(tiny, tmp_path)
    results = {s["agent"]: s for _, s in suite_runs(suite)}
    assert set(results) == {"random", "ga", "ppo", "hybrid", "ppo_short"}
    hybrid = results["hybrid"]
    assert hybrid["ga_steps"] + hybrid["ppo_steps"] <= 1024
    assert hybrid["ppo_steps"] == results["ppo_short"]["ppo_steps"]
    for result in results.values():
        assert result["total_steps"] == result["optimization_steps"] + result["selection_steps"] + result["test_steps"]
    transfer = read_json(suite / "hybrid_seed11" / "transfer.json")
    assert transfer["actor_changed_after_ppo"]
    assert transfer["ga_selected_validation"] == transfer["post_transfer_validation"]
    reevaluated = reevaluate(suite)
    assert len(list(reevaluated.glob("*.csv"))) == 5
    assert run_suite(tiny, tmp_path, resume=suite) == suite
