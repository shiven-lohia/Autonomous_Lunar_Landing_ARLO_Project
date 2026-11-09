import pandas as pd

from garl_lander.evaluation import evaluate, summarize
from garl_lander.policy import GAPolicyNetwork


def test_evaluation_reproducible(tiny):
    actor = GAPolicyNetwork()
    one = evaluate(actor, tiny, [40000, 40001])
    two = evaluate(actor, tiny, [40000, 40001])
    pd.testing.assert_frame_equal(one, two)
    summary = summarize(one)
    assert summary["episodes"] == 2 and summary["test_steps"] == 40
    assert 0 <= summary["solved_rate"] <= 1


def test_random_reproducible(tiny):
    pd.testing.assert_frame_equal(evaluate(None, tiny, [40000]), evaluate(None, tiny, [40000]))
