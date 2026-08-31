import numpy as np

from eval_baselines import next_day_returns, score_forecast, walk_forward_baselines


def test_next_day_returns():
    closes = np.array([100.0, 110.0, 99.0])
    rets = next_day_returns(closes)
    assert len(rets) == 2
    assert abs(rets[0] - 0.1) < 1e-9


def test_naive_beats_nothing_on_flat_then_move():
    # Strictly increasing: persistence should have lower MAE than naive zero.
    closes = np.linspace(100, 120, 80)
    report = walk_forward_baselines(closes)
    persist = report["persistence_last_return"]["mae"]
    naive = report["naive_zero_return"]["mae"]
    assert persist is not None and naive is not None
    assert persist < naive


def test_score_forecast_empty():
    assert score_forecast(np.array([]), np.array([]))["n"] == 0
