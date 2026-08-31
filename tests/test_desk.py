from eval_baselines import merge_lstm
from research_policy import desk_verdict


def test_merge_lstm_sets_flag():
    payload = merge_lstm({"tickers": {}}, {"AAPL": {"beats_naive_on_mae": False, "lstm": {"mae": 0.02}}})
    assert payload["tickers"]["AAPL"]["lstm_walk_forward"]["beats_naive_on_mae"] is False


def test_desk_verdict_missing_file_safe(tmp_path, monkeypatch):
    import research_policy as rp
    monkeypatch.setattr(rp, "METRICS_PATH", tmp_path / "none.json")
    v = rp.desk_verdict("AAPL")
    assert v["lstm_scored"] is False
    assert v["beats_naive"] is None


def test_desk_verdict_ns_alias(tmp_path, monkeypatch):
    import json
    import research_policy as rp

    path = tmp_path / "model_metrics.json"
    path.write_text(
        json.dumps(
            {
                "tickers": {
                    "RELIANCE.NS": {
                        "lstm_walk_forward": {
                            "beats_naive_on_mae": False,
                            "lstm": {"mae": 0.01, "directional_accuracy": 0.5},
                            "naive_zero_return_same_window": {"mae": 0.009},
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(rp, "METRICS_PATH", path)
    v = rp.desk_verdict("RELIANCE")
    assert v["lstm_scored"] is True
    assert v["beats_naive"] is False
    assert v["hit_rate"] == 0.5
