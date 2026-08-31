from trade_signal import build_signal, parse_return


def test_parse_return_percent_string():
    assert abs(parse_return("1.20%") - 0.012) < 1e-9
    assert abs(parse_return("-0.50%") - (-0.005)) < 1e-9


def test_buy_when_validated_and_above_band():
    pred = {
        "mean_return": "1.50%",
        "desk": {
            "lstm_scored": True,
            "beats_naive": True,
            "lstm_mae": 0.01,
        },
    }
    sig = build_signal(pred, "AAPL")
    assert sig["lean"] == "BUY"
    assert sig["action"] == "BUY"
    assert sig["execute"] is True
    assert sig["stamp"] == "VALIDATED"


def test_no_trade_when_lstm_loses():
    pred = {
        "mean_return": "2.00%",
        "desk": {
            "lstm_scored": True,
            "beats_naive": False,
            "lstm_mae": 0.01,
        },
    }
    sig = build_signal(pred, "RELIANCE.NS")
    assert sig["lean"] == "BUY"
    assert sig["action"] == "NO TRADE"
    assert sig["execute"] is False
    assert sig["stamp"] == "DO NOT SIZE"


def test_hold_inside_deadband():
    pred = {
        "mean_return": "0.05%",
        "desk": {
            "lstm_scored": True,
            "beats_naive": True,
            "lstm_mae": 0.013,
        },
    }
    sig = build_signal(pred, "AAPL")
    assert sig["lean"] == "HOLD"
    assert sig["action"] == "HOLD"
    assert sig["execute"] is False
