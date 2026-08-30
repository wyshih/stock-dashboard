"""`compute_forward_returns` 的契約測試。

稽核（2026-08-27）指出這是最容易安靜出錯的地方：`groupby.shift` 前若沒排序，
會把別檔股票或別天的價格算進來，而且不會報任何錯。
"""
from __future__ import annotations

import pandas as pd
import pytest

from streamlit_app import compute_forward_returns


def _price(rows):
    return pd.DataFrame(rows, columns=["date", "stock_id", "close"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_return_matches_hand_calculation():
    price = _price([(f"2026-01-{d:02d}", "1101", 100 + d) for d in range(1, 6)])
    out = compute_forward_returns(price, bars=2)
    first = out[out["date"] == "2026-01-01"]["fwd20"].iloc[0]
    assert first == pytest.approx((103 / 101 - 1) * 100)


def test_shuffled_input_gives_the_same_answer():
    """輸入順序不該影響結果 —— 少了排序時這條會紅。"""
    rows = [(f"2026-01-{d:02d}", "1101", 100 + d) for d in range(1, 6)]
    ordered = compute_forward_returns(_price(rows), bars=2)
    shuffled = compute_forward_returns(
        _price(rows).sample(frac=1, random_state=0), bars=2)
    merged = ordered.merge(shuffled, on=["date", "stock_id"], suffixes=("_a", "_b"))
    assert len(merged) == len(ordered)
    pd.testing.assert_series_equal(merged["fwd20_a"], merged["fwd20_b"],
                                   check_names=False)


def test_two_stocks_interleaved_do_not_bleed_into_each_other():
    """兩檔交錯時，1101 不可以拿到 2330 的價格。"""
    rows = []
    for d in range(1, 5):
        rows.append((f"2026-01-{d:02d}", "1101", 100 + d))
        rows.append((f"2026-01-{d:02d}", "2330", 1000 * d))
    out = compute_forward_returns(_price(rows), bars=1)
    got = out[(out["stock_id"] == "1101") & (out["date"] == "2026-01-01")]
    assert got["fwd20"].iloc[0] == pytest.approx((102 / 101 - 1) * 100)


def test_tail_rows_are_nan_not_zero():
    """走不完的日子要是 NaN（顯示成未定案），不可以變成 0%（看起來像沒賺沒賠）。"""
    price = _price([(f"2026-01-{d:02d}", "1101", 100 + d) for d in range(1, 4)])
    out = compute_forward_returns(price, bars=2)
    assert out["fwd20"].isna().sum() == 2
    assert not (out["fwd20"] == 0).any()
