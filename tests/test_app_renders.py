"""公開站能不能真的畫出來（2026-09-05）。

語法檢查抓不到「manifest 多了欄位但讀取端沒處理」這類錯誤。
2026-09-05 資料包新增第 3 個模型 swing，且各模型出場規則不同（manifest 的
`exit` 欄位），這支確保讀取端接得住。
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
APP = REPO / "streamlit_app.py"
needs_data = pytest.mark.skipif(
    not (REPO / "public_data" / "manifest.json").exists(),
    reason="需要 public_data（跑過 engine 的 make export-public）")


@needs_data
def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [str(e) for e in at.exception]


@needs_data
def test_swing_is_selectable_and_shows_its_own_exit_rule():
    """swing 的出場跟 m1 不同，選單要有它，而且要顯示它自己的規則。"""
    import json
    from streamlit.testing.v1 import AppTest

    manifest = json.loads((REPO / "public_data" / "manifest.json").read_text())
    keys = [m["key"] for m in manifest["models"]]
    assert "swing" in keys, "資料包裡沒有 swing"
    swing = next(m for m in manifest["models"] if m["key"] == "swing")
    assert swing["exit"]["type"] == "score"
    assert swing["exit"]["sell_threshold"] == 0.20

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [str(e) for e in at.exception]
    # 切到 swing
    sel = [s for s in at.selectbox if s.label == "模型"]
    assert sel, "找不到模型選單"
    at = sel[0].select("swing").run()
    assert not at.exception, [str(e) for e in at.exception]
    blob = " ".join(c.value for c in at.caption)
    assert "分數跌回 0.20 以下" in blob, "沒有顯示 swing 自己的出場規則"


needs_trade_rules = pytest.mark.skipif(
    not (REPO / "public_data" / "trade_rule_hits.parquet").exists(),
    reason="需要 public_data/trade_rule_hits.parquet（engine 的 make export-public）")


@needs_trade_rules
def test_trade_rules_page_shows_the_three_daily_sections():
    """每日買賣點頁：選定一天要看得到「買進 / 賣出 / 收盤後選出」三段。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.session_state["page"] = "每日買賣點"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    heads = " ".join(s.value for s in at.subheader)
    assert "開盤買進" in heads
    assert "開盤賣出" in heads
    assert "收盤後選出" in heads
    assert any(d.label == "看哪一天" for d in at.date_input)


@needs_trade_rules
def test_trade_rules_all_view_lists_every_record():
    """「全部」檢視要看得到整份買賣紀錄，不是只有單日。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.session_state["page"] = "每日買賣點"
    at.run()
    at.radio(key="trade_rule_view").set_value("全部").run()
    assert not at.exception, [str(e) for e in at.exception]

    text = " ".join(c.value for c in at.caption)
    assert "所有買賣紀錄" in text
    assert "共" in text and "筆" in text


needs_swing_trades = pytest.mark.skipif(
    not (REPO / "public_data" / "swing_trades.parquet").exists(),
    reason="需要 public_data/swing_trades.parquet（engine 的 make export-public）")


@needs_swing_trades
def test_trade_rules_defaults_to_the_swing_model():
    """這頁的預設來源是波段模型 —— 使用者要看的是模型給的買賣點。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.session_state["page"] = "每日買賣點"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="trade_rule_select").value == "swing"

    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "模型分數" in text
    assert "不設停損" not in text        # 那是規則來源的說明，模型來源不該出現


@needs_swing_trades
def test_swing_thresholds_default_to_manifest_values_and_are_adjustable():
    """0.97 / 0.20 當初始值（讀 manifest，不硬編），而且可以調。"""
    import json
    from streamlit.testing.v1 import AppTest

    manifest = json.loads((REPO / "public_data" / "manifest.json").read_text())
    swing = next(m for m in manifest["models"] if m["key"] == "swing")

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.session_state["page"] = "每日買賣點"
    at.run()

    buy = at.select_slider(key="swing_buy_th")
    sell = at.select_slider(key="swing_sell_th")
    assert buy.value == swing["threshold"]
    assert sell.value == swing["exit"]["sell_threshold"]

    buy.set_value(0.95).run()
    assert not at.exception, [str(e) for e in at.exception]
