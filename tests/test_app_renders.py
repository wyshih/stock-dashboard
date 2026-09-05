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
