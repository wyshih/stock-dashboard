"""前端接線測試：148 條說法都在、統計讀得到、門檻不硬編。"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestPatterns:
    def test_has_148_patterns(self):
        from frontend.patterns import PATTERNS, PATTERNS_BY_KEY
        assert len(PATTERNS) == 148
        assert len(PATTERNS_BY_KEY) == 148

    def test_shared_modules_import_without_talib(self):
        """148 條說法的判定式在 public 端不會被呼叫，所以不需要 TA-Lib。"""
        import frontend.patterns          # noqa: F401
        import frontend.patterns_talib    # noqa: F401
        import frontend.patterns_chip     # noqa: F401
        import frontend.verdict           # noqa: F401
        import frontend.technical_chart   # noqa: F401
        import frontend.indicators        # noqa: F401
        import frontend.levels            # noqa: F401


class TestConditionalStats:
    def test_missing_data_degrades_gracefully(self):
        """public_data/ 是空的時候不可以爆 —— 頁面要自己說「請先 export」。"""
        from frontend import conditional_stats as cs
        assert isinstance(cs.baseline(), dict)
        assert isinstance(cs.all_pattern_entries(), dict)

    def test_thresholds_are_the_same_definition_as_engine(self):
        from frontend import conditional_stats as cs
        assert cs.HOLD_DAYS == (5, 20)
        assert cs.MIN_SAMPLES == 30
        assert cs.EDGE_TOLERANCE == 0.01


class TestApp:
    def test_app_parses(self):
        source = (REPO / "streamlit_app.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_app_has_four_pages(self):
        import re
        source = (REPO / "streamlit_app.py").read_text(encoding="utf-8")
        block = re.search(r"PAGES = \{(.*?)\}", source, re.S)
        assert block and block.group(1).count(":") == 4

    def test_no_hardcoded_thresholds(self):
        """門檻一律讀 manifest.json（CLAUDE.md 規則 7）。"""
        source = (REPO / "streamlit_app.py").read_text(encoding="utf-8")
        assert "CHOSEN_THRESHOLDS" not in source
        assert 'manifest.get("models"' in source or "manifest.get('models'" in source

    def test_app_never_reads_outside_public_data(self):
        source = (REPO / "streamlit_app.py").read_text(encoding="utf-8")
        for forbidden in ("features.parquet", "features_v3.parquet",
                          "bundle_", "models/", "predict_proba"):
            assert forbidden not in source, f"public 前端不可以碰 {forbidden}"

    def test_every_page_states_it_is_backtest_data(self):
        source = (REPO / "streamlit_app.py").read_text(encoding="utf-8")
        assert source.count("TEST_BANNER") >= 5   # 定義 1 次 + 四頁各用 1 次
