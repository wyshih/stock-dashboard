"""這個 repo 是**公開**的，測試的重點就是「不該出現的東西沒有出現」。

推出去就收不回來，所以這幾條要用測試釘住，不能只寫在 CLAUDE.md 裡。
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# 絕對不可以出現在這個 repo 的檔名 / 副檔名
FORBIDDEN_SUFFIXES = (".pkl", ".pickle", ".pt", ".joblib")
FORBIDDEN_NAMES = {
    "features.parquet", "features_v3.parquet", "labels.parquet",
    "labels_nobear.parquet", "price.parquet", "price_official.parquet",
    "BACKTEST_LOG.md", ".env",
}
# 前端不可以相依的重量套件（Streamlit Community Cloud 裝不起來 / 不需要）
FORBIDDEN_PACKAGES = ("ta-lib", "talib", "scikit-learn", "sklearn", "lightgbm",
                      "torch", "yfinance", "finmind")

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def _walk() -> list[Path]:
    out = []
    for path in REPO.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            out.append(path)
    return out


class TestNoSecretsOrModels:
    def test_no_model_files(self):
        bad = [p for p in _walk() if p.suffix.lower() in FORBIDDEN_SUFFIXES]
        assert not bad, f"public repo 不可以有模型檔：{[str(p) for p in bad]}"

    def test_no_forbidden_filenames(self):
        bad = [p for p in _walk() if p.name in FORBIDDEN_NAMES]
        assert not bad, f"public repo 不可以有這些檔：{[str(p) for p in bad]}"

    def test_no_env_file(self):
        assert not (REPO / ".env").exists()


class TestRequirements:
    def test_no_heavy_packages(self):
        text = (REPO / "requirements.txt").read_text(encoding="utf-8")
        lines = [ln.strip().lower() for ln in text.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        for package in FORBIDDEN_PACKAGES:
            assert not any(ln.startswith(package) for ln in lines), (
                f"{package} 不可以出現在 public 前端的 requirements.txt")

    def test_streamlit_is_declared(self):
        text = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
        assert "streamlit" in text


class TestPublicDataScope:
    """public_data/ 裡如果已經有資料包，期間不可以早於測試期起點。"""

    def test_price_starts_no_earlier_than_test_period(self):
        path = REPO / "public_data" / "price_test.parquet"
        if not path.exists():
            pytest.skip("public_data 還沒產生")
        import pandas as pd
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        assert dates.min() >= pd.Timestamp("2025-02-01"), (
            f"public_data 含有 2025-02 之前的資料（最早 {dates.min().date()}）")

    def test_manifest_declares_disclaimer(self):
        path = REPO / "public_data" / "manifest.json"
        if not path.exists():
            pytest.skip("public_data 還沒產生")
        import json
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert "不構成" in manifest.get("disclaimer", "")
