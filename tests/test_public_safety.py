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
    "labels.parquet", "labels_nobear.parquet", "price.parquet",
    "BACKTEST_LOG.md", ".env",
}
# 用 glob 而非精確檔名 —— 只列精確檔名的話，`features_v4.parquet` 這種新檔名
# 會直接漏掉（2026-08-27 稽核 MEDIUM-2）。
FORBIDDEN_PATTERNS = ("features*.parquet", "price_official*.parquet",
                      "score_*.parquet", "*.env")
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


class TestNoForbiddenPatterns:
    def test_no_file_matches_a_forbidden_pattern(self):
        import fnmatch
        bad = [str(p.relative_to(REPO)) for p in _walk()
               if any(fnmatch.fnmatch(p.name, pat) for pat in FORBIDDEN_PATTERNS)]
        assert not bad, f"public repo 不可以有這些檔：{bad}"


class TestNoStaleOutputs:
    """契約改了但 out_dir 沒清，舊產物會被算進體積、也會誤導讀者。"""

    STALE = ("scores_test.parquet",)

    def test_no_superseded_bundle_files(self):
        data = REPO / "public_data"
        if not data.exists():
            pytest.skip("public_data 還沒產生")
        bad = [n for n in self.STALE if (data / n).exists()]
        assert not bad, f"這些是被取代的舊產物，請刪除：{bad}"


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

    def test_every_dated_file_starts_no_earlier_than_test_period(self):
        """**每一個**有 date 欄的 parquet 都要守同一條期間下限。

        原本只掃 scores_test_*，pattern_hits 完全沒被釘住 —— 那支的 TEST_START
        過濾只有一行，拿掉的話會把 2015 年起的全市場命中矩陣推上 public repo，
        而測試一條都不會紅（2026-08-27 稽核 MEDIUM-2）。
        """
        import pandas as pd
        paths = sorted((REPO / "public_data").glob("*.parquet"))
        if not paths:
            pytest.skip("public_data 還沒產生")
        checked = []
        for path in paths:
            if "date" not in pd.read_parquet(path).columns:
                continue
            dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
            assert dates.min() >= pd.Timestamp("2025-02-01"), (
                f"{path.name} 含有 2025-02-01 之前的資料（最早 {dates.min().date()}）")
            checked.append(path.name)
        assert any(n.startswith("pattern_hits") for n in checked), (
            f"pattern_hits 沒被檢查到，實際掃到：{checked}")
        # 數量對齊 manifest，不寫死 —— 模型增減時這條不該變成需要手改的地方。
        import json
        n_models = len(json.loads(
            (REPO / "public_data" / "manifest.json").read_text(encoding="utf-8"))["models"])
        assert sum(n.startswith("scores_test_") for n in checked) == n_models

    def test_scores_are_split_one_file_per_model(self):
        """一個模型一個檔；不可以再退回合併成一個大檔。"""
        data = REPO / "public_data"
        if not (data / "manifest.json").exists():
            pytest.skip("public_data 還沒產生")
        import json
        keys = [m["key"] for m in
                json.loads((data / "manifest.json").read_text(encoding="utf-8"))["models"]]
        missing = [k for k in keys if not (data / f"scores_test_{k}.parquet").exists()]
        assert not missing, f"缺少這些模型的分數檔：{missing}"
        assert not (data / "scores_test.parquet").exists(), (
            "合併版 scores_test.parquet 不該再產生")

    def test_manifest_declares_disclaimer(self):
        path = REPO / "public_data" / "manifest.json"
        if not path.exists():
            pytest.skip("public_data 還沒產生")
        import json
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert "不構成" in manifest.get("disclaimer", "")
