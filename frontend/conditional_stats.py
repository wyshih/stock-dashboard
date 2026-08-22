"""型態的歷史條件統計 —— public 版：讀預先算好的檔案，不碰特徵值。

跟 private engine 版的差別只有「資料從哪來」，判定與統計的**定義完全一樣**
（engine 版是用同一套程式離線算好再匯出的）：

- 進場價 = 訊號日**隔一天**的收盤（訊號要收盤後才知道，當天買不到）。
- 出場價 = 進場後第 5 / 20 個交易日的收盤，與 ground truth `label_up20`
  的 20 個交易日視窗一致。
- 對照組是**全市場所有日子**的同一組報酬。沒有基準的勝率沒有意義：
  多頭年份隨便買 20 天勝率都有五成以上。

資料來源：
    public_data/pattern_stats.json    單條說法的統計 —— **全市場全歷史**
    public_data/pattern_hits.parquet  148 條說法的命中矩陣（int8，測試期）
    public_data/price_test.parquet    測試期全市場 OHLCV

⚠️ 兩種統計的口徑不同，畫面上一定要分開講：
  - 單條說法（`pattern_stats`）是**全歷史**，樣本多、結論比較穩。
  - 聯合條件（`joint_stats`）只能用**測試期（2025-02~2026-07）**算 ——
    任意組合沒辦法預先窮舉（148 條的組合數是天文數字），只好現場從命中矩陣
    交集。樣本少很多，`MIN_SAMPLES` 一樣擋，但要標明期間。

已知的限制（畫面上一定要一起講）：
- 資料不含已下市股票，統計本身帶生存偏差，數字偏樂觀。
- 這是全市場統計，不是這一檔的統計，個股的產業與籌碼結構沒有納入。
- 樣本數低於 `MIN_SAMPLES` 一律不給結論。
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend.pattern_base import Pattern
from paths import DATA_DIR

HOLD_DAYS = (5, 20)
MIN_SAMPLES = 30
# 勝率差距小於這個值視為沒有差別
EDGE_TOLERANCE = 0.01


@st.cache_data(ttl=3600, show_spinner=False)
def _payload() -> dict:
    path = DATA_DIR / "pattern_stats.json"
    if not path.exists():
        return {"baseline": {}, "patterns": {}}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600, show_spinner=False)
def baseline() -> dict:
    """對照組：全市場所有日子隨便買的結果（全歷史）。"""
    return _payload().get("baseline") or {}


def pattern_stats(pattern: Pattern) -> dict | None:
    """某條說法的全市場全歷史統計；沒有就回 None。"""
    entry = _payload().get("patterns", {}).get(pattern.key)
    return entry.get("stats") if entry else None


def stats_by_key(key: str) -> dict | None:
    entry = _payload().get("patterns", {}).get(key)
    return entry.get("stats") if entry else None


def all_pattern_entries() -> dict[str, dict]:
    """key → {name, tone, group, text, stats}，給「148 條全表」那一頁用。"""
    return _payload().get("patterns", {})


# ── 命中矩陣（測試期）──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def hits() -> pd.DataFrame:
    path = DATA_DIR / "pattern_hits.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def pattern_keys() -> tuple[str, ...]:
    frame = hits()
    return tuple(c for c in frame.columns if c not in ("date", "stock_id"))


def hits_for(stock_id: str, date: pd.Timestamp) -> tuple[str, ...]:
    """這檔股票這一天成立的所有說法（讀矩陣，前端不做任何判定）。"""
    frame = hits()
    if frame.empty:
        return ()
    row = frame[(frame["stock_id"] == stock_id) & (frame["date"] == pd.Timestamp(date))]
    if row.empty:
        return ()
    values = row.iloc[0]
    return tuple(k for k in pattern_keys() if int(values[k]) == 1)


@st.cache_data(ttl=3600, show_spinner=False)
def forward_returns() -> pd.DataFrame:
    """全市場每日的未來報酬（進場 = 隔日收盤）。定義與 engine 版一字不差。"""
    path = DATA_DIR / "price_test.parquet"
    if not path.exists():
        return pd.DataFrame()
    price = pd.read_parquet(path, columns=["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"])
    entry = price.groupby("stock_id")["close"].shift(-1)
    out = price[["date", "stock_id"]].copy()
    for days in HOLD_DAYS:
        exit_price = price.groupby("stock_id")["close"].shift(-(days + 1))
        out[f"ret{days}"] = (exit_price - entry) / entry
    return out


def _summarise(frame: pd.DataFrame) -> dict:
    out: dict = {"n": int(len(frame))}
    for days in HOLD_DAYS:
        series = frame[f"ret{days}"].dropna()
        out[f"n{days}"] = int(len(series))
        out[f"win{days}"] = float((series > 0).mean()) if len(series) else None
        out[f"median{days}"] = float(series.median()) if len(series) else None
        out[f"mean{days}"] = float(series.mean()) if len(series) else None
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def test_period_baseline() -> dict:
    """聯合條件要跟**同期間**的對照組比，不能拿全歷史的 baseline 去比。"""
    frame = forward_returns()
    return _summarise(frame) if not frame.empty else {}


@st.cache_data(ttl=3600, show_spinner=False)
def joint_stats(keys: tuple[str, ...]) -> dict | None:
    """**同時**符合這幾個條件的日子，之後怎麼走（測試期口徑）。

    綜合結論一定要走這裡，不能把各條說法的勝率平均起來 —— 今天成立的十條說法
    彼此高度重疊（均線多頭排列與創新高幾乎同時發生），平均等於把同一件事算了
    很多次，得到的數字沒有任何意義。
    """
    frame = hits()
    if frame.empty:
        return None
    usable = [k for k in keys if k in frame.columns]
    if not usable:
        return None
    mask = pd.Series(True, index=frame.index)
    for key in usable:
        mask &= frame[key] == 1
    hit = frame.loc[mask, ["date", "stock_id"]]
    if hit.empty:
        return {"n": 0, "n5": 0, "n20": 0}
    joined = hit.merge(forward_returns(), on=["date", "stock_id"], how="inner")
    return _summarise(joined)


@st.cache_data(ttl=3600, show_spinner=False)
def crossing_history(stock_id: str, price: float, downward: bool) -> dict:
    """這檔股票歷史上「穿越某個價位」之後的走勢（測試期口徑）。

    事件定義是**穿越當天**：前一天收盤在價位的另一側，當天收盤穿過去。用「收盤在
    價位下方的所有日子」當樣本是錯的 —— 那會把一路陰跌的整段期間重複計入，樣本
    彼此高度重疊，勝率會被同一段行情灌爆。
    """
    path = DATA_DIR / "price_test.parquet"
    if not path.exists():
        return {"n": 0}
    price_all = pd.read_parquet(path, columns=["date", "stock_id", "close"])
    one = price_all[price_all["stock_id"] == stock_id].copy()
    if one.empty:
        return {"n": 0}
    one["date"] = pd.to_datetime(one["date"])
    one = one.sort_values("date")
    previous = one["close"].shift(1)
    if downward:
        event = (previous > price) & (one["close"] <= price)
    else:
        event = (previous < price) & (one["close"] >= price)

    hit = one[event.fillna(False)]
    if hit.empty:
        return {"n": 0}
    joined = hit[["date", "stock_id"]].merge(
        forward_returns(), on=["date", "stock_id"], how="inner")
    return _summarise(joined)


def is_conclusive(stats: dict | None, days: int = 20) -> bool:
    return bool(stats) and (stats.get(f"n{days}") or 0) >= MIN_SAMPLES


def verdict(stats: dict, base: dict, days: int = 20) -> str:
    """跟對照組比出來的一句話。差距小於 1 個百分點視為沒有差別。"""
    edge = (stats.get(f"win{days}") or 0) - (base.get(f"win{days}") or 0)
    if abs(edge) < EDGE_TOLERANCE:
        return "與隨機進場沒有明顯差別"
    return f"勝率{'高' if edge > 0 else '低'}於隨機進場 {abs(edge):.1%}"
