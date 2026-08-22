"""台股預測系統 — 公開展示站。

⚠️ 這裡顯示的全部是模型**測試期（2025-02-01 ~ 2026-07-31）**的回溯結果，
不是即時預測，不構成投資建議。

刻意的設計：
- 只讀 `public_data/`，不做任何即時推論、不載入完整特徵、不需要模型檔。
- 148 條技術說法的判定是 engine 端離線算好的（`pattern_hits.parquet`），
  前端不再算 —— 所以這裡連 TA-Lib 都不需要，可以直接跑在 Streamlit
  Community Cloud 上。
- 圖表上的指標（MA / 布林 / VWAP / 量能）由 `price_test.parquet` 現算，
  那些只需要 OHLCV。

四頁：推薦名單 / 個股技術面 / 模型成效 / 關於。
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend import conditional_stats as stats_mod
from frontend import indicators
from frontend import technical_chart
from frontend import verdict as verdict_mod
from frontend.levels import Level, classify, cluster
from frontend.pattern_base import TONE_ICON, TONE_NAME
from frontend.patterns import PATTERNS_BY_KEY
from paths import DATA_DIR

st.set_page_config(page_title="台股預測系統（測試期展示）", page_icon="📈", layout="wide")

TEST_BANNER = ("📌 本站顯示的是模型**測試期（2025-02 ~ 2026-07）**的回溯結果，"
               "**不是即時預測**，不構成投資建議。")


# ── 資料讀取 ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(ttl=3600)
def load_scores() -> pd.DataFrame:
    path = DATA_DIR / "scores_test.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(ttl=3600)
def load_price(stock_id: str | None = None) -> pd.DataFrame:
    path = DATA_DIR / "price_test.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    if stock_id:
        frame = frame[frame["stock_id"] == stock_id]
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_stock_list() -> pd.DataFrame:
    path = DATA_DIR / "stock_list.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_sigcurve(model_key: str) -> pd.DataFrame:
    path = DATA_DIR / f"sigcurve_{model_key}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["threshold", "n", "win_rate", "avg_return"])
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_backtest() -> pd.DataFrame:
    path = DATA_DIR / "backtest_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_keys() -> list[str]:
    return [m["key"] for m in load_manifest().get("models", [])]


def chosen_threshold(key: str) -> float:
    """門檻讀 manifest，**不硬編**（CLAUDE.md 規則 7）。"""
    for entry in load_manifest().get("models", []):
        if entry["key"] == key:
            return float(entry["threshold"])
    return 0.5


def sigcurve_stats_at(curve: pd.DataFrame, threshold: float) -> dict | None:
    """曲線上「門檻 >= threshold」的那一段：筆數、勝率、平均報酬。"""
    if curve.empty:
        return None
    above = curve[curve["threshold"] >= threshold]
    if above.empty:
        return None
    row = above.iloc[-1]
    return {"n": int(row["n"]), "win_rate": float(row["win_rate"]),
            "avg_return": float(row["avg_return"]), "threshold": float(row["threshold"])}


def stock_name_map() -> dict[str, str]:
    frame = load_stock_list()
    if frame.empty or "stock_name" not in frame.columns:
        return {}
    return dict(zip(frame["stock_id"].astype(str), frame["stock_name"]))


# ── 第一頁：推薦名單 ──────────────────────────────────────────────────────────

def page_picks() -> None:
    st.title("推薦名單")
    st.warning(TEST_BANNER)

    scores = load_scores()
    if scores.empty:
        st.error("找不到 `public_data/scores_test.parquet`。"
                 "請在 engine 端執行 `make export-public`。")
        return

    keys = model_keys() or sorted(scores["model"].unique())
    dates = sorted(scores["date"].unique())

    left, right = st.columns([1, 1])
    with left:
        model_key = st.selectbox("模型", keys, index=0)
    with right:
        pick_date = st.selectbox("日期（測試期任一天）", dates,
                                 index=len(dates) - 1,
                                 format_func=lambda d: pd.Timestamp(d).date().isoformat())

    curve = load_sigcurve(model_key)
    default_thr = chosen_threshold(model_key)
    day = scores[(scores["model"] == model_key) & (scores["date"] == pick_date)]
    if day.empty:
        st.info("這一天沒有這個模型的分數。")
        return

    lo, hi = float(day["score"].min()), float(day["score"].max())
    thr = st.slider("分數門檻", min_value=round(lo, 4), max_value=round(hi, 4),
                    value=float(min(max(default_thr, lo), hi)), step=0.005,
                    help="預設值是使用者當初在驗證期（val_sel）曲線上挑定的門檻。")

    stats = sigcurve_stats_at(curve, thr)
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("驗證期訊號數", f"{stats['n']:,}")
        c2.metric("驗證期勝率", f"{stats['win_rate']:.1%}")
        c3.metric("驗證期平均報酬", f"{stats['avg_return']:+.2%}")
        st.caption("上面三個數字是**驗證期（2024 下半年）**曲線上的值，"
                   "不是這一天的結果 —— 門檻就是在那條曲線上挑的。")
    else:
        st.caption("這個門檻高過驗證期曲線的範圍，沒有對應的統計。")

    names = stock_name_map()
    picks = day[day["score"] >= thr].sort_values("score", ascending=False).copy()
    picks.insert(1, "stock_name", picks["stock_id"].map(names).fillna(""))
    st.subheader(f"{pd.Timestamp(pick_date).date()} 達標 {len(picks):,} 檔")
    st.dataframe(
        picks[["stock_id", "stock_name", "score"]].rename(
            columns={"stock_id": "代號", "stock_name": "名稱", "score": "分數"}),
        use_container_width=True, hide_index=True,
        column_config={"分數": st.column_config.NumberColumn(format="%.4f")})
    st.caption("名單是模型在**當天收盤後**算出來的分數，實際進場價會是隔天。"
               "這是回溯結果，不是今天的預測。")


# ── 第二頁：個股技術面 ────────────────────────────────────────────────────────

def _levels_from_price(ohlc: pd.DataFrame, close: float) -> list[Level]:
    """只用 OHLCV 算得出來的價位：均線、費波南希、樞紐點。

    engine 版是從 swing / trendline 特徵欄取前波高低點，那些特徵不會出到
    public，所以這裡改用 `indicators` 現算的價位 —— 少了前波高低點，
    但都是同樣「一條水平線」的語意，`verdict.build()` 照樣吃得下。
    """
    levels: list[Level] = []
    for window, series in indicators.moving_averages(ohlc).items():
        value = series.iloc[-1]
        if pd.notna(value):
            levels.append(Level(price=float(value), label=f"MA{window}", source="ma"))
    for label, price in indicators.fibonacci_levels(ohlc).items():
        if pd.notna(price):
            levels.append(Level(price=float(price), label=f"費波南希 {label}",
                                source="pivot"))
    for label, price in indicators.pivot_points(ohlc).items():
        if pd.notna(price):
            levels.append(Level(price=float(price), label=f"樞紐 {label}",
                                source="pivot"))
    return [lv for lv in levels if lv.price > 0]


def page_stock() -> None:
    st.title("個股技術面")
    st.warning(TEST_BANNER)

    listing = load_stock_list()
    price_all = load_price()
    if price_all.empty:
        st.error("找不到 `public_data/price_test.parquet`。")
        return

    ids = sorted(price_all["stock_id"].unique())
    names = stock_name_map()
    stock_id = st.selectbox(f"股票（{len(ids):,} 檔可查）", ids,
                            format_func=lambda s: f"{s} {names.get(s, '')}".strip())

    ohlc = load_price(stock_id)
    if len(ohlc) < 30:
        st.info("這檔在測試期的資料太短，畫不出有意義的圖。")
        return

    dates = list(ohlc["date"])
    as_of = st.select_slider("看哪一天", options=dates, value=dates[-1],
                             format_func=lambda d: pd.Timestamp(d).date().isoformat())
    window = ohlc[ohlc["date"] <= as_of].reset_index(drop=True)
    close = float(window["close"].iloc[-1])

    c1, c2, c3 = st.columns(3)
    c1.metric("收盤", f"{close:,.2f}")
    prev = float(window["close"].iloc[-2]) if len(window) > 1 else close
    c2.metric("漲跌", f"{close - prev:+.2f}", f"{(close / prev - 1):+.2%}" if prev else "")
    c3.metric("成交量", f"{float(window['volume'].iloc[-1]):,.0f}")

    levels = _levels_from_price(window, close)
    below, above = classify(levels, close)
    groups = cluster(below + above)
    figure = technical_chart.build_figure(window, close, groups, trendlines=[],
                                          layers=("ma", "bb", "vwap"))
    st.plotly_chart(figure, use_container_width=True)

    # ── 148 條說法 ───────────────────────────────────────────────────────────
    st.subheader("今天成立的說法")
    hit_keys = stats_mod.hits_for(stock_id, as_of)
    if not hit_keys:
        st.info("這一天沒有任何一條說法成立。")
    base = stats_mod.baseline()
    hits = [PATTERNS_BY_KEY[k] for k in hit_keys if k in PATTERNS_BY_KEY]

    if hits:
        conclusion = verdict_mod.build(hits, levels, close, base)
        col1, col2 = st.columns(2)
        col1.metric("偏多說法", conclusion.bullish_count)
        col2.metric("偏空說法", conclusion.bearish_count)
        if conclusion.is_conclusive and conclusion.edge is not None:
            st.success(
                f"同時符合 {len(conclusion.used)} 個條件的歷史樣本 "
                f"{conclusion.stats['n20']:,} 筆，20 日勝率 "
                f"{conclusion.stats['win20']:.1%}（vs 隨機進場 {base.get('win20', 0):.1%}，"
                f"差 {conclusion.edge:+.1%}）")
            st.caption("⚠️ 聯合條件的統計只有**測試期**的樣本（單條說法才是全歷史），"
                       "樣本數少很多，看看就好。")
        else:
            st.info("同時符合這些條件的歷史樣本太少，不給結論。")

        rows = []
        for pattern in hits:
            stats = stats_mod.pattern_stats(pattern)
            rows.append({
                "": TONE_ICON.get(pattern.tone, ""),
                "傾向": TONE_NAME.get(pattern.tone, ""),
                "說法": pattern.name,
                "分類": pattern.group,
                "敘述": pattern.text,
                "歷史樣本": stats.get("n20") if stats else None,
                "20日勝率": stats.get("win20") if stats else None,
                "20日中位報酬": stats.get("median20") if stats else None,
                "結論": (stats_mod.verdict(stats, base)
                          if stats_mod.is_conclusive(stats) else "樣本不足"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "20日勝率": st.column_config.NumberColumn(format="%.1f%%"),
                         "20日中位報酬": st.column_config.NumberColumn(format="%.2f%%"),
                     })

    with st.expander(f"148 條說法全表與各自的全歷史統計"):
        entries = stats_mod.all_pattern_entries()
        table = []
        for key, entry in entries.items():
            stats = entry.get("stats") or {}
            table.append({
                "": TONE_ICON.get(entry.get("tone"), ""),
                "說法": entry.get("name"), "分類": entry.get("group"),
                "今天成立": "✔" if key in hit_keys else "",
                "歷史樣本": stats.get("n20"),
                "20日勝率": stats.get("win20"),
                "20日中位報酬": stats.get("median20"),
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(f"對照組（全市場所有日子隨便買）：20 日勝率 "
                   f"{base.get('win20', 0):.1%}、中位報酬 {base.get('median20', 0):+.2%}、"
                   f"樣本 {base.get('n20', 0):,} 筆。"
                   "低於 30 筆樣本的一律不給結論。")


# ── 第三頁：模型成效 / 回測 ───────────────────────────────────────────────────

def page_performance() -> None:
    st.title("模型成效與回測")
    st.warning(TEST_BANNER)

    summary = load_backtest()
    if summary.empty:
        st.error("找不到 `public_data/backtest_summary.csv`。")
        return

    manifest = load_manifest()
    thresholds = {m["key"]: m["threshold"] for m in manifest.get("models", [])}

    st.subheader("① 絕對門檻版")
    st.caption("每個模型各自用**使用者挑定的門檻**跑完整測試期。"
               "口徑 `dedup=False`（每筆超過門檻的訊號都獨立進場），"
               "與挑門檻時看的曲線同一把尺。")
    absolute = summary[summary["mode"] == "absolute"].copy()
    absolute["threshold"] = absolute["model"].map(thresholds).fillna(absolute["threshold"])
    st.dataframe(_fmt_backtest(absolute), use_container_width=True, hide_index=True)

    st.subheader("② 訊號數對齊版（每日前 1.5%）")
    st.error("⚠️ **模型比較一定要看這一張。** 本系統「訊號越少報酬越高」，"
             "固定門檻的比較會退化成「門檻鬆緊」的比較，不是模型好壞的比較。"
             "這張表讓每個模型每天都只發相同比例的訊號，才是同一條起跑線。")
    matched = summary[summary["mode"] == "matched_top"]
    st.dataframe(_fmt_backtest(matched), use_container_width=True, hide_index=True)

    st.subheader("③ 驗證期門檻曲線")
    st.caption("門檻是**由人看這條曲線挑的**，不是自動規則算出來的。"
               "曲線畫的是驗證期（2024 下半年）在各個門檻下的訊號數與績效。")
    keys = model_keys()
    picked = st.multiselect("模型", keys, default=keys[:3])
    metric = st.radio("指標", ["win_rate", "avg_return"], horizontal=True,
                      format_func=lambda m: {"win_rate": "勝率",
                                             "avg_return": "平均報酬"}[m])
    curves = {}
    for key in picked:
        curve = load_sigcurve(key)
        if not curve.empty:
            curves[key] = curve.set_index("n")[metric]
    if curves:
        st.line_chart(pd.DataFrame(curves))
        st.caption("橫軸是訊號數（越右邊門檻越鬆）。")

    exit_rules = manifest.get("backtest", {}).get("exit_rules", {})
    if exit_rules:
        st.caption(
            f"出場規則：獲利 {exit_rules.get('trail_trigger', 0):.0%} 後啟動移動停利、"
            f"從最高收盤回落 {exit_rules.get('trail_pct', 0):.0%} 出場、"
            f"固定停損 {exit_rules.get('stop_loss', 0):.0%}"
            "（停損有值時 MA20 停損不生效）。")


def _fmt_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    cols = {"model": "模型", "threshold": "門檻", "trades": "交易筆數",
            "win_rate": "勝率", "avg_return": "平均單筆報酬",
            "total_return": "累積報酬", "sharpe": "Sharpe",
            "max_drawdown": "最大回撤"}
    out = frame[[c for c in cols if c in frame.columns]].rename(columns=cols)
    return out


# ── 第四頁：關於 ──────────────────────────────────────────────────────────────

def page_about() -> None:
    st.title("關於這個站")
    manifest = load_manifest()
    st.warning(TEST_BANNER)

    period = manifest.get("period", {})
    coverage = manifest.get("coverage", {})
    st.markdown(f"""
### 這是什麼

一個台股的「未來 20 個交易日會不會漲過半數天數」預測模型的**測試期成績單**。
10 個模型全部是 RandomForest，差別只在特徵集與 ground truth 的定義。

- 期間：**{period.get('start', '?')} ~ {period.get('end', '?')}**
  （{period.get('trading_days', '?')} 個交易日）
- 涵蓋：{coverage.get('stocks', '?'):,} 檔、{coverage.get('price_rows', 0):,} 列行情
- 資料源：{manifest.get('data_source', 'TWSE / TPEx 官方端點')}
- 產生時間：{manifest.get('generated_at', '?')}

### 切分

{manifest.get('split', '')}

兩個切分交界各留**一個月 embargo** —— ground truth 要看未來 20 個交易日，
交界緊貼的話訓練期末端的答案會落在驗證期裡，等於偷看。

### 10 個模型
""")
    models = manifest.get("models", [])
    if models:
        st.dataframe(pd.DataFrame([
            {"模型": m["key"], "挑定門檻": m["threshold"]} for m in models]),
            use_container_width=True, hide_index=True)
    st.markdown("""
門檻是**由人看驗證期曲線挑的**，不是自動規則算出來的，每次重訓都要重挑。

### 回測方法
""")
    backtest = manifest.get("backtest", {})
    st.json(backtest, expanded=True)

    st.markdown("### 已知的限制（請一定要看完）")
    for caveat in manifest.get("caveats", []):
        st.markdown(f"- {caveat}")
    st.markdown("""
- **生存偏差**：資料來自官方端點的現存清單，已下市的股票不在裡面。
  「歷史上出現這個型態後平均漲 X%」這種數字，天生偏樂觀 ——
  真的跌到下市的那些樣本從一開始就沒被算進去。
- **不是即時預測**：這裡的每一個數字都是回溯出來的。模型沒有在這個站上跑推論。
- **不是投資建議**：歷史績效不代表未來表現。任何投資決策的後果由你自己承擔。
""")
    st.error(manifest.get("disclaimer", "本站不構成任何投資建議。"))


# ── 進入點 ────────────────────────────────────────────────────────────────────

PAGES = {
    "推薦名單": page_picks,
    "個股技術面": page_stock,
    "模型成效／回測": page_performance,
    "關於": page_about,
}


def main() -> None:
    st.sidebar.title("台股預測系統")
    st.sidebar.caption("測試期展示站 · 非即時預測")
    choice = st.sidebar.radio("頁面", list(PAGES))
    manifest = load_manifest()
    if manifest:
        period = manifest.get("period", {})
        st.sidebar.info(f"資料期間\n\n{period.get('start')} ~ {period.get('end')}")
    st.sidebar.markdown("---")
    st.sidebar.caption("本站不構成投資建議。")
    PAGES[choice]()


main()
