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

五頁：推薦名單 / 訊號清單 / 個股預測走勢 / 個股技術面 / 關於。
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
def load_scores(model_key: str) -> pd.DataFrame:
    """單一模型的分數 —— 資料包一個模型一個檔，不用把所有模型全讀進來。"""
    path = DATA_DIR / f"scores_test_{model_key}.parquet"
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


# 門檻滑桿的刻度：0.50 ~ 1.00，每 1% 一格。資料包裡的曲線就是抽樣到這些點上匯出的
# （engine/export/build_public_bundle.py 的 SLIDER_GRID），兩邊必須一致。
SLIDER_MIN = 0.50
SLIDER_MAX = 1.00
SLIDER_STEP = 0.01


@st.cache_data(ttl=3600)
def load_sigcurve(model_key: str) -> pd.DataFrame:
    # .gz：資料包裡的曲線是整條 gzip 的（不抽樣，一列不少）。read_csv 認得副檔名，
    # 會自己解壓。留 .csv 的後備路徑，舊資料包也讀得起來。
    path = DATA_DIR / f"sigcurve_{model_key}.csv.gz"
    if not path.exists():
        path = DATA_DIR / f"sigcurve_{model_key}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["threshold", "n", "win_rate", "avg_return"])
    return pd.read_csv(path)


def model_keys() -> list[str]:
    return [m["key"] for m in load_manifest().get("models", [])]


def model_name(key: str) -> str:
    """選單顯示的名稱（<特徵集>·<目標>）。manifest 沒帶 name 就退回代號。

    名字由 engine 端的 `bundle.MODEL_NAMES` 決定，這裡只是顯示 —— 兩邊各寫一份
    會飄掉，所以不在這個 repo 裡硬編任何模型名稱。
    """
    for entry in load_manifest().get("models", []):
        if entry["key"] == key:
            return entry.get("name") or key
    return key


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

    keys = model_keys()
    if not keys:
        st.error("找不到 `public_data/manifest.json`。"
                 "請在 engine 端執行 `make export-public`。")
        return

    # 先選模型才知道要讀哪一個檔，日期選單再由那個檔的內容決定。
    left, right = st.columns([1, 1])
    with left:
        model_key = st.selectbox("模型", keys, index=0, format_func=model_name)

    scores = load_scores(model_key)
    if scores.empty:
        st.error(f"找不到 `public_data/scores_test_{model_key}.parquet`。"
                 "請在 engine 端執行 `make export-public`。")
        return

    dates = sorted(scores["date"].unique())
    with right:
        pick_date = st.selectbox("日期（測試期任一天）", dates,
                                 index=len(dates) - 1,
                                 format_func=lambda d: pd.Timestamp(d).date().isoformat())

    curve = load_sigcurve(model_key)
    default_thr = chosen_threshold(model_key)
    day = scores[scores["date"] == pick_date]
    if day.empty:
        st.info("這一天沒有這個模型的分數。")
        return

    thr = st.slider("分數門檻", min_value=SLIDER_MIN, max_value=SLIDER_MAX,
                    value=round(min(max(default_thr, SLIDER_MIN), SLIDER_MAX), 2),
                    step=SLIDER_STEP,
                    help="預設值是使用者當初在驗證期（val_sel）曲線上挑定的門檻。"
                         "刻度是固定的 1%，與資料包裡的曲線刻度一致。")

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



# ── 訊號清單：該模型在測試期所有「分數 ≥ 門檻」的紀錄 ────────────────────────

def compute_forward_returns(price: pd.DataFrame, bars: int = 20) -> pd.DataFrame:
    """每一列往後第 `bars` 根 K 的收盤報酬（%）。純函式，可單獨測試。

    資料包的價格刻意多帶 20 個交易日，所以期間最後那批訊號也算得出結果；
    真的還沒走完 20 天的會是 NaN，前端要顯示成「未定案」而不是虧損。

    ⚠️ `shift` 前一定要先依 (stock_id, date) 排序 —— 少了排序不會報錯，
    只會安靜地把別檔股票或別天的價格算進來。
    """
    frame = price[["date", "stock_id", "close"]].sort_values(["stock_id", "date"])
    future = frame.groupby("stock_id")["close"].shift(-bars)
    return pd.DataFrame({"date": frame["date"], "stock_id": frame["stock_id"],
                         "fwd20": (future / frame["close"] - 1) * 100})


@st.cache_data(ttl=3600)
def forward_returns(bars: int = 20) -> pd.DataFrame:
    """快取版：自己去讀價格，不吃 DataFrame 參數。

    `st.cache_data` 必須雜湊參數，傳 733k 列的 DataFrame 進來會讓每次 rerun
    都重新雜湊整份（Streamlit Community Cloud 只有 1GB 記憶體）。
    """
    return compute_forward_returns(load_price(), bars)



SIGNAL_ROW_LIMIT = 3000     # 一次最多渲染這麼多列，再多瀏覽器會卡


def page_signals() -> None:
    st.title("訊號清單")
    st.warning(TEST_BANNER)
    st.caption("這個模型在測試期所有「分數 ≥ 門檻」的紀錄。"
               "點任一列可以跳到那檔股票在那一天的技術面。")

    keys = model_keys()
    if not keys:
        st.error("找不到 `public_data/manifest.json`。"
                 "請在 engine 端執行 `make export-public`。")
        return

    model_key = st.selectbox("模型", keys, index=0, key="siglist_model",
                             format_func=model_name)
    scores = load_scores(model_key)
    if scores.empty:
        st.error(f"找不到 `public_data/scores_test_{model_key}.parquet`。")
        return

    default_thr = chosen_threshold(model_key)
    days = sorted(scores["date"].dt.date.unique())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        thr = st.slider("分數門檻", min_value=SLIDER_MIN, max_value=SLIDER_MAX,
                        value=round(min(max(default_thr, SLIDER_MIN), SLIDER_MAX), 2),
                        step=SLIDER_STEP, key="siglist_thr",
                        help="預設值是使用者當初在驗證期曲線上挑定的門檻。")
    with c2:
        d_from = st.date_input("起", value=days[0], min_value=days[0],
                               max_value=days[-1], key="siglist_from")
    with c3:
        d_to = st.date_input("迄", value=days[-1], min_value=days[0],
                             max_value=days[-1], key="siglist_to")
    if d_from > d_to:
        st.error("起始日期不能晚於結束日期。")
        return

    sig = scores[(scores["score"] >= thr)
                 & (scores["date"] >= pd.Timestamp(d_from))
                 & (scores["date"] <= pd.Timestamp(d_to))].copy()
    if sig.empty:
        st.info(f"門檻 {thr:.2f} 在這段期間沒有任何訊號。")
        return

    # 附上股名、產業與訊號當天收盤價
    listing = load_stock_list()
    if not listing.empty:
        cols = [c for c in ["stock_id", "stock_name", "industry"] if c in listing.columns]
        sig = sig.merge(listing[cols], on="stock_id", how="left")
    price = load_price()
    if not price.empty:
        sig = sig.merge(price[["date", "stock_id", "close"]],
                        on=["date", "stock_id"], how="left")
        sig = sig.merge(forward_returns(), on=["date", "stock_id"], how="left")

    sig = sig.sort_values(["date", "score"], ascending=[False, False]).reset_index(drop=True)
    st.success(f"共 {len(sig):,} 筆訊號　"
               f"（{sig['date'].min().date()} ~ {sig['date'].max().date()}，"
               f"{sig['stock_id'].nunique():,} 檔股票）")

    view = pd.DataFrame({"日期": sig["date"].dt.date, "代號": sig["stock_id"]})
    if "stock_name" in sig:
        view["名稱"] = sig["stock_name"].fillna("")
    if "industry" in sig:
        view["產業"] = sig["industry"].fillna("")
    if "close" in sig:
        view["收盤"] = sig["close"]
    view["分數"] = sig["score"]
    if "fwd20" in sig:
        view["20日後報酬"] = sig["fwd20"]
        pending = int(sig["fwd20"].isna().sum())
        if pending:
            st.caption(f"其中 {pending:,} 筆的 20 個交易日還沒走完，報酬欄是空的 —— "
                       "**未定案，不是虧損**。")

    if len(view) > SIGNAL_ROW_LIMIT:
        st.caption(f"⚠️ 只顯示最新 {SIGNAL_ROW_LIMIT:,} 筆 —— "
                   "縮小日期區間或提高門檻就看得到更早的。")
    event = st.dataframe(
        view.head(SIGNAL_ROW_LIMIT), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="siglist_table",
        column_config={"分數": st.column_config.NumberColumn(format="%.4f"),
                       "收盤": st.column_config.NumberColumn(format="%.2f"),
                       "20日後報酬": st.column_config.NumberColumn(format="%+.2f%%")})
    try:
        rows = list(event.selection.rows)
    except Exception:
        rows = []
    if rows:
        hit = sig.iloc[rows[0]]
        target = (hit["stock_id"], hit["date"].date())
        # 只在「這次選取還沒處理過」時跳轉。表格的選取狀態會留在 session_state，
        # 少了這道判斷，使用者清除跳轉後切回本頁會立刻又被跳走（跳轉迴圈）。
        if st.session_state.get("siglist_handled") != target:
            st.session_state["siglist_handled"] = target
            st.session_state["stock_jump"] = target
            st.session_state["page"] = "個股技術面"
            st.rerun()

    st.download_button(
        "下載這份清單（CSV）", view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"signals_{model_key}_{d_from}_{d_to}.csv", mime="text/csv")
    st.caption("分數是**當天收盤後**算出來的，實際進場價會是隔天開盤之後。"
               "這是回溯結果，不是今天的預測。")


# ── 型態規則：fpm 專案挖出的平盤起漲點規則，測試期樣本外命中買點 ─────────────

@st.cache_data(ttl=3600)
def load_fpm_rule_hits() -> pd.DataFrame:
    path = DATA_DIR / "fpm_rule_hits.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["stock_id"] = df["stock_id"].astype(str)
    return df


@st.cache_data(ttl=3600)
def load_fpm_rule_stats() -> dict:
    path = DATA_DIR / "fpm_rule_stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def page_fpm_rules() -> None:
    st.title("型態規則")
    st.warning(TEST_BANNER)
    st.caption("另一個獨立專案（fpm）從歷史資料挖出的「平盤起漲點」規則，"
               "以下只顯示**測試期（2025-02~2026-07）樣本外**的歷史命中買點——"
               "規則沒看過這段資料，才能當驗證用。點任一列可以跳到那檔股票的走勢。")

    stats = load_fpm_rule_stats()
    hits = load_fpm_rule_hits()
    if not stats or hits.empty:
        st.error("找不到 `public_data/fpm_rule_hits.parquet` 或 `fpm_rule_stats.json`。"
                 "請在 engine 端執行 `make export-public`。")
        return

    with st.expander("⚠️ 這些規則的已知限制（一定要看）", expanded=False):
        for c in stats.get("caveats", []):
            st.markdown(f"- {c}")

    rules = stats.get("rules", [])
    name_by_id = {r["id"]: r["name"] for r in rules}
    stats_by_id = {r["id"]: r["stats"] for r in rules}

    selected = st.multiselect(
        "選規則（可複選，預設全選）", options=list(name_by_id),
        default=list(name_by_id),
        format_func=lambda rid: f"{rid}：{name_by_id[rid][:40]}",
        key="fpm_rule_select",
    )
    if not selected:
        st.info("至少選一條規則")
        return

    stats_df = pd.DataFrame([
        {"規則": rid, "名稱": name_by_id[rid], **stats_by_id[rid]}
        for rid in selected
    ])
    st.caption("勝率是嚴格定義（20日內漲幅夠大+統計顯著+過程無大回檔）下的命中比例，"
               "全市場基準約 2.3%——lift 是勝率相對基準的倍數。")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    filtered = hits[hits["rule_id"].isin(selected)].sort_values("date", ascending=False)
    st.success(f"共 {len(filtered):,} 筆歷史命中買點（{filtered['date'].min().date()} ~ "
               f"{filtered['date'].max().date()}，{filtered['stock_id'].nunique():,} 檔股票）")

    st.subheader("依買點聚合（同一天同一檔股票，符合了幾條規則）")
    st.caption("同時符合越多規則的買點，樣本外表現通常越好——這是型態疊加的訊號，"
               "不是單一規則各自獨立的訊號。")
    names_for_agg = stock_name_map()
    agg = filtered.groupby(["date", "stock_id"]).agg(
        符合規則數=("rule_id", "nunique"),
        規則清單=("rule_id", lambda s: ", ".join(sorted(s))),
        r_end=("r_end", "first"),
        label=("label", "first"),
    ).reset_index().sort_values(["符合規則數", "date"], ascending=[False, False])
    agg["名稱"] = agg["stock_id"].map(names_for_agg).fillna("")
    agg_display = agg.rename(columns={
        "date": "進場決策日", "stock_id": "股票", "r_end": "20日後超額報酬", "label": "是否起漲",
    })
    agg_display["20日後超額報酬"] = agg_display["20日後超額報酬"] * 100
    st.dataframe(
        agg_display.head(500), use_container_width=True, hide_index=True,
        column_config={"20日後超額報酬": st.column_config.NumberColumn(format="%+.2f%%")},
    )

    names = stock_name_map()
    filtered = filtered.reset_index(drop=True)
    view = pd.DataFrame({
        "日期": filtered["date"].dt.date,
        "代號": filtered["stock_id"],
        "名稱": filtered["stock_id"].map(names).fillna(""),
        "規則": filtered["rule_id"],
        "20日後超額報酬": filtered["r_end"] * 100,
        "期間最大回檔": filtered["mdd"] * 100,
        "是否起漲": filtered["label"].map({1: "✅", 0: "—"}),
    })
    st.dataframe(
        view.head(SIGNAL_ROW_LIMIT), use_container_width=True, hide_index=True,
        column_config={"20日後超額報酬": st.column_config.NumberColumn(format="%+.2f%%"),
                       "期間最大回檔": st.column_config.NumberColumn(format="%.2f%%")})

    # 用下拉選單＋按鈕跳轉，不用畫布表格的「點列」互動——後者在 glide-data-grid
    # 上不夠可靠（2026-09-04 實測：選取事件有時候不會真的觸發 rerun）。
    st.subheader("跳轉走勢圖")
    options = list(filtered.head(SIGNAL_ROW_LIMIT).index)
    pick_idx = st.selectbox(
        "選一筆看走勢圖（可打字搜尋股票代號，已按日期新到舊排序）", options,
        format_func=lambda i: (
            f"{filtered.loc[i,'date'].date()}　{filtered.loc[i,'stock_id']} "
            f"{names.get(filtered.loc[i,'stock_id'], '')}　{filtered.loc[i,'rule_id']}　"
            f"20日後{filtered.loc[i,'r_end']:+.1%}"
        ),
        key="fpm_rules_pick",
    )
    if st.button("跳轉 →", key="fpm_rules_jump"):
        row = filtered.loc[pick_idx]
        st.session_state["stock_jump"] = (row["stock_id"], row["date"].date())
        # 跳「個股預測走勢」而不是「個股技術面」：後者只看到跳轉日「為止」的圖，
        # 看不到之後有沒有真的漲；前者預設就是整段測試期，天生看得到後續走勢。
        st.session_state["page"] = "個股預測走勢"
        st.rerun()

    st.download_button(
        "下載這份清單（CSV）", view.to_csv(index=False).encode("utf-8-sig"),
        file_name="fpm_rule_hits.csv", mime="text/csv")


# ── 個股預測走勢：一檔股票在測試期每天的分數與股價 ──────────────────────────

# 圖裡各模型的線色。取自 dataviz 的分類色階前幾個 slot，明暗兩種主題都驗過
# 對比與色盲可辨識度。順序固定，不循環 —— 同一個模型在任何組合下都是同一個顏色。
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def page_history() -> None:
    st.title("個股預測走勢")
    st.warning(TEST_BANNER)
    st.caption("一檔股票在測試期每天的模型分數，與同期間的股價對照 —— "
               "看得出分數是不是在漲之前就先亮。")

    keys = model_keys()
    if not keys:
        st.error("找不到 `public_data/manifest.json`。請在 engine 端執行 `make export-public`。")
        return

    price_all = load_price()
    if price_all.empty:
        st.error("找不到 `public_data/price_test.parquet`。")
        return

    names = stock_name_map()
    ids = sorted(price_all["stock_id"].unique())
    left, right = st.columns([1, 1])
    with left:
        jump_id = st.session_state.get("stock_jump", (None, None))[0]
        if jump_id in ids and st.session_state.get("hist_pick") != jump_id:
            st.session_state["hist_pick"] = jump_id
        stock_id = st.selectbox(f"股票（{len(ids):,} 檔）", ids, key="hist_pick",
                                format_func=lambda s: f"{s} {names.get(s, '')}".strip())
    with right:
        picked = st.multiselect("模型（可複選）", keys, default=keys[:1],
                                format_func=model_name, key="hist_models")
    if not picked:
        st.info("請至少選一個模型。")
        return

    scores = {}
    for key in picked:
        frame = load_scores(key)
        one = frame[frame["stock_id"] == stock_id][["date", "score"]]
        if not one.empty:
            scores[key] = one.sort_values("date")
    if not scores:
        st.info("這檔股票在測試期沒有任何模型分數（多半是上市不久，特徵暖機期不足）。")
        return

    ohlc = load_price(stock_id)
    dates = sorted(set(ohlc["date"]).union(*(set(s["date"]) for s in scores.values())))
    c1, c2 = st.columns(2)
    with c1:
        d_from = st.date_input("起", value=dates[0].date(),
                               min_value=dates[0].date(), max_value=dates[-1].date(),
                               key="hist_from")
    with c2:
        d_to = st.date_input("迄", value=dates[-1].date(),
                             min_value=dates[0].date(), max_value=dates[-1].date(),
                             key="hist_to")
    if d_from > d_to:
        st.error("起始日期不能晚於結束日期。")
        return
    lo, hi = pd.Timestamp(d_from), pd.Timestamp(d_to)

    window = ohlc[(ohlc["date"] >= lo) & (ohlc["date"] <= hi)]
    if window.empty:
        st.info("這段期間沒有價格資料。")
        return

    st.plotly_chart(_history_figure(window, scores, picked, lo, hi),
                    use_container_width=True)
    st.caption("上圖是股價（K 棒），下圖是模型分數。虛線是各模型挑定的門檻，"
               "分數在虛線之上就是一筆訊號。兩張圖共用同一條時間軸。")

    _history_signal_table(scores, picked, lo, hi, stock_id)


def _history_figure(window: pd.DataFrame, scores: dict, picked: list,
                    lo: pd.Timestamp, hi: pd.Timestamp):
    """價格與分數上下排、共用 x 軸。

    刻意**不用雙 y 軸** —— 價格和分數的量級完全不同，疊在一起時兩條線的交叉點
    沒有任何意義，卻很容易被讀成「黃金交叉」。上下排一樣看得出時間先後關係。
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=("股價", "模型分數"))
    fig.add_trace(go.Candlestick(
        x=window["date"], open=window["open"], high=window["high"],
        low=window["low"], close=window["close"], name="股價",
        increasing_line_color="#d1493f", decreasing_line_color="#1b7f4d",
        showlegend=False), row=1, col=1)

    for i, key in enumerate(picked):
        if key not in scores:
            continue
        colour = MODEL_COLORS[i % len(MODEL_COLORS)]
        one = scores[key]
        one = one[(one["date"] >= lo) & (one["date"] <= hi)]
        fig.add_trace(go.Scatter(
            x=one["date"], y=one["score"], name=model_name(key), mode="lines",
            line=dict(color=colour, width=2),
            hovertemplate="%{x|%Y-%m-%d}　%{y:.4f}<extra>" + model_name(key) + "</extra>"),
            row=2, col=1)
        thr = chosen_threshold(key)
        fig.add_hline(y=thr, line=dict(color=colour, width=1, dash="dash"),
                      opacity=0.6, row=2, col=1)

    fig.update_layout(height=620, margin=dict(l=10, r=10, t=44, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.12),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="分數", row=2, col=1, range=[0, 1])
    return fig


def _history_signal_table(scores: dict, picked: list, lo: pd.Timestamp,
                          hi: pd.Timestamp, stock_id: str) -> None:
    """這段期間這檔股票發出過哪些訊號，以及結果。"""
    fwd = forward_returns()
    rows = []
    for key in picked:
        if key not in scores:
            continue
        thr = chosen_threshold(key)
        one = scores[key]
        hit = one[(one["date"] >= lo) & (one["date"] <= hi) & (one["score"] >= thr)]
        for _, r in hit.iterrows():
            rows.append({"日期": r["date"].date(), "模型": model_name(key),
                         "分數": r["score"], "門檻": thr, "date": r["date"]})
    if not rows:
        st.info("這段期間這檔股票沒有任何模型發出訊號。")
        return

    table = pd.DataFrame(rows)
    table["stock_id"] = stock_id
    table = table.merge(fwd, on=["date", "stock_id"], how="left")
    pending = int(table["fwd20"].isna().sum())
    st.subheader(f"這段期間的訊號　{len(table)} 筆")
    st.dataframe(
        table[["日期", "模型", "分數", "門檻", "fwd20"]].rename(
            columns={"fwd20": "20日後報酬"}).sort_values("日期", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={"分數": st.column_config.NumberColumn(format="%.4f"),
                       "門檻": st.column_config.NumberColumn(format="%.2f"),
                       "20日後報酬": st.column_config.NumberColumn(format="%+.2f%%")})
    if pending:
        st.caption(f"其中 {pending} 筆的 20 個交易日還沒走完，報酬欄是空的 —— "
                   "**未定案，不是虧損**。")


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

    # 從「訊號清單」點過來時，預先選好那檔股票、那一天。
    jump_id, jump_date = st.session_state.get("stock_jump", (None, None))
    # 選單用固定 key，不靠 index 決定 widget 身分 —— 否則 index 一變（例如按下
    # 「清除跳轉」）widget 會被當成新的，選擇被重置回第一檔。
    if jump_id in ids and st.session_state.get("stock_pick") != jump_id:
        st.session_state["stock_pick"] = jump_id
    stock_id = st.selectbox(f"股票（{len(ids):,} 檔可查）", ids, key="stock_pick",
                            format_func=lambda s: f"{s} {names.get(s, '')}".strip())
    if jump_id is not None:
        jc1, jc2 = st.columns([3, 1])
        jc1.info(f"已從訊號清單跳轉到 **{stock_id} {names.get(stock_id, '')}** 的 **{jump_date}**")
        if jc2.button("清除跳轉", use_container_width=True):
            del st.session_state["stock_jump"]
            st.rerun()

    ohlc = load_price(stock_id)
    if len(ohlc) < 30:
        st.info("這檔在測試期的資料太短，畫不出有意義的圖。")
        return

    dates = list(ohlc["date"])
    default_as_of = dates[-1]
    if jump_date is not None and stock_id == jump_id:
        on_or_before = [d for d in dates if d <= pd.Timestamp(jump_date)]
        if on_or_before:
            default_as_of = on_or_before[-1]
    as_of = st.select_slider("看哪一天", options=dates, value=default_as_of,
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


def page_about() -> None:
    st.title("關於這個站")
    manifest = load_manifest()
    st.warning(TEST_BANNER)
    st.info(
        "**本站純為學術研究與技術展示之用。**\n\n"
        "這裡呈現的是一個機器學習模型在歷史資料上的回溯測試結果，"
        "目的是檢驗方法論、公開研究過程供人檢視與批評 —— "
        "不是選股服務，不提供投資建議，也沒有任何營利行為。\n\n"
        "作者與本站不對任何人依據此處內容所作的決策負責。"
        "台灣證券市場的投資決策請自行研究，或諮詢合法的持牌投顧。")

    period = manifest.get("period", {})
    coverage = manifest.get("coverage", {})
    st.markdown(f"""
### 這是什麼

一個台股的「未來 20 個交易日會不會漲過半數天數」預測模型的**測試期成績單**。
{len(manifest.get('models', []))} 個模型全部是 RandomForest，用同一份特徵集，差別只在 ground truth 的定義。

- 期間：**{period.get('start', '?')} ~ {period.get('end', '?')}**
  （{period.get('trading_days', '?')} 個交易日）
- 涵蓋：{coverage.get('stocks', '?'):,} 檔、{coverage.get('price_rows', 0):,} 列行情
- 資料源：{manifest.get('data_source', 'TWSE / TPEx 官方端點')}
- 產生時間：{manifest.get('generated_at', '?')}

### 切分

{manifest.get('split', '')}

兩個切分交界各留**一個月 embargo** —— ground truth 要看未來 20 個交易日，
交界緊貼的話訓練期末端的答案會落在驗證期裡，等於偷看。

### 模型
""")
    models = manifest.get("models", [])
    if models:
        st.dataframe(pd.DataFrame([
            {"模型": m.get("name", m["key"]), "代號": m["key"],
             "挑定門檻": m["threshold"]} for m in models]),
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
- **研究用途**：本站是公開的研究紀錄，方法、限制與失敗都寫出來給人檢視。
  數字會隨資料更新而變動，也可能因為方法缺陷而是錯的 —— 發現問題歡迎指正。
""")
    st.error(manifest.get("disclaimer", "本站不構成任何投資建議。"))


# ── 進入點 ────────────────────────────────────────────────────────────────────

PAGES = {
    "推薦名單": page_picks,
    "訊號清單": page_signals,
    "個股預測走勢": page_history,
    "個股技術面": page_stock,
    "型態規則": page_fpm_rules,
    "關於": page_about,
}


def main() -> None:
    st.sidebar.title("台股預測系統")
    st.sidebar.caption("測試期展示站 · 非即時預測")
    # 各頁的「跳轉」都是 st.session_state["page"] = 目標頁 + st.rerun()，但
    # radio 元件一旦建立，Streamlit 就不准程式再改它綁定的 session_state
    # （2026-09-04 實測：StreamlitAPIException，型態規則頁的跳轉因此直接死掉，
    # 訊號清單的跳轉大概率也是同一個病，只是沒人特別點過測到）。
    # 修法：radio 換綁一個沒人會去改的 key，每次執行開頭先把「跳轉目標」
    # 同步進去，這時候元件還沒建立，改這個 key 合法。
    if "page" in st.session_state:
        st.session_state["page_radio"] = st.session_state.pop("page")
    choice = st.sidebar.radio("頁面", list(PAGES), key="page_radio")
    manifest = load_manifest()
    if manifest:
        period = manifest.get("period", {})
        st.sidebar.info(f"資料期間\n\n{period.get('start')} ~ {period.get('end')}")
    st.sidebar.markdown("---")
    st.sidebar.caption("本站不構成投資建議。")
    PAGES[choice]()


main()
