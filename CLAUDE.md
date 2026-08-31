# 台股預測系統 — dashboard（public）

這是**公開**展示站。程式與資料都會被任何人看到，所以第一條規則是：

> **絕對不可以放進這個 repo：**
> 任何 `.pkl` 模型檔、`features.parquet` / `features_v3.parquet`、
> 2025-02 之前的任何資料、`.env` 或任何 secret、`doc/BACKTEST_LOG.md`。

資料包由隔壁的 private repo（`../engine/`）用 `make export-public` 產出，
放在 `public_data/`。**這個 repo 不自己算任何東西**：不推論、不建特徵、
不判定技術型態（148 條說法的命中在 engine 端就算好了）。

因此 `requirements.txt` 刻意**不含** TA-Lib、scikit-learn、lightgbm ——
這個站要能直接跑在 Streamlit Community Cloud 上。

## public_data/ 的內容

| 檔案 | 是什麼 |
|---|---|
| `price_test.parquet` | 測試期（2025-02-01 ~ 2026-07-31）全市場 OHLCV |
| `scores_test_m*.parquet` | 各模型該期間每日每股的分數（一個模型一個檔，×5）|
| `pattern_hits.parquet` | 148 條說法的命中矩陣（int8） |
| `pattern_stats.json` | 148 條說法的**全市場全歷史**條件統計 + 對照組 |
| `sigcurve_m*.csv.gz` | 各模型 val_sel 門檻曲線，整條 gzip、不抽樣（滑桿旁的數字讀這個）|
| `backtest_summary.csv` | 絕對門檻版 + 訊號數對齊版兩張表 |
| `stock_list.parquet` | 代號 / 名稱 / 市場 / 產業 |
| `manifest.json` | 期間、5 個模型與門檻、產生時間、資料口徑、免責聲明 |

`frontend/` 底下的 `indicators.py` / `levels.py` / `technical_chart.py` /
`pattern_base.py` / `patterns.py` / `patterns_talib.py` / `patterns_chip.py` /
`verdict.py` 與 private repo **完全相同**，改動要兩邊一起改。
`conditional_stats.py` 是 public 專用版（讀 `pattern_stats.json`，不碰特徵值）。

---

## 14 條踩坑規則（與 private repo 同一份）

1. **不用 yfinance，只用 TWSE / TPEx 官方端點。** yfinance 的 `Close` 永遠會做
   分割調整，某檔股票一旦分割就回頭改寫整段歷史；而抓取是增量的、只重寫尾端
   視窗 —— 每次分割都留下一道永久假跳空。全市場曾有 905/2028 檔中招、
   **3,599 道假接縫**。
2. **特徵一律經 `submodel_config.feature_cols()` 白名單取用**，不可以直接把
   `features.parquet` 的全部欄位餵給模型。
3. **shape 特徵永久排除。** KMeans 群心是用 2023-12-31 之前的資料 fit 的，
   是分布層級的洩漏。
4. **bundle 裡的訓練期 stats（median / mean / std）推論時一律沿用，不得重算。**
5. **`--full` 重建前先刪舊衍生檔。** builder 是 upsert 寫檔，舊資料獨有的列會殘留
   （2026-08-22 踩過假交易日的 1,948 列）。
6. **兩個切分交界各留一個月 embargo。**
7. **門檻由人看 val_sel 曲線挑，不用自動規則；程式讀檔不硬編；每次重訓必重挑。**
   本站的門檻一律讀 `manifest.json`，不寫死在 `streamlit_app.py` 裡。
   挑定的門檻必須落在滑桿刻度上（0.50~1.00，每 1% 一格），否則使用者拉不到它 ——
   engine 端 `build_public_bundle` 匯出時會擋，不會四捨五入成鄰近刻度。
8. **回測只用 engine 的 `backtest.simulate()` / `performance()`**，
   不得另寫 scratchpad 版。本站只顯示 engine 算好的結果，不重算。
9. **模型比較必須附訊號數對齊版（每日前 1.5%）。** 本系統「訊號越少報酬越高」，
   固定門檻的比較會退化成「門檻鬆緊」的比較。
   2026-08-31：「模型成效／回測」頁已從本站移除，所以這條目前沒有作用點；
   哪天要在公開站放回模型比較，兩張表都要在，而且要明講第二張才是公平比較。
   engine 端的 `make backtest` 與 BACKTEST_LOG 仍照這條規則走。
10. **不並行訓練**（10 核機器 RF 內層已吃 8 核，並行反而更慢）。
11. **`validate_data.py` 驗「資料裡最新那一天」，不是「今天」。**
12. **`fetch_stock_list` 必須最先跑，`build_features` 必須最後跑。**
13. **路徑只走 `paths.py`**（本 repo 是 `DATA_DIR = public_data/`，同樣只有一處定義）。
14. **跑完回測立刻寫 `doc/BACKTEST_LOG.md`** —— 那份 log 留在 private repo，
    **不要**複製過來。

## 回測方法（顯示的是 engine 用這套算出來的結果）

- `simulate()` / `performance()`，出場用 `CURRENT_EXIT_RULES`：
  `trail_trigger=0.15`、`trail_pct=0.10`、`stop_loss=0.20`
  （`stop_loss` 有值時 MA20 停損不生效）
- **`dedup=False`**（訊號層級口徑，與挑門檻的曲線同口徑）
- 每次比較附訊號數對齊版（每日前 1.5%）

## 頁面上的紀律

- 每一頁都要標明「測試期資料，非即時預測」。
- 任何統計都要附樣本數；低於 30 筆一律不給結論。
- 勝率一定要跟對照組（全市場隨便買）並列 —— 沒有基準的勝率沒有意義。
- 生存偏差、聯合條件只有測試期樣本 —— 這兩件事要寫在畫面上，不是只寫在這裡。

## Commit 規範

`feat / fix / refactor / test / docs / chore`
