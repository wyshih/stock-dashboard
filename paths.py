"""public repo 路徑的唯一來源。

engine repo 有 `engine/paths.py`，這裡是它的對應物。理由相同：路徑在多個檔案
裡各推導一遍，寫法遲早會漂移（engine 那邊曾經 22 個檔案各寫一次，其中一部分
沒有 resolve，在 symlink 下會指到別的地方）。

⚠️ 這個 repo **只**讀 `public_data/` —— 沒有 `data/`、沒有 `models/`。
資料包由 engine 的 `make export-public` 產生。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "public_data"
