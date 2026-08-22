.DEFAULT_GOAL := help
SHELL := /bin/bash
# 直譯器解析順序：本 repo 的 venv → engine 的 venv → 系統 python3。
# 中間那層是為了 engine 的 `make export-public` —— 它匯出完會來跑這裡的
# 安全檢查（public_data/ 有資料時那幾條才驗得到），但 dashboard 未必裝過
# 自己的 venv，寫死 .venv 會讓整條匯出流程斷在這裡。
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python \
        || (test -x ../engine/.venv/bin/python && echo ../engine/.venv/bin/python) \
        || echo python3)
PORT ?= 8501

.PHONY: help install app test

help:
	@echo ""
	@echo "  make install   建立 venv 並安裝套件（不需要 TA-Lib）"
	@echo "  make app       啟動展示站  http://localhost:$(PORT)"
	@echo "  make test      跑單元測試"
	@echo ""
	@echo "  資料包由 engine repo 的 \`make export-public\` 產生，"
	@echo "  放在 public_data/。這個 repo 不自己算任何東西。"
	@echo ""

install:
	@test -d .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

app:
	@if [ ! -f public_data/manifest.json ]; then \
	  echo ""; \
	  echo "  ⚠️  public_data/ 是空的。先到 engine repo 執行："; \
	  echo "        make export-public"; \
	  echo ""; \
	fi
	$(PY) -m streamlit run streamlit_app.py --server.port $(PORT)

test:
	$(PY) -m pytest tests/ -q
