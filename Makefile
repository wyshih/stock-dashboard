.DEFAULT_GOAL := help
SHELL := /bin/bash
PY   := .venv/bin/python
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
