# 🦙 HuggingFace → Ollama Downloader

> A web UI to download models from **HuggingFace** (GGUF or Safetensors) and import them into **Ollama** — works with a local install *or* a dockerised Ollama, no CLI required.
>
> 一個把 **HuggingFace** 模型（GGUF / Safetensors）下載並匯入 **Ollama** 的網頁工具，支援本機或 Docker 裡的 Ollama，全程不需指令列。

<p>
  <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL%203.0-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688.svg">
  <img alt="Built with AI + Human" src="https://img.shields.io/badge/built%20with-AI%20%2B%20Human-ff69b4.svg">
</p>

---

## 🤝 AI × Human co-development｜AI 與人類共同開發

**This project was built collaboratively by a human developer and an AI assistant (Anthropic's Claude, via Claude Code).**
The human set the goals, made the product decisions, tested against a real Ollama instance, and reviewed every change; the AI proposed the architecture, wrote the code, diagnosed the environment, and iterated on feedback. Neither did it alone — it is a genuine pair effort.

**本專案由人類開發者與 AI 助手（Anthropic 的 Claude，透過 Claude Code）共同開發完成。**
由人類訂定需求、做產品決策、在真實 Ollama 環境上實測並審查每一次修改；由 AI 提出架構、撰寫程式、診斷環境並依回饋反覆調整。這不是任一方獨力完成，而是一次真正的「人機協作」。

> 🧠 We believe in being transparent about AI involvement. This README, the code, and the design were produced through human–AI collaboration.
> 我們主張對 AI 參與保持透明：本說明文件、程式碼與設計皆來自人機協作的成果。

---

## ✨ Features｜功能

- **🌐 多國語言 / i18n** — 右上角切換 **繁體中文 / English / 日本語 / 한국어 / 简体中文**，**會記住上次選擇**（存於瀏覽器 `localStorage`，首次依瀏覽器語言自動判斷）。
- **🖥️ Web UI** — 純瀏覽器操作，無需指令列。
- **📂 Ollama 資料目錄** — UI 上可填（預設 `/mnt/ollama`），本地模式會以此設定 `OLLAMA_MODELS`。
- **⚙️ 三種執行模式：**
  - **HTTP API（推薦，預設）** — 直接連 Ollama 服務（預設 `http://127.0.0.1:11434`）。本機或 Docker 內皆可用，**免 sudo、免掛載 volume**。
  - **本地 binary** — 呼叫本機安裝的 `ollama` 指令。
  - **Docker（`docker exec`）** — 對容器執行 `docker exec`，socket 需權限時可設 `sudo docker`。
- **📊 狀態檢視** — 顯示 `ollama ps`（執行中）與 `ollama list`（已安裝），可重新整理。
- **⬇️ 匯入流程** — 列出 HF repo 內檔案、勾選下載、即時進度，完成後自動建立 Ollama 模型；GGUF 與 Safetensors 皆支援。

---

## 🚀 Quick start｜安裝與啟動

```bash
./run.sh
# 預設啟動於 http://127.0.0.1:8765
```

第一次執行會自動建立 `.venv` 並安裝相依套件。也可手動：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

可用環境變數覆寫 `HOST` / `PORT`，例如要在區網內存取：

```bash
HOST=0.0.0.0 PORT=8765 ./run.sh
```

> Requirements: Python 3.10+, and a running Ollama (local binary, systemd, or Docker with port `11434` published).

---

## 📖 Usage｜使用步驟

1. **Ollama 設定**
   - 填資料目錄（如 `/mnt/ollama`，僅作參考；實際存放位置由 Ollama 服務本身設定決定）。
   - 選執行模式：
     - **HTTP API（推薦）**：填 API 位址（預設 `http://127.0.0.1:11434`）。最簡單，本機 / Docker 皆通。
     - **本地 binary**：填 ollama 路徑（如 `ollama` 或 `/usr/local/bin/ollama`）。
     - **Docker**：填容器名稱與 docker 指令（權限不足時用 `sudo docker`）。
   - （選填）填 HuggingFace Token 以下載私有 / 受限模型。
   - 按 **測試連線**（會自動先儲存目前表單）確認 Ollama 可用。
2. **Ollama 狀態** 區按重新整理，即可看到 `ollama ps` / `ollama list`。
3. **從 HuggingFace 匯入**
   - 填 repo id（如 `Qwen/Qwen2.5-0.5B-Instruct-GGUF`），按 **列出檔案**。
   - 勾選要下載的檔案（GGUF 可只選一個量化檔；Safetensors 需選權重 + 設定檔）。
   - 選格式、填模型名稱（如 `my-qwen:0.5b`），可加額外 Modelfile 指令。
   - 按 **開始下載並匯入**，於 **工作進度** 看即時 log 與進度條。

---

## 🔧 How it works｜運作方式

### HTTP API 模式（推薦）

1. 把 HF 檔案下載到主機暫存目錄。
2. 對每個模型檔計算 sha256，用 `POST /api/blobs/sha256:<digest>` 上傳到 Ollama（已存在則跳過）。
3. 呼叫 `POST /api/create`，把 `files` 對應到剛上傳的 blob，串流回傳建立進度。
4. 額外的 Modelfile 指令（`SYSTEM` / `TEMPLATE` / `PARAMETER`）會被解析成結構化欄位一併送出。

因為檔案透過 HTTP 上傳，**即使 Ollama 跑在 Docker 容器內也不需掛載 volume 或 `docker cp`**，也不需要 sudo。狀態查詢改用 `GET /api/ps` 與 `GET /api/tags`。

### Docker（docker exec）模式

1. 把 HF 檔案下載到主機暫存目錄。
2. 用 `docker cp` 把暫存目錄（含產生的 `Modelfile`）複製進容器的 `/tmp/h2o/`。
3. 在容器內執行 `ollama create <name> -f /tmp/h2o/.../Modelfile`。
4. 完成後清掉容器內暫存。

---

## 🗂️ Architecture｜架構

```
app/
  main.py            FastAPI 路由 + 靜態頁面
  settings.py        設定持久化 (settings.json)
  ollama_api.py      Ollama HTTP API：version / ps / tags / blob 上傳 / create
  runner.py          Ollama 執行抽象（api / binary / docker exec）+ 表格解析
  downloader.py      HuggingFace 檔案列舉與串流下載（含進度、續傳）
  jobs.py            背景匯入工作（下載 → 建立模型）
  static/index.html  單頁 Web UI（含 i18n）
```

執行階段產生的設定 `settings.json`、下載暫存 `downloads/`，皆已列入 `.gitignore`（前者可能含 HF Token，後者可能很大）。

---

## 📜 License｜授權

Licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE).

本專案以 **GNU GPL-3.0** 授權釋出，詳見 [LICENSE](LICENSE)。

```
HuggingFace → Ollama Downloader
Copyright (C) 2026  the project contributors

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed WITHOUT ANY WARRANTY; see the GNU GPL for
more details.
```

---

## 🙏 Acknowledgements｜致謝

- [Ollama](https://ollama.com) · [HuggingFace Hub](https://huggingface.co) · [FastAPI](https://fastapi.tiangolo.com)
- Built with [Claude Code](https://claude.com/claude-code) through human–AI collaboration.
