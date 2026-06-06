# 🦙 HuggingFace → Ollama Downloader

> A web UI to download models from **HuggingFace** (GGUF or Safetensors) and import them into **Ollama** — works with a local install *or* a dockerised Ollama, no CLI required.

<p>
  <img alt="License: GPL-3.0" src="https://img.shields.io/badge/License-GPL%203.0-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688.svg">
  <img alt="Built with AI + Human" src="https://img.shields.io/badge/built%20with-AI%20%2B%20Human-ff69b4.svg">
</p>

---

## 🤝 AI × Human co-development

**This project was built collaboratively by a human developer and an AI assistant (Anthropic's Claude, via Claude Code).**

The human set the goals, made the product decisions, tested against a real Ollama instance, and reviewed every change. The AI proposed the architecture, wrote the code, diagnosed the environment, and iterated on feedback. Neither did it alone — it is a genuine pair effort.

> 🧠 We believe in being transparent about AI involvement. This README, the code, and the design were all produced through human–AI collaboration.

---

## ✨ Features

- **🌐 Internationalization** — Switch between **Traditional Chinese / English / Japanese / Korean / Simplified Chinese** from the top-right corner. **Your choice is remembered** (stored in the browser's `localStorage`; on first visit it auto-detects from your browser language).
- **🖥️ Web UI** — Everything in the browser, no command line needed.
- **📂 Ollama data directory** — Configurable in the UI (defaults to `/mnt/ollama`); binary mode exports it as `OLLAMA_MODELS`.
- **⚙️ Three run modes:**
  - **HTTP API (recommended, default)** — Talks to the Ollama service directly (default `http://127.0.0.1:11434`). Works for local *or* dockerised Ollama, with **no sudo and no volume mounts**.
  - **Local binary** — Invokes the locally installed `ollama` command.
  - **Docker (`docker exec`)** — Runs `docker exec` against the container; set the command to `sudo docker` when the socket needs privileges.
- **📊 Status view** — Shows `ollama ps` (running) and `ollama list` (installed), refreshable.
- **⬇️ Import flow** — Lists files in a HF repo, lets you pick which to download, streams live progress, then creates the Ollama model automatically. Both GGUF and Safetensors are supported.

---

## 🚀 Quick start

```bash
./run.sh
# Serves on http://127.0.0.1:8765 by default
```

The first run automatically creates `.venv` and installs dependencies. To do it manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Override `HOST` / `PORT` via environment variables, e.g. to expose it on your LAN:

```bash
HOST=0.0.0.0 PORT=8765 ./run.sh
```

> **Requirements:** Python 3.10+, and a running Ollama (local binary, systemd, or Docker with port `11434` published).

---

## 📖 Usage

1. **Ollama settings**
   - Set the data directory (e.g. `/mnt/ollama`; this is informational — the actual storage location is decided by the Ollama service itself).
   - Pick a run mode:
     - **HTTP API (recommended):** enter the API URL (default `http://127.0.0.1:11434`). Simplest option; works for both local and Docker.
     - **Local binary:** enter the ollama path (e.g. `ollama` or `/usr/local/bin/ollama`).
     - **Docker:** enter the container name and docker command (use `sudo docker` if the socket needs privileges).
   - (Optional) Enter a HuggingFace token to download private/gated models.
   - Click **Test connection** (it auto-saves the current form first) to confirm Ollama is reachable.
2. In the **Ollama status** section, click refresh to see `ollama ps` / `ollama list`.
3. **Import from HuggingFace**
   - Enter a repo id (e.g. `Qwen/Qwen2.5-0.5B-Instruct-GGUF`) and click **List files**.
   - Tick the files to download (for GGUF, one quantization file is enough; for Safetensors, select the weights + config files).
   - Choose the format, set a model name (e.g. `my-qwen:0.5b`), and optionally add extra Modelfile directives.
   - Click **Download & import** and watch the live log and progress bar under **Job progress**.

---

## 🔧 How it works

### HTTP API mode (recommended)

1. Download the HF files into a staging directory on the host.
2. Compute the sha256 of each model file and upload it via `POST /api/blobs/sha256:<digest>` (skipped if Ollama already has it).
3. Call `POST /api/create`, mapping `files` to the uploaded blobs, and stream back creation progress.
4. Extra Modelfile directives (`SYSTEM` / `TEMPLATE` / `PARAMETER`) are parsed into the structured fields and sent along.

Because files are uploaded over HTTP, **importing works even when Ollama runs inside a Docker container — no volume mount or `docker cp` needed**, and no sudo. Status is read via `GET /api/ps` and `GET /api/tags`.

### Docker (`docker exec`) mode

1. Download the HF files into a staging directory on the host.
2. Use `docker cp` to copy the staging directory (including the generated `Modelfile`) into the container's `/tmp/h2o/`.
3. Run `ollama create <name> -f /tmp/h2o/.../Modelfile` inside the container.
4. Clean up the in-container staging afterward.

---

## 🗂️ Architecture

```
app/
  main.py            FastAPI routes + static page
  settings.py        Settings persistence (settings.json)
  ollama_api.py      Ollama HTTP API: version / ps / tags / blob upload / create
  runner.py          Ollama execution abstraction (api / binary / docker exec) + table parsing
  downloader.py      HuggingFace file listing + streaming download (progress, resume)
  jobs.py            Background import jobs (download → create model)
  static/index.html  Single-page web UI (with i18n)
```

The runtime-generated `settings.json` and the `downloads/` staging directory are both listed in `.gitignore` (the former may contain your HF token, the latter can be large).

---

## 📜 License

Licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE).

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

## 🙏 Acknowledgements

- [Ollama](https://ollama.com) · [HuggingFace Hub](https://huggingface.co) · [FastAPI](https://fastapi.tiangolo.com)
- Built with [Claude Code](https://claude.com/claude-code) through human–AI collaboration.
