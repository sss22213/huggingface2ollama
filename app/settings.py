"""Persisted application settings.

Settings are stored as JSON next to the project so they survive restarts.
The defaults match the user's test environment (ollama data dir at /mnt/ollama).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict, field

# settings.json lives at the project root (one level above this file's app/ dir)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(_ROOT, "settings.json")
# Where downloaded HuggingFace files are staged before being imported into ollama.
DEFAULT_DOWNLOAD_DIR = os.path.join(_ROOT, "downloads")


@dataclass
class Settings:
    # Ollama data directory (host path) -- exported as OLLAMA_MODELS in binary mode.
    ollama_data_dir: str = "/mnt/ollama"
    # How to reach ollama: "api" (HTTP API, recommended), "binary" (local
    # install) or "docker" (docker exec).
    run_mode: str = "api"
    # Ollama HTTP API base URL (api mode). Works for local or dockerised ollama.
    ollama_host: str = "http://127.0.0.1:11434"
    # Path to the local ollama binary (binary mode).
    binary_path: str = "ollama"
    # Docker container name/id that runs ollama (docker mode).
    docker_container: str = "ollama"
    # The docker command prefix. Some hosts need "sudo docker".
    docker_command: str = "docker"
    # Optional HuggingFace access token for gated/private repos.
    hf_token: str = ""
    # Host directory where HuggingFace files are downloaded.
    download_dir: str = DEFAULT_DOWNLOAD_DIR

    def public(self) -> dict:
        """Return a dict safe to send to the browser (token masked)."""
        d = asdict(self)
        d["hf_token_set"] = bool(self.hf_token)
        d.pop("hf_token", None)
        return d


_lock = threading.Lock()
_current: Settings | None = None


def load() -> Settings:
    global _current
    with _lock:
        if _current is not None:
            return _current
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
                _current = Settings(**known)
            except Exception:
                _current = Settings()
        else:
            _current = Settings()
        return _current


def update(patch: dict) -> Settings:
    """Apply a partial update and persist. Empty hf_token in patch is ignored
    so the user doesn't have to retype it on every save."""
    global _current
    with _lock:
        cur = _current or Settings()
        for k, v in patch.items():
            if k not in Settings.__dataclass_fields__:
                continue
            if k == "hf_token" and v == "":
                continue  # keep existing token
            setattr(cur, k, v)
        _current = cur
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(cur), f, indent=2, ensure_ascii=False)
        return cur
