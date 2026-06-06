"""Talk to ollama over its HTTP API (default http://127.0.0.1:11434).

This is the most robust transport: it works the same whether ollama runs as a
local process, a systemd service, or inside a Docker container with the port
published -- no binary on PATH and no `docker exec`/sudo required.

Model creation uploads the model files as blobs over HTTP, so the files reach
ollama's store even when ollama lives in a container, without any volume mount
or `docker cp`.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Optional

import requests


def _base(host: str) -> str:
    host = (host or "http://127.0.0.1:11434").strip().rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return host


def version(host: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        r = requests.get(_base(host) + "/api/version", timeout=timeout)
        r.raise_for_status()
        return True, r.json().get("version", "")
    except Exception as e:
        return False, str(e)


def _human(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def ps(host: str, timeout: int = 15) -> tuple[bool, list[dict], str]:
    """Running models -> rows matching the `ollama ps` columns."""
    try:
        r = requests.get(_base(host) + "/api/ps", timeout=timeout)
        r.raise_for_status()
        models = r.json().get("models", [])
        rows = []
        for m in models:
            until = m.get("expires_at", "")
            proc = _processor(m)
            rows.append({
                "NAME": m.get("name", ""),
                "ID": (m.get("digest", "")[:12]),
                "SIZE": _human(m.get("size", 0)),
                "PROCESSOR": proc,
                "CONTEXT": str(m.get("context_length", "") or ""),
                "UNTIL": until,
            })
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


def _processor(m: dict) -> str:
    total = m.get("size", 0) or 0
    vram = m.get("size_vram", 0) or 0
    if total <= 0:
        return ""
    if vram >= total:
        return "100% GPU"
    if vram <= 0:
        return "100% CPU"
    gpu = round(vram * 100 / total)
    return f"{gpu}% GPU/{100 - gpu}% CPU"


def tags(host: str, timeout: int = 20) -> tuple[bool, list[dict], str]:
    """Installed models -> rows matching the `ollama list` columns."""
    try:
        r = requests.get(_base(host) + "/api/tags", timeout=timeout)
        r.raise_for_status()
        models = r.json().get("models", [])
        rows = []
        for m in models:
            rows.append({
                "NAME": m.get("name", ""),
                "ID": (m.get("digest", "")[:12]),
                "SIZE": _human(m.get("size", 0)),
                "MODIFIED": m.get("modified_at", "")[:19].replace("T", " "),
            })
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_blob(host: str, path: str, log: Callable[[str], None]) -> str:
    """Upload a file as a blob if the server doesn't already have it. Returns
    the 'sha256:<digest>' reference."""
    base = _base(host)
    log(f"hashing {os.path.basename(path)} ...")
    digest = _sha256_file(path)
    ref = f"sha256:{digest}"
    # Already present?
    try:
        head = requests.head(f"{base}/api/blobs/{ref}", timeout=30)
        if head.status_code == 200:
            log(f"blob {ref[:19]}… already on server, skip upload")
            return ref
    except Exception:
        pass
    size = os.path.getsize(path)
    log(f"uploading {os.path.basename(path)} ({_human(size)}) ...")
    with open(path, "rb") as f:
        resp = requests.post(f"{base}/api/blobs/{ref}", data=f,
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=None)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"blob upload failed ({resp.status_code}): {resp.text[:300]}")
    log(f"uploaded {ref[:19]}…")
    return ref


def parse_modelfile_extras(text: str) -> dict:
    """Best-effort parse of a few common Modelfile directives into the
    structured /api/create fields. Unknown lines are ignored."""
    out: dict = {}
    params: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("SYSTEM "):
            out["system"] = _unquote(line[7:].strip())
        elif upper.startswith("TEMPLATE "):
            out["template"] = _unquote(line[9:].strip())
        elif upper.startswith("PARAMETER "):
            rest = line[10:].strip().split(None, 1)
            if len(rest) == 2:
                k, v = rest[0], _unquote(rest[1])
                # repeated params (e.g. stop) accumulate into a list
                if k in params:
                    if not isinstance(params[k], list):
                        params[k] = [params[k]]
                    params[k].append(v)
                else:
                    params[k] = v
        # FROM is handled separately by the caller
    if params:
        out["parameters"] = params
    return out


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def create(host: str, name: str, files: dict[str, str], extras: dict,
           log: Callable[[str], None]) -> bool:
    """Create a model via /api/create.

    `files` maps the in-request filename -> local host path. Each is uploaded as
    a blob first. `extras` may contain system/template/parameters.
    """
    base = _base(host)
    file_refs: dict[str, str] = {}
    for fname, path in files.items():
        file_refs[fname] = _ensure_blob(host, path, log)

    payload = {"model": name, "files": file_refs, "stream": True}
    payload.update(extras or {})

    log(f"POST /api/create  model={name}  files={list(file_refs.keys())}")
    with requests.post(f"{base}/api/create", json=payload, stream=True, timeout=None) as r:
        if r.status_code != 200:
            raise RuntimeError(f"create failed ({r.status_code}): {r.text[:500]}")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                log(line)
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"])
            status = obj.get("status")
            if status:
                log(status)
    return True
