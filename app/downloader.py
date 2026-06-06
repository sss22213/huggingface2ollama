"""HuggingFace model downloading.

We use huggingface_hub's HfApi to enumerate repo files, then stream each file
ourselves with `requests` so we can report real byte-level progress to the web
UI (huggingface_hub's own progress goes to tqdm/stderr which is awkward to
surface in a browser). Downloads resume via HTTP Range when a partial file
exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from huggingface_hub import HfApi, hf_hub_url


@dataclass
class RepoFile:
    path: str
    size: int  # bytes, may be 0 if unknown


def list_files(repo_id: str, token: Optional[str], revision: str = "main") -> list[RepoFile]:
    """List files in a HuggingFace model repo with sizes."""
    api = HfApi(token=token or None)
    info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True, repo_type="model")
    files: list[RepoFile] = []
    for sib in info.siblings or []:
        size = getattr(sib, "size", None) or 0
        files.append(RepoFile(path=sib.rfilename, size=size))
    files.sort(key=lambda f: f.path)
    return files


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download_file(
    repo_id: str,
    filename: str,
    dest_dir: str,
    token: Optional[str],
    revision: str,
    log: Callable[[str], None],
    progress: Callable[[int, int], None],
) -> str:
    """Download one file from the repo into dest_dir, returning the local path.

    `progress(done_bytes, total_bytes)` is called as bytes arrive.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Preserve repo subdirectory structure under dest_dir.
    local_path = os.path.join(dest_dir, filename.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_path) or dest_dir, exist_ok=True)
    tmp_path = local_path + ".part"

    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    # Probe total size with a HEAD (follow redirects to the CDN).
    total = 0
    try:
        h = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        total = int(h.headers.get("Content-Length", 0))
    except Exception:
        total = 0

    if resume_from and total and resume_from >= total:
        os.replace(tmp_path, local_path)
        progress(total, total)
        log(f"{filename}: already complete ({_human(total)})")
        return local_path

    req_headers = dict(headers)
    mode = "wb"
    if resume_from:
        req_headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"
        log(f"{filename}: resuming from {_human(resume_from)}")

    with requests.get(url, headers=req_headers, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        if total == 0:
            cl = r.headers.get("Content-Length")
            if cl:
                total = int(cl) + resume_from
        done = resume_from
        last_logged = -1
        with open(tmp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                progress(done, total)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last_logged and pct % 5 == 0:
                        last_logged = pct
                        log(f"{filename}: {pct}%  ({_human(done)} / {_human(total)})")

    os.replace(tmp_path, local_path)
    progress(total or done, total or done)
    log(f"{filename}: done ({_human(os.path.getsize(local_path))})")
    return local_path
