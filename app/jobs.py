"""Background import jobs: download from HuggingFace, then `ollama create`.

Each job runs in its own thread and accumulates a log + progress that the web
UI polls via /api/jobs/{id}.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import settings as settings_mod
from . import downloader, runner, ollama_api


@dataclass
class Job:
    id: str
    title: str
    status: str = "queued"          # queued | running | done | error
    stage: str = ""                 # human-readable current stage
    progress: float = 0.0           # 0..100 for the active file
    progress_label: str = ""
    log_lines: list[str] = field(default_factory=list)
    created: float = field(default_factory=lambda: time.time())
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 1),
            "progress_label": self.progress_label,
            "log": "\n".join(self.log_lines[-500:]),
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def all_jobs() -> list[dict]:
    with _lock:
        items = sorted(_jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.to_dict() for j in items]


def _log(job: Job, msg: str) -> None:
    for line in str(msg).splitlines() or [""]:
        job.log_lines.append(line)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s) or "model"


def start_import(
    repo_id: str,
    files: list[str],
    fmt: str,            # "gguf" | "safetensors"
    model_name: str,
    revision: str = "main",
    extra_modelfile: str = "",
) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], title=f"{repo_id} -> {model_name}")
    with _lock:
        _jobs[job.id] = job
    t = threading.Thread(
        target=_run_import,
        args=(job, repo_id, files, fmt, model_name, revision, extra_modelfile),
        daemon=True,
    )
    t.start()
    return job


def _run_import(job, repo_id, files, fmt, model_name, revision, extra_modelfile):
    s = settings_mod.load()
    try:
        job.status = "running"
        token = s.hf_token or None

        stage_dir = os.path.join(s.download_dir, _safe_name(model_name))
        os.makedirs(stage_dir, exist_ok=True)
        _log(job, f"Staging directory: {stage_dir}")

        if not files:
            raise ValueError("no files selected to download")

        # ---- Download stage ----
        local_paths: list[str] = []
        for idx, fname in enumerate(files, 1):
            job.stage = f"Downloading {idx}/{len(files)}: {fname}"
            _log(job, f"=== {job.stage} ===")

            def _progress(done, total, _f=fname):
                job.progress = (done * 100.0 / total) if total else 0.0
                job.progress_label = f"{_f}"

            path = downloader.download_file(
                repo_id, fname, stage_dir, token, revision,
                log=lambda m: _log(job, m),
                progress=_progress,
            )
            local_paths.append(path)

        # ---- Build Modelfile ----
        job.stage = "Importing into ollama (ollama create)"
        job.progress = 0.0
        job.progress_label = ""
        _log(job, f"=== {job.stage} ===")

        if s.run_mode == "api":
            # HTTP API: upload the relevant files as blobs and create.
            files_map = _api_files_map(fmt, local_paths)
            if not files_map:
                raise ValueError("no usable model files to upload")
            extras = ollama_api.parse_modelfile_extras(extra_modelfile)
            ollama_api.create(
                s.ollama_host, model_name, files_map, extras,
                log=lambda m: _log(job, m),
            )
        else:
            # binary / docker exec: write a Modelfile and run `ollama create`.
            modelfile = _build_modelfile(fmt, files, local_paths, stage_dir, extra_modelfile, s)
            res = runner.create_model(
                model_name, modelfile, stage_dir,
                log=lambda m: _log(job, m),
            )
            if not res.ok:
                raise RuntimeError(f"ollama create failed (exit {res.returncode})")

        job.stage = "Completed"
        job.progress = 100.0
        job.status = "done"
        _log(job, f"\n✔ Model '{model_name}' created successfully.")
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        _log(job, f"\n[ERROR] {e}")


def _api_files_map(fmt, local_paths) -> dict[str, str]:
    """Pick which downloaded files to upload via the API, keyed by base name.

    GGUF: only the .gguf parts. Safetensors: the weights plus config/tokenizer
    files ollama needs for conversion (docs/licenses are skipped)."""
    if fmt == "gguf":
        chosen = [p for p in local_paths if p.lower().endswith(".gguf")]
    else:
        skip = {".md", ".txt"}
        skip_names = {".gitattributes", "license", "license.txt", "readme.md"}
        chosen = [
            p for p in local_paths
            if os.path.splitext(p)[1].lower() not in skip
            and os.path.basename(p).lower() not in skip_names
        ]
    return {os.path.basename(p): p for p in chosen}


def _build_modelfile(fmt, files, local_paths, stage_dir, extra, s) -> str:
    """Construct the Modelfile FROM directive.

    GGUF: FROM points at the (first / merged) .gguf file.
    Safetensors: FROM points at the directory holding the converted weights.

    The path written into the Modelfile is relative ('./name') so that it
    resolves correctly both for a local `ollama create -f` (run with cwd not
    guaranteed) -- ollama resolves FROM relative to the Modelfile's directory --
    and after the whole stage_dir is `docker cp`'d into the container.
    """
    lines: list[str] = []
    if fmt == "gguf":
        ggufs = [p for p in local_paths if p.lower().endswith(".gguf")]
        if not ggufs:
            raise ValueError("no .gguf file among selected files")
        ggufs.sort()
        first = os.path.basename(ggufs[0])  # split models: point at -00001-of-...
        lines.append(f"FROM ./{first}")
    else:  # safetensors / full repo -> ollama converts the directory
        lines.append("FROM ./")
    if extra.strip():
        lines.append("")
        lines.append(extra.strip())
    return "\n".join(lines) + "\n"
