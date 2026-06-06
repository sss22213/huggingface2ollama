"""FastAPI app: HuggingFace -> Ollama model downloader web UI."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import settings as settings_mod
from . import runner, downloader, jobs

app = FastAPI(title="HuggingFace -> Ollama")

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


# ---------------- Settings ----------------
class SettingsPatch(BaseModel):
    ollama_data_dir: str | None = None
    run_mode: str | None = None
    ollama_host: str | None = None
    binary_path: str | None = None
    docker_container: str | None = None
    docker_command: str | None = None
    hf_token: str | None = None
    download_dir: str | None = None


@app.get("/api/settings")
def get_settings():
    return settings_mod.load().public()


@app.post("/api/settings")
def save_settings(patch: SettingsPatch):
    data = {k: v for k, v in patch.model_dump().items() if v is not None}
    s = settings_mod.update(data)
    return s.public()


@app.get("/api/test-connection")
def test_connection():
    """Verify ollama is reachable and the data dir is usable."""
    s = settings_mod.load()
    ver = runner.version()
    ok_dir, dir_msg = runner.check_dir_writable(s.ollama_data_dir)
    return {
        "ollama_ok": ver.ok,
        "ollama_version": ver.output if ver.ok else "",
        "ollama_error": "" if ver.ok else ver.output,
        "run_mode": s.run_mode,
        "ollama_host": s.ollama_host,
        "data_dir": s.ollama_data_dir,
        "data_dir_ok": ok_dir,
        "data_dir_msg": dir_msg,
    }


# ---------------- Ollama status ----------------
@app.get("/api/ps")
def api_ps():
    res = runner.ps()
    return {
        "ok": res.ok,
        "raw": res.output,
        "rows": runner.parse_table(res.stdout) if res.ok else [],
    }


@app.get("/api/installed")
def api_installed():
    res = runner.list_models()
    return {
        "ok": res.ok,
        "raw": res.output,
        "rows": runner.parse_table(res.stdout) if res.ok else [],
    }


# ---------------- HuggingFace ----------------
@app.get("/api/hf/files")
def hf_files(repo_id: str, revision: str = "main"):
    s = settings_mod.load()
    try:
        files = downloader.list_files(repo_id, s.hf_token or None, revision)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "repo_id": repo_id,
        "files": [
            {"path": f.path, "size": f.size,
             "is_gguf": f.path.lower().endswith(".gguf"),
             "is_safetensors": f.path.lower().endswith(".safetensors")}
            for f in files
        ],
    }


# ---------------- Import jobs ----------------
class ImportReq(BaseModel):
    repo_id: str
    files: list[str]
    fmt: str  # gguf | safetensors
    model_name: str
    revision: str = "main"
    extra_modelfile: str = ""


@app.post("/api/import")
def api_import(req: ImportReq):
    if not req.repo_id or not req.model_name:
        raise HTTPException(status_code=400, detail="repo_id and model_name are required")
    if not req.files:
        raise HTTPException(status_code=400, detail="select at least one file to download")
    job = jobs.start_import(
        req.repo_id, req.files, req.fmt, req.model_name, req.revision, req.extra_modelfile
    )
    return {"job_id": job.id}


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": jobs.all_jobs()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
