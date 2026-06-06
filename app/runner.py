"""Run ollama commands either via a local binary or via `docker exec`.

The two modes differ in two ways:
  * how the command is assembled, and
  * how host files (a Modelfile + a GGUF/safetensors dir) are made reachable
    by ollama when importing a model.

In binary mode we set OLLAMA_MODELS so ollama writes into the chosen data dir,
and pass host paths straight through.

In docker mode we `docker cp` the staging files into the container and run
`ollama create` against the in-container path -- this avoids requiring the user
to pre-configure a volume mount.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import shutil
from dataclasses import dataclass
from typing import Callable, Iterable

from . import settings as settings_mod
from . import ollama_api

# In-container staging dir used in docker mode.
CONTAINER_STAGE = "/tmp/h2o"


@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr else "")).strip()


def _docker_prefix(s: settings_mod.Settings) -> list[str]:
    # docker_command may be e.g. "docker" or "sudo docker"
    return shlex.split(s.docker_command or "docker")


def base_command(s: settings_mod.Settings) -> list[str]:
    """The argv prefix that turns trailing ollama args into a full command."""
    if s.run_mode == "docker":
        return _docker_prefix(s) + ["exec", s.docker_container, "ollama"]
    return [s.binary_path or "ollama", ]


def _env(s: settings_mod.Settings) -> dict:
    env = os.environ.copy()
    if s.run_mode == "binary" and s.ollama_data_dir:
        env["OLLAMA_MODELS"] = s.ollama_data_dir
    return env


def run(args: Iterable[str], timeout: int = 120) -> RunResult:
    """Run `ollama <args>` and capture output."""
    s = settings_mod.load()
    cmd = base_command(s) + list(args)
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_env(s),
        )
        return RunResult(p.returncode == 0, p.returncode, p.stdout, p.stderr)
    except FileNotFoundError as e:
        return RunResult(False, 127, "", f"command not found: {e}")
    except subprocess.TimeoutExpired:
        return RunResult(False, -1, "", f"timed out after {timeout}s")
    except Exception as e:  # pragma: no cover - defensive
        return RunResult(False, -1, "", str(e))


def ps() -> RunResult:
    """`ollama ps` -- currently loaded/running models."""
    s = settings_mod.load()
    if s.run_mode == "api":
        ok, rows, err = ollama_api.ps(s.ollama_host)
        return _api_result(ok, rows, err)
    return run(["ps"])


def list_models() -> RunResult:
    """`ollama list` -- installed models."""
    s = settings_mod.load()
    if s.run_mode == "api":
        ok, rows, err = ollama_api.tags(s.ollama_host)
        return _api_result(ok, rows, err)
    return run(["list"])


def version() -> RunResult:
    s = settings_mod.load()
    if s.run_mode == "api":
        ok, ver = ollama_api.version(s.ollama_host)
        return RunResult(ok, 0 if ok else 1, ver if ok else "", "" if ok else ver)
    return run(["--version"], timeout=20)


def _api_result(ok: bool, rows: list[dict], err: str) -> RunResult:
    """Wrap API rows in a RunResult, rendering a text table for the raw view
    and stashing rows so parse_table reproduces them."""
    if not ok:
        return RunResult(False, 1, "", err)
    if not rows:
        return RunResult(True, 0, "", "")
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    body = "\n".join("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows)
    return RunResult(True, 0, header + "\n" + body, "")


def parse_table(text: str) -> list[dict]:
    """Parse the whitespace-aligned table that `ollama ps`/`list` print.

    The header row defines column names; data columns are split on runs of 2+
    spaces so single-spaced values (e.g. '4.5 GB', '100% GPU') stay intact.
    """
    import re

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = re.split(r"\s{2,}", lines[0].strip())
    rows = []
    for ln in lines[1:]:
        cols = re.split(r"\s{2,}", ln.strip())
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        rows.append(dict(zip(header, cols[: len(header)])))
    return rows


def create_model(name: str, modelfile_text: str, stage_dir: str,
                 log: Callable[[str], None], timeout: int = 36000) -> RunResult:
    """Create an ollama model from a Modelfile.

    `stage_dir` is a host directory containing the model files referenced by the
    Modelfile (via FROM). The Modelfile is written into stage_dir as 'Modelfile'.
    In docker mode the whole stage_dir is copied into the container first.
    """
    s = settings_mod.load()
    os.makedirs(stage_dir, exist_ok=True)
    modelfile_path = os.path.join(stage_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_text)
    log(f"Modelfile written to {modelfile_path}")
    log("---- Modelfile ----\n" + modelfile_text + "\n-------------------")

    if s.run_mode == "docker":
        container_dir = f"{CONTAINER_STAGE}/{os.path.basename(stage_dir.rstrip('/'))}"
        prefix = _docker_prefix(s)
        # ensure target parent exists in container
        _stream(prefix + ["exec", s.docker_container, "mkdir", "-p", CONTAINER_STAGE], log)
        log(f"Copying staging files into container {s.docker_container}:{container_dir} ...")
        cp = subprocess.run(
            prefix + ["cp", stage_dir, f"{s.docker_container}:{container_dir}"],
            capture_output=True, text=True,
        )
        if cp.returncode != 0:
            return RunResult(False, cp.returncode, cp.stdout, "docker cp failed: " + cp.stderr)
        in_container_modelfile = f"{container_dir}/Modelfile"
        cmd = prefix + ["exec", s.docker_container, "ollama", "create", name, "-f", in_container_modelfile]
    else:
        cmd = [s.binary_path or "ollama", "create", name, "-f", modelfile_path]

    rc, out, err = _stream(cmd, log, env=_env(s), timeout=timeout)

    if s.run_mode == "docker":
        # best-effort cleanup of in-container staging
        try:
            container_dir = f"{CONTAINER_STAGE}/{os.path.basename(stage_dir.rstrip('/'))}"
            subprocess.run(_docker_prefix(s) + ["exec", s.docker_container, "rm", "-rf", container_dir],
                           capture_output=True, text=True, timeout=60)
        except Exception:
            pass

    return RunResult(rc == 0, rc, out, err)


def _stream(cmd: list[str], log: Callable[[str], None], env=None, timeout: int = 36000):
    """Run a command, streaming combined output line-by-line to `log`."""
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except FileNotFoundError as e:
        log(f"command not found: {e}")
        return 127, "", str(e)
    out_lines: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        out_lines.append(line)
        log(line)
    p.wait(timeout=timeout)
    return p.returncode, "\n".join(out_lines), ""


def check_dir_writable(path: str) -> tuple[bool, str]:
    if not path:
        return False, "empty path"
    if not os.path.isdir(path):
        return False, "directory does not exist"
    if not os.access(path, os.W_OK):
        return False, "directory is not writable by this process"
    return True, "ok"
