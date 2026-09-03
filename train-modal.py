import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

CPU = 16
MEMORY_MIB = 16384
TIMEOUT_S = 3600
REPO = "/repo"
VENV_PY = f"{REPO}/.venv/bin/python"
TB_PORT = 6006

VOL_NAME = "emojic-artifacts"
ARTIFACTS = "/artifacts"

DEP_FILES = ["pyproject.toml", "uv.lock", ".python-version", "README.md"]
CODE_FILES = [
    "config.py",
    "data.py",
    "model.py",
    "train.py",
    "test_emoji.py",
    "export_onnx.py",
    "run.py",
    "labels.json",
    "words.json",
    "data.jsonl",
    "train.jsonl",
    "eval.jsonl",
]
COLLECT_GLOBS = ["*.pt"]
COLLECT_TREES = ["runs", "web/public", "report"]

image = modal.Image.debian_slim(python_version="3.13").pip_install("uv")
for name in DEP_FILES:
    image = image.add_local_file(name, f"{REPO}/{name}", copy=True)
image = image.run_commands(f"cd {REPO} && uv sync --frozen")
for name in CODE_FILES:
    image = image.add_local_file(name, f"{REPO}/{name}", copy=True)

app = modal.App("emojic-train", image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _run_env(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "EMOJIC_NO_PROGRESS_BAR": "1",
        "EMOJIC_TASK_ONLY": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }


def _stash(dst: str) -> int:
    root, out = Path(REPO), Path(dst)
    n = 0
    for pattern in COLLECT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                shutil.copy2(path, out / path.name)
                n += 1
    for tree in COLLECT_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(root)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out / rel)
                n += 1
    return n


@app.function(
    cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, volumes={ARTIFACTS: vol}
)
def train_remote(threads: int = CPU) -> dict[str, int]:
    env = _run_env(threads)
    code = 1
    try:
        tb = subprocess.Popen(
            [
                VENV_PY,
                "-m",
                "tensorboard.main",
                "--logdir",
                f"{REPO}/runs",
                "--host",
                "0.0.0.0",
                "--port",
                str(TB_PORT),
                "--reload_interval",
                "5",
            ],
            cwd=REPO,
            env=env,
        )
        try:
            with modal.forward(TB_PORT) as tunnel:
                print(f"TensorBoard: {tunnel.url}", flush=True)
                proc = subprocess.Popen(
                    [VENV_PY, "train.py"],
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = proc.wait()
        finally:
            tb.terminate()
        if code == 0:
            subprocess.run(
                [VENV_PY, "test_emoji.py"], cwd=REPO, env=env, check=False
            )
    finally:
        n = _stash(ARTIFACTS)
        vol.commit()
        print(f"stashed {n} files to volume {VOL_NAME}", flush=True)
    if code != 0:
        raise RuntimeError(f"train.py exited with {code}")
    return {"files": n}


def _modal(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "modal", *args], capture_output=True, text=True
    )


def _retrieve_and_cleanup() -> None:
    staging = Path(tempfile.mkdtemp(prefix="emojic-modal-"))
    try:
        got = _modal("volume", "get", "--force", VOL_NAME, "/", str(staging))
        if got.returncode != 0:
            print(f"volume get failed, leaving {VOL_NAME} intact:")
            print(got.stdout, got.stderr)
            return

        src = staging
        kids = list(staging.iterdir())
        if len(kids) == 1 and kids[0].is_dir() and kids[0].name not in {
            "runs",
            "report",
            "web",
        }:
            src = kids[0]

        landed: list[str] = []
        for item in sorted(src.iterdir()):
            target = Path.cwd() / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            landed.append(item.name)

        print(f"retrieved into {Path.cwd()}: {', '.join(landed) or '(nothing)'}")
        if not landed:
            return

        rm = _modal("volume", "delete", "-y", VOL_NAME)
        print(
            f"deleted volume {VOL_NAME}"
            if rm.returncode == 0
            else f"volume cleanup failed:\n{rm.stderr}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@app.local_entrypoint()
def main(cpu: int = CPU, memory: int = MEMORY_MIB):
    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB:
        fn = train_remote.with_options(cpu=cpu, memory=memory)
    try:
        print(fn.remote(threads=cpu))
    finally:
        _retrieve_and_cleanup()
