import os
import subprocess
import sys
from pathlib import Path

import modal

CPU = 16
MEMORY_MIB = 16384
TIMEOUT_S = 3600
REPO = "/repo"
VENV_PY = f"{REPO}/.venv/bin/python"
TB_PORT = 6006

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


def _collect() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    root = Path(REPO)
    for pattern in COLLECT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                out[path.name] = path.read_bytes()
    for tree in COLLECT_TREES:
        for path in (root / tree).rglob("*"):
            if path.is_file():
                out[str(path.relative_to(root))] = path.read_bytes()
    return out


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S)
def train_remote(threads: int = CPU) -> dict[str, bytes]:
    env = _run_env(threads)
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
    if code != 0:
        raise RuntimeError(f"train.py exited with {code}")
    subprocess.run([VENV_PY, "test_emoji.py"], cwd=REPO, env=env, check=False)
    return _collect()


@app.local_entrypoint()
def main(cpu: int = CPU, memory: int = MEMORY_MIB):
    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB:
        fn = train_remote.with_options(cpu=cpu, memory=memory)
    artifacts = fn.remote(threads=cpu)
    for relpath, blob in sorted(artifacts.items()):
        dest = Path(relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        print(f"wrote {relpath} ({len(blob)} bytes)")
