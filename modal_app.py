"""Run the emojic training loop on Modal (cheap T4 GPU), finish locally.

    uv run modal setup            # one-time browser auth
    uv run modal run modal_app.py            # train on Modal, finish on this machine
    uv run modal run modal_app.py --resume   # continue from runs/last.ckpt on the Volume

What runs where
---------------
Modal (GPU): ``python train.py --no-post`` -- data load, ``trainer.fit``, ``model.pt``
saved on every new best ``acc/f/val``, TensorBoard event files + ``runs/last.ckpt``
written to a persistent Volume. A live TensorBoard URL is printed at the start of the
run. No ONNX export, no behavioral report.

This machine: the ``local_entrypoint`` downloads ``model.pt`` + ``runs/last.ckpt`` + the
event files (into ``./runs/``), then runs ``python train.py --post-only`` so the
``docs/`` web artifacts and ``report/model/<stamp>.md`` are regenerated locally, exactly
as a plain local ``python train.py`` would.

The image is built from the repo's own ``pyproject.toml`` + ``uv.lock`` (``uv sync``),
then the CPU-pinned ``torch`` is swapped for a CUDA build so the GPU is usable.
``pyproject.toml`` / ``uv.lock`` are not modified. The training subprocess runs the
venv's Python **directly** (not ``uv run``), because ``uv run`` re-syncs against the
lockfile on every start and would reinstall the CPU ``torch`` wheel.
"""

import io
import subprocess
import zipfile
from pathlib import Path

import modal

GPU = "T4"  # cheapest Modal GPU; fine for a ~42K-param char-CNN
REMOTE = "/repo"
VENV_PY = f"{REMOTE}/.venv/bin/python"
VENV_TB = f"{REMOTE}/.venv/bin/tensorboard"
PORT = 6006

# Model stack + data the training loop reads. Kept in sync with the repo root;
# added to the image after the deps layer so edits / corpus growth don't rebuild it.
REPO_FILES = [
    "config.py",
    "data.py",
    "model.py",
    "train.py",
    "test_model.py",
    "data.jsonl",
    "eval.jsonl",
    "labels.json",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    # deps layer: build the exact locked environment (incl. onnx/onnxscript).
    .add_local_file("pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file("README.md", f"{REMOTE}/README.md", copy=True)  # [project].readme
    .add_local_file(".python-version", f"{REMOTE}/.python-version", copy=True)
    .run_commands(
        f"cd {REMOTE} && uv sync --frozen",
        # Replace the CPU-pinned torch with a CUDA build (canonical PyTorch index).
        # `uv pip install` ignores [tool.uv.sources]; --python targets the synced
        # venv. The training subprocess must NOT use `uv run` or this is reverted.
        f"uv pip install --python {VENV_PY} --reinstall torch "
        "--index-url https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)
for _f in REPO_FILES:
    image = image.add_local_file(_f, f"{REMOTE}/{_f}")

app = modal.App("emojic-train", image=image)

# Persists runs/last.ckpt + TensorBoard event files across runs (so --resume works
# and the logs survive even if the return payload is lost).
runs_volume = modal.Volume.from_name("emojic-runs", create_if_missing=True)


@app.function(gpu=GPU, cpu=4, timeout=60 * 60, volumes={f"{REMOTE}/runs": runs_volume})
def train_remote(resume: bool = False) -> dict[str, bytes]:
    """Run `train.py --no-post` on the GPU; return model.pt + last.ckpt + logs."""
    import glob
    import os

    os.chdir(REMOTE)
    Path("runs").mkdir(exist_ok=True)

    # Confirm the CUDA torch survived into the venv (see module docstring).
    subprocess.run(
        [
            VENV_PY,
            "-c",
            "import torch; print('torch', torch.__version__, "
            "'cuda?', torch.cuda.is_available(), "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
        ],
        check=True,
    )

    tb = subprocess.Popen(
        [
            VENV_TB,
            "--logdir",
            "runs",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--reload_interval",
            "5",
        ]
    )
    try:
        with modal.forward(PORT) as tunnel:
            print(f"\n  TensorBoard (live): {tunnel.url}\n", flush=True)
            cmd = [VENV_PY, "train.py", "--no-post"]
            if resume:
                cmd.append("--resume")
            subprocess.run(cmd, check=True)
    finally:
        tb.terminate()

    runs_volume.commit()

    out: dict[str, bytes] = {"model.pt": Path("model.pt").read_bytes()}
    last = Path("runs/last.ckpt")
    if last.exists():
        out["runs/last.ckpt"] = last.read_bytes()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob("runs/**", recursive=True):
            if os.path.isfile(p) and p != "runs/last.ckpt":
                z.write(p, p)
    out["runs.zip"] = buf.getvalue()
    return out


@app.local_entrypoint()
def main(resume: bool = False):
    art = train_remote.remote(resume=resume)

    Path("model.pt").write_bytes(art["model.pt"])
    if "runs/last.ckpt" in art:
        Path("runs").mkdir(exist_ok=True)
        Path("runs/last.ckpt").write_bytes(art["runs/last.ckpt"])
    with zipfile.ZipFile(io.BytesIO(art["runs.zip"])) as z:
        z.extractall(".")  # -> ./runs/<CONFIG_NAME>/events.out.tfevents.*

    # Post-training runs locally: regenerate docs/ artifacts + report/model/.
    subprocess.run(["uv", "run", "python", "train.py", "--post-only"], check=True)
    print("\nmodel.pt, docs/ artifacts and report/model/ refreshed locally.")
