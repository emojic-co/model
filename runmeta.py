import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from config import CONFIG_PARTS


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _train_sha() -> str | None:
    p = Path("train.jsonl")
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def run_meta() -> dict:
    sha = (
        os.environ.get("EMOJIC_GIT_SHA") or _git("rev-parse", "--short", "HEAD") or "unknown"
    )
    if "EMOJIC_GIT_DIRTY" in os.environ:
        dirty = os.environ["EMOJIC_GIT_DIRTY"] == "1"
    else:
        dirty = bool(_git("status", "--porcelain"))
    return {
        "sha": sha,
        "dirty": dirty,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": list(CONFIG_PARTS),
        "train_sha": _train_sha(),
    }


def save_pt(state_dict: dict, path, **extra) -> None:
    torch.save({"state_dict": state_dict, "meta": {**run_meta(), **extra}}, path)


def load_pt(path, *, map_location: str = "cpu"):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"], blob.get("meta")
    return blob, None


def model_slug(model_meta: dict | None) -> str:
    if model_meta and model_meta.get("sha"):
        return model_meta["sha"]
    return "nometa"
