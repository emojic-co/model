import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

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
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"], blob.get("meta")
    return blob, None


def model_slug(model_meta: dict | None) -> str:
    if model_meta and model_meta.get("sha"):
        return model_meta["sha"]
    return "nometa"


def require_clean_tree() -> None:
    if os.environ.get("EMOJIC_DISPATCH_CHECKED") == "1":
        return
    porcelain = _git("status", "--porcelain")
    if porcelain is None:
        sys.exit("training requires a clean git checkout (git unavailable)")
    if porcelain:
        sys.exit("training aborted: clean git tree required; uncommitted:\n" + porcelain)


def write_meta_yml(out_dir, doc: dict) -> None:
    Path(out_dir, "meta.yml").write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def stamp_lines(model_meta: dict | None, model_name: str, probe_meta: dict) -> list[str]:
    probe = probe_meta["sha"] + (" dirty" if probe_meta["dirty"] else "")
    run_line = f"probe run: {probe_meta['generated']} (code {probe})"
    if not model_meta:
        return [f"model: `{model_name}` — no embedded metadata (legacy .pt)", run_line]
    head = (
        f"model: `{model_name}` — trained {model_meta['sha']} @ "
        f"{model_meta['generated']}  ·  see meta.yml"
    )
    return [head, run_line]
