# Reports Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the report zoo down to `test-emoji` + `test-color`, give every report an on-disk link to the exact model that produced it, and remove the 13 MB of obsolete `report/` history from git.

**Architecture:** A new `runmeta.py` module owns run provenance: it stamps a `meta` dict (git SHA, timestamp, `CONFIG_PARTS`, `train.jsonl` hash, stage) into every `.pt` file at save time, reads it back at load time, hard-blocks training on a dirty git tree, and writes a `meta.yml` next to each report. Reports move to `report/<type>/<timestamp>-<sha>/{meta.yml,report.md,report.json}` and stay git-tracked. Every other report generator (`stat.ts`, `color-analysis.ts`, `fails.ts`, the `data-quality` skill) stops writing files; the preview tools move to a gitignored `preview/`. Finally a one-time `git-filter-repo` run strips historical `report/` blobs.

**Tech Stack:** Python 3.13, PyTorch (CPU wheel), Lightning, PyYAML; Bun + TypeScript for `tools/`; `uv` for Python deps; `git-filter-repo` for the history rewrite.

**Spec:** `docs/superpowers/specs/2026-09-04-reports-reorg-design.md`

## Global Constraints

- **No comments or docstrings** in Python source. Keep `# type: ignore` / `# noqa` / shebangs only.
- **Package management is `uv` only.** Never `pip install`. Add deps with `uv add`.
- **`torch` is pinned to the PyTorch CPU wheel index** (`[tool.uv.sources]` in `pyproject.toml`) — do not disturb that config when running `uv add`.
- **Lint/format gate before every commit:** `uv run ruff check <files you touched>` and `uv run ruff format --check <files you touched>` must pass. The repo baseline does NOT pass `ruff format --check .` repo-wide, so scope the gate to the files your task changed. If a file you touch is not already ruff-formatted, run `uv run ruff format <that file>` — the resulting whole-file reformat hunk is expected; note it in your report as "pre-existing reformat" separate from your task change. Never hand-format.
- **No Python test suite exists.** Follow the repo pattern: script-style checks (`test_emoji.py`, `test_color.py` are plain scripts with `sys.exit`). Unit checks for `runmeta.py` go in a plain-assert script `test_runmeta.py` run with `uv run python test_runmeta.py`, not pytest.
- **`train.jsonl` / `eval.jsonl` / `labels.json` are gitignored** and rebuilt from `data.jsonl` by `bun run regen`. Run `bun run regen` before anything that imports `config` on a fresh checkout.
- **The model stack stays at repo root:** `config.py`, `data.py`, `model.py`, `train.py`, `train_gan.py`, `train-modal.py`, `run.py`, `export_onnx.py`, `test_emoji.py`, `test_color.py`, `loop_emoji.py`. `runmeta.py` joins them.
- **Commit attribution** — end every commit message with:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
  ```
- **A background job auto-commits to `main`.** Stage explicit paths only (`git add <path>`), never `git add -A` / `git add .`.

---

## Prerequisites

- [ ] **P1: Sync the workspace**

```bash
cd /home/gilad/Work/emojic
uv sync
bun install
bun run regen
git status --porcelain   # note anything already dirty (e.g. todo.txt) — leave it alone
```

- [ ] **P2: Confirm a baseline lint pass**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (or note pre-existing failures so you don't blame them on your changes).

---

## Task 1: `config.CONFIG_PARTS`

**Files:**
- Modify: `config.py:88-99` (the `CONFIG_NAME` block)

**Interfaces:**
- Produces: `config.CONFIG_PARTS: list[str]` — exactly `["ENCODER: <enc_str>", "EMOJI: <emj_str>", "STYLE: <style_str>", "GAN: <gan_str>", "TRAIN: <train_str>"]`, the same five substrings `CONFIG_NAME` already embeds.
- Produces: `config.CONFIG_NAME` unchanged in value.

- [ ] **Step 1: Capture the current `CONFIG_NAME` value**

```bash
uv run python -c "import config; print(config.CONFIG_NAME)"
```
Copy the output somewhere; Step 3 must reproduce it byte-for-byte (except the `TIME:` timestamp).

- [ ] **Step 2: Rewrite the block**

In `config.py`, replace:

```python
# TENSORBOARD RUN NAME
CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'ENCODER: {enc_str}',
    f'EMOJI: {emj_str}',
    f'STYLE: {style_str}',
    f'GAN: {gan_str}',
    f'TRAIN: {train_str}',
])
```

with:

```python
# TENSORBOARD RUN NAME
CONFIG_PARTS = [
    f'ENCODER: {enc_str}',
    f'EMOJI: {emj_str}',
    f'STYLE: {style_str}',
    f'GAN: {gan_str}',
    f'TRAIN: {train_str}',
]
CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    *CONFIG_PARTS,
])
```

- [ ] **Step 3: Verify the value is unchanged**

```bash
uv run python -c "import config; print(config.CONFIG_NAME); print(config.CONFIG_PARTS)"
```
Expected: `CONFIG_NAME` matches Step 1 (bar the timestamp); `CONFIG_PARTS` is a 5-element list starting `'ENCODER: 16 3 [80, 120, 180]'`.

- [ ] **Step 4: Lint**

Run: `uv run ruff check config.py && uv run ruff format --check config.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add config.py
git commit -m "$(printf 'Expose CONFIG_PARTS from config.py\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 2: `runmeta.run_meta()`

**Files:**
- Create: `runmeta.py`
- Create: `test_runmeta.py`

**Interfaces:**
- Consumes: `config.CONFIG_PARTS` (Task 1).
- Produces: `runmeta.run_meta() -> dict` with keys `sha: str`, `dirty: bool`, `generated: str` (ISO-8601 seconds), `config: list[str]`, `train_sha: str | None`.
- Produces: private helpers `runmeta._git(*args) -> str | None`, `runmeta._train_sha() -> str | None` (later tasks in this file reuse `_git`).

- [ ] **Step 1: Write the failing test**

Create `test_runmeta.py`:

```python
import os

from runmeta import run_meta


def test_run_meta_shape():
    m = run_meta()
    assert set(m) == {"sha", "dirty", "generated", "config", "train_sha"}
    assert isinstance(m["sha"], str) and m["sha"]
    assert isinstance(m["dirty"], bool)
    assert "T" in m["generated"]
    assert isinstance(m["config"], list) and m["config"][0].startswith("ENCODER: ")
    assert m["train_sha"] is None or (
        isinstance(m["train_sha"], str) and len(m["train_sha"]) == 12
    )


def test_run_meta_sha_env_override():
    os.environ["EMOJIC_GIT_SHA"] = "deadbee"
    os.environ["EMOJIC_GIT_DIRTY"] = "0"
    try:
        m = run_meta()
        assert m["sha"] == "deadbee"
        assert m["dirty"] is False
    finally:
        del os.environ["EMOJIC_GIT_SHA"]
        del os.environ["EMOJIC_GIT_DIRTY"]


if __name__ == "__main__":
    test_run_meta_shape()
    test_run_meta_sha_env_override()
    print("ok")
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run python test_runmeta.py`
Expected: `ModuleNotFoundError: No module named 'runmeta'`.

- [ ] **Step 3: Create `runmeta.py`**

Import only what this task uses. Task 3 adds `import torch`; Task 5 adds `import yaml`.

```python
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
        os.environ.get("EMOJIC_GIT_SHA")
        or _git("rev-parse", "--short", "HEAD")
        or "unknown"
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
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run python test_runmeta.py`
Expected: `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check runmeta.py test_runmeta.py && uv run ruff format --check runmeta.py test_runmeta.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add runmeta.py test_runmeta.py
git commit -m "$(printf 'Add runmeta.run_meta with git SHA + train.jsonl hash\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 3: `runmeta.save_pt` / `load_pt` / `model_slug`

**Files:**
- Modify: `runmeta.py`
- Modify: `test_runmeta.py`

**Interfaces:**
- Consumes: `run_meta()` (Task 2).
- Produces: `runmeta.save_pt(state_dict: dict, path, **extra) -> None` — writes `torch.save({"state_dict": state_dict, "meta": {**run_meta(), **extra}}, path)`.
- Produces: `runmeta.load_pt(path, *, map_location="cpu") -> tuple[dict, dict | None]` — `(state_dict, meta)` for the new format, `(state_dict, None)` for a bare legacy `state_dict`.
- Produces: `runmeta.model_slug(model_meta: dict | None) -> str` — `model_meta["sha"]` when present and truthy, else `"nometa"`.

- [ ] **Step 1: Add failing tests**

Append to `test_runmeta.py` (and add the new calls to `__main__`):

```python
def test_save_load_round_trip(tmp_path="/tmp/runmeta-rt.pt"):
    import torch

    from runmeta import load_pt, save_pt

    save_pt({"w": torch.zeros(1)}, tmp_path, stage="task")
    sd, meta = load_pt(tmp_path)
    assert list(sd) == ["w"]
    assert meta["stage"] == "task"
    assert meta["sha"]


def test_load_pt_legacy_bare(tmp_path="/tmp/runmeta-legacy.pt"):
    import torch

    from runmeta import load_pt

    torch.save({"w": torch.zeros(1)}, tmp_path)
    sd, meta = load_pt(tmp_path)
    assert list(sd) == ["w"]
    assert meta is None


def test_model_slug():
    from runmeta import model_slug

    assert model_slug({"sha": "abc1234"}) == "abc1234"
    assert model_slug(None) == "nometa"
    assert model_slug({}) == "nometa"
    assert model_slug({"sha": ""}) == "nometa"
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run python test_runmeta.py`
Expected: `ImportError: cannot import name 'save_pt'` (and `model_slug`).

- [ ] **Step 3: Add `import torch` and the functions to `runmeta.py`**

Add `import torch` to the import block (after `from pathlib import Path`, blank line, then `import torch`, then blank line, then `from config import CONFIG_PARTS`). Add:

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run python test_runmeta.py`
Expected: `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check runmeta.py test_runmeta.py && uv run ruff format --check runmeta.py test_runmeta.py`

- [ ] **Step 6: Commit**

```bash
git add runmeta.py test_runmeta.py
git commit -m "$(printf 'Add runmeta.save_pt / load_pt / model_slug\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 4: `runmeta.require_clean_tree()`

**Files:**
- Modify: `runmeta.py`
- Modify: `test_runmeta.py`

**Interfaces:**
- Consumes: `_git` (Task 2).
- Produces: `runmeta.require_clean_tree() -> None` — returns silently when `os.environ.get("EMOJIC_DISPATCH_CHECKED") == "1"`, or when `git status --porcelain` is empty; calls `sys.exit(<str>)` when the tree is dirty (message includes the porcelain listing) or when git is unavailable.

- [ ] **Step 1: Add failing tests**

Append to `test_runmeta.py` + `__main__`:

```python
def test_require_clean_tree_dispatch_skip():
    import os

    from runmeta import require_clean_tree

    os.environ["EMOJIC_DISPATCH_CHECKED"] = "1"
    try:
        require_clean_tree()
    finally:
        del os.environ["EMOJIC_DISPATCH_CHECKED"]


def test_require_clean_tree_dirty_exits(tmp_path="/tmp/runmeta-gitdirty"):
    import os
    import subprocess

    import runmeta

    subprocess.run(["rm", "-rf", tmp_path], check=True)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    open(f"{tmp_path}/x.txt", "w").write("hi")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        raised = False
        try:
            runmeta.require_clean_tree()
        except SystemExit as e:
            raised = True
            assert "clean git tree" in str(e)
        assert raised
    finally:
        os.chdir(cwd)
```

(The subprocess approach fails here: `import runmeta` imports `config`, which reads `labels.json` by relative path — absent in the temp dir. `os.chdir` after the top-level import sidesteps that; `runmeta` is already imported from the repo root.)

- [ ] **Step 2: Run, verify failure**

Run: `uv run python test_runmeta.py`
Expected: `ImportError: cannot import name 'require_clean_tree'`.

- [ ] **Step 3: Implement**

Add to `runmeta.py`:

```python
def require_clean_tree() -> None:
    if os.environ.get("EMOJIC_DISPATCH_CHECKED") == "1":
        return
    porcelain = _git("status", "--porcelain")
    if porcelain is None:
        sys.exit("training requires a clean git checkout (git unavailable)")
    if porcelain:
        sys.exit("training aborted: clean git tree required; uncommitted:\n" + porcelain)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run python test_runmeta.py`
Expected: `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check runmeta.py test_runmeta.py && uv run ruff format --check runmeta.py test_runmeta.py`

- [ ] **Step 6: Commit**

```bash
git add runmeta.py test_runmeta.py
git commit -m "$(printf 'Add runmeta.require_clean_tree hard gate\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 5: `runmeta.write_meta_yml` / `stamp_lines` + PyYAML dep

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `runmeta.py`
- Modify: `test_runmeta.py`

**Interfaces:**
- Produces: `runmeta.write_meta_yml(out_dir, doc: dict) -> None` — writes `Path(out_dir, "meta.yml")` via `yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)`.
- Produces: `runmeta.stamp_lines(model_meta: dict | None, model_name: str, probe_meta: dict) -> list[str]` — plain `"key: value"` strings, no markdown/HTML prefix:
  - with meta: `["model: `<name>` — trained <sha> @ <generated>  ·  see meta.yml", "probe run: <probe.generated> (code <probe.sha>[ dirty])"]`
  - without meta: `["model: `<name>` — no embedded metadata (legacy .pt)", "probe run: <probe.generated> (code <probe.sha>[ dirty])"]`

- [ ] **Step 1: Add the dependency**

```bash
uv add pyyaml
```
Verify `pyproject.toml` gained `pyyaml` under `[project] dependencies` and `[tool.uv.sources]` for `torch` is untouched.

- [ ] **Step 2: Add failing tests**

Append to `test_runmeta.py` + `__main__`:

```python
def test_write_meta_yml(tmp_dir="/tmp/runmeta-yml"):
    import os

    import yaml

    from runmeta import write_meta_yml

    os.makedirs(tmp_dir, exist_ok=True)
    doc = {"report_type": "test-emoji", "models": {"enc.pt": {"sha": "abc"}}}
    write_meta_yml(tmp_dir, doc)
    back = yaml.safe_load(open(f"{tmp_dir}/meta.yml"))
    assert back["report_type"] == "test-emoji"
    assert back["models"]["enc.pt"]["sha"] == "abc"


def test_stamp_lines():
    from runmeta import stamp_lines

    probe = {"sha": "cafe", "dirty": True, "generated": "2026-09-04T05:48:00"}
    with_meta = stamp_lines(
        {"sha": "beef", "generated": "2026-09-04T05:12:00"}, "enc.pt", probe
    )
    assert "trained beef" in with_meta[0]
    assert "see meta.yml" in with_meta[0]
    assert "code cafe dirty" in with_meta[1]
    legacy = stamp_lines(None, "enc.pt", probe)
    assert "no embedded metadata" in legacy[0]
```

- [ ] **Step 3: Run, verify failure**

Run: `uv run python test_runmeta.py`
Expected: `ImportError: cannot import name 'stamp_lines'`.

- [ ] **Step 4: Add `import yaml`, `write_meta_yml`, and `stamp_lines`**

Add `import yaml` next to `import torch` in the import block. Add both functions:

```python
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
```

- [ ] **Step 5: Run, verify pass**

Run: `uv run python test_runmeta.py`
Expected: `ok`.

- [ ] **Step 6: Lint**

Run: `uv run ruff check runmeta.py test_runmeta.py && uv run ruff format --check runmeta.py test_runmeta.py`

- [ ] **Step 7: Commit**

```bash
git add runmeta.py test_runmeta.py pyproject.toml uv.lock
git commit -m "$(printf 'Add runmeta.write_meta_yml / stamp_lines; add pyyaml dep\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 6: Save sites use `save_pt`

**Files:**
- Modify: `train.py:324-329` (task-stage save loop), `train.py:365-370` (gan-stage save loop)
- Modify: `train_gan.py:72-77` (gan-stage save loop)

**Interfaces:**
- Consumes: `runmeta.save_pt` (Task 3).
- Produces: `enc.pt` / `style.pt` / `emoji.pt` carry `meta.stage == "task"`; `gen.pt` / `tst.pt` carry `meta.stage == "gan"`.

- [ ] **Step 1: `train.py` — import**

`train.py` currently has no `runmeta` import; add one after the `from config import (...)` block. Import only what this task uses (Task 7 extends this line to add `require_clean_tree`, in the same commit that uses it — keeps every commit F401-clean):

```python
from runmeta import save_pt
```

- [ ] **Step 2: `train.py` — task-stage save loop**

Replace:

```python
    for name, mod in (
        ("enc", task.enc),
        ("style", task.style),
        ("emoji", task.emoji),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
```

with:

```python
    for name, mod in (
        ("enc", task.enc),
        ("style", task.style),
        ("emoji", task.emoji),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="task")
```

- [ ] **Step 3: `train.py` — gan-stage save loop**

Replace:

```python
    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
```

with:

```python
    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="gan")
```

- [ ] **Step 4: `train_gan.py` — import + save loop**

Add to `train_gan.py` imports (only `save_pt` — Task 7 adds `require_clean_tree`, Task 8 adds `load_pt`, each in the commit that uses it):

```python
from runmeta import save_pt
```

Replace:

```python
    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
```

with:

```python
    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="gan")
```

- [ ] **Step 5: Verify imports resolve**

Run: `uv run python -c "import train, train_gan; print('ok')"`
Expected: `ok` (Lightning import warnings are fine). If it fails on `train.jsonl`, run `bun run regen` first.

- [ ] **Step 6: Lint**

Run: `uv run ruff check train.py train_gan.py && uv run ruff format --check train.py train_gan.py`

- [ ] **Step 7: Commit**

```bash
git add train.py train_gan.py
git commit -m "$(printf 'Save .pt files with embedded runmeta via save_pt\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 7: Clean-tree gate at training entrypoints + Modal env passthrough

**Files:**
- Modify: `train.py` (start of `__main__`)
- Modify: `train_gan.py` (start of `__main__`, currently `pl.seed_everything(SEED, workers=True)`)
- Modify: `train-modal.py` — `main()` (`@app.local_entrypoint`) and `train_remote()` (`@app.function`)

**Interfaces:**
- Consumes: `runmeta.require_clean_tree` (Task 4).
- Produces: `train.py` / `train_gan.py` / `modal run train-modal.py` all `sys.exit` on a dirty tree.
- Produces: `train_remote(threads, git_sha)` — new `git_sha` kwarg; the box env dict gets `EMOJIC_GIT_SHA = git_sha` and `EMOJIC_DISPATCH_CHECKED = "1"` before the `train.py` / `test_emoji.py` subprocesses run.

**Ruling (plan defect corrected):** the plan originally assumed `os.environ` set in `main()` propagates to the Modal box via `_run_env()`'s `**os.environ`. It does not — `_run_env` reads the box's environ. Corrected to pass `git_sha` as a `train_remote` function argument and inject it into the `env` dict there. `EMOJIC_GIT_DIRTY` is not passed at all (a dirty dispatch tree fails `require_clean_tree()` before `fn.remote()` is called, so the box never needs it).

- [ ] **Step 1: `train.py` gate**

Extend the `runmeta` import line (Task 6 made it `from runmeta import save_pt`):

```python
from runmeta import require_clean_tree, save_pt
```

In `train.py`, the `__main__` block starts:

```python
if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False
```

Insert `require_clean_tree()` as the first line inside `__main__`:

```python
if __name__ == "__main__":
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False
```

- [ ] **Step 2: `train_gan.py` gate**

Extend its `runmeta` import line to `from runmeta import require_clean_tree, save_pt`. `train_gan.py`'s `__main__` starts with `pl.seed_everything(SEED, workers=True)`. Insert `require_clean_tree()` immediately before it:

```python
if __name__ == "__main__":
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
```

- [ ] **Step 3: `train-modal.py` — confirm the current code matches**

```bash
grep -n "def main(\|def train_remote(\|def _run_env(\|env = _run_env\|fn.remote(\|import subprocess\|if fetch_only" train-modal.py
```
Confirm: `train_remote` currently `def train_remote(threads: int = CPU) -> dict[str, int]:` with `env = _run_env(threads)` as its first body line; `main` currently `def main(cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False):` with an `if fetch_only:` early-return then `fn.remote(threads=cpu)`; `import subprocess` present at module top. If any differ from Step 4's "current" snippet, adapt the edit to the real text and note it in the report.

- [ ] **Step 4: `train-modal.py` — dispatch-side gate + pass the SHA as a function argument**

The current `main()` is:

```python
@app.local_entrypoint()
def main(cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False):
    if fetch_only:
        _retrieve_and_cleanup()
        return
    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB:
        fn = train_remote.with_options(cpu=cpu, memory=memory)
    try:
        print(fn.remote(threads=cpu))
    finally:
        _retrieve_and_cleanup()
```

`main()` runs on the dispatch machine (has git); `train_remote` runs on the Modal
box (no git, and the box's `os.environ` is NOT the dispatch machine's — env does
not auto-propagate). So the SHA must travel as a **function argument**.

Change `main()` to:

```python
@app.local_entrypoint()
def main(cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False):
    if fetch_only:
        _retrieve_and_cleanup()
        return
    from runmeta import require_clean_tree

    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB:
        fn = train_remote.with_options(cpu=cpu, memory=memory)
    try:
        print(fn.remote(threads=cpu, git_sha=git_sha))
    finally:
        _retrieve_and_cleanup()
```

Change `train_remote`'s signature and inject the vars into the env dict it
already builds:

```python
def train_remote(threads: int = CPU, git_sha: str = "") -> dict[str, int]:
    env = _run_env(threads)
    env["EMOJIC_GIT_SHA"] = git_sha
    env["EMOJIC_DISPATCH_CHECKED"] = "1"
    ...
```

`subprocess` is already imported in `train-modal.py`. `require_clean_tree` is
imported locally inside `main()` (which only runs on the dispatch machine) to
avoid importing `runmeta`/`torch` during the Modal app's box-side module load.
`_run_env()` itself is unchanged.

- [ ] **Step 5: Verify the gate fires**

```bash
echo "scratch" > _dirty_probe.txt
uv run python train.py; echo "exit=$?"
rm _dirty_probe.txt
```
Expected: prints `training aborted: clean git tree required; uncommitted:` followed by `?? _dirty_probe.txt`, `exit=1`, and Lightning never starts. Do the same for `uv run python train_gan.py` (same message, `exit=1`).

- [ ] **Step 6: Verify the dispatch-check skip + `train-modal.py` parses**

```bash
echo "scratch" > _dirty_probe.txt
EMOJIC_DISPATCH_CHECKED=1 uv run python -c "from runmeta import require_clean_tree; require_clean_tree(); print('skipped ok')"
rm _dirty_probe.txt
uv run python -m py_compile train-modal.py && echo "train-modal.py compiles"
```
Expected: `skipped ok` then `train-modal.py compiles`. (Modal itself is not run here.)

- [ ] **Step 7: Lint**

Run: `uv run ruff check train.py train_gan.py train-modal.py && uv run ruff format --check train.py train_gan.py train-modal.py`

`train-modal.py` passes `ruff check` but FAILS `ruff format --check` at baseline (pre-existing). Run `uv run ruff format train-modal.py` to satisfy the gate and note the resulting whole-file reformat separately in the report (as in Task 6). `train.py` / `train_gan.py` are already ruff-formatted after Task 6, so their diffs here are just the gate line + import.

- [ ] **Step 8: Commit**

```bash
git add train.py train_gan.py train-modal.py
git commit -m "$(printf 'Hard-block training on a dirty git tree; pass SHA to Modal\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 8: Load sites use `load_pt` and stash `_pt_meta`

**Files:**
- Modify: `test_emoji.py:20-23` (`_load`)
- Modify: `test_color.py:40-43` (`_load`)
- Modify: `export_onnx.py:28-31` (`_load`)
- Modify: `run.py:26-29` (`_load`)
- Modify: `train_gan.py:29-31` (`load`)

**Interfaces:**
- Consumes: `runmeta.load_pt` (Task 3).
- Produces: every `_load(mod, path)` / `load(mod, path)` sets `mod._pt_meta` to the loaded `meta` dict or `None`, and still returns `mod`. Behaviour with a bare legacy `.pt` is unchanged (loads fine, `_pt_meta is None`).

Import only `load_pt` in this task. Tasks 9 and 10 extend the `test_emoji.py` /
`test_color.py` import lines to add `model_slug`, `run_meta`, `stamp_lines`,
`write_meta_yml` in the same commit that uses them — this keeps every commit
F401-clean.

- [ ] **Step 1: `test_emoji.py`**

Add import after the `from data import ...` block:

```python
from runmeta import load_pt
```

Replace:

```python
def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    mod.eval()
    return mod
```

with:

```python
def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    mod.eval()
    return mod
```

- [ ] **Step 2: `test_color.py`** — same `_load` edit, plus `from runmeta import load_pt`.

- [ ] **Step 3: `export_onnx.py`** — same edit to its `_load` (note it uses `nn.Module`, not `torch.nn.Module` — keep the existing annotation), plus import:

```python
from runmeta import load_pt
```

Body:

```python
def _load(mod: nn.Module, path: str) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    mod.eval()
    return mod
```

- [ ] **Step 4: `run.py`** — same edit, plus `from runmeta import load_pt`.

- [ ] **Step 5: `train_gan.py`** — its helper is named `load`:

```python
def load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    return mod
```

Add `load_pt` to the existing `from runmeta import ...` line (after Task 7 it reads `require_clean_tree, save_pt`):

```python
from runmeta import load_pt, require_clean_tree, save_pt
```

- [ ] **Step 6: Verify imports + a real load**

```bash
uv run python -c "import test_emoji, test_color, export_onnx, run, train_gan; print('imports ok')"
```
Expected: `imports ok`.

If `enc.pt` exists locally:
```bash
uv run python -c "
from model import TextEncoder
from test_emoji import _load
m = _load(TextEncoder(), 'enc.pt')
print('meta:', getattr(m, '_pt_meta', 'MISSING'))
"
```
Expected: `meta: None` (existing `.pt` is legacy) or a dict — never `MISSING`.

- [ ] **Step 7: Lint**

Run: `uv run ruff check test_emoji.py test_color.py export_onnx.py run.py train_gan.py && uv run ruff format --check test_emoji.py test_color.py export_onnx.py run.py train_gan.py`

Only `load_pt` is imported here, so there is no unused-import warning. Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add test_emoji.py test_color.py export_onnx.py run.py train_gan.py
git commit -m "$(printf 'Load .pt via runmeta.load_pt, stash _pt_meta on the module\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 9: `test_emoji.py` — folder layout + `meta.yml` + header

**Files:**
- Modify: `test_emoji.py` — import line, `_evaluate` return, `_render` signature + header lines, `test_emoji()` write block.

**Interfaces:**
- Consumes: `load_pt`, `model_slug`, `run_meta`, `stamp_lines`, `write_meta_yml`.
- Produces: `test_emoji(write_report=True)` writes `report/test-emoji/<stamp>-<enc_sha>/` containing `report.md`, `report.json`, `meta.yml`. `stamp` is `%y-%m-%d-%H-%M`; `<enc_sha>` is `model_slug(enc._pt_meta)`.
- Produces: `_evaluate(...) -> tuple[list[dict], dict[str, dict | None]]` where the second element maps `enc_path` / `emoji_path` to their `_pt_meta`.

- [ ] **Step 1: Extend the import** (from Task 8 Step 7 choice (b), the line is `from runmeta import load_pt`)

```python
from runmeta import load_pt, model_slug, run_meta, stamp_lines, write_meta_yml
```

- [ ] **Step 2: `_evaluate` returns metas**

Current head:

```python
def _evaluate(
    enc_path: str, emoji_path: str, words: list[dict], counts: Counter
) -> list[dict]:
    enc = _load(TextEncoder(), enc_path)
    head = _load(EmojiHead(), emoji_path)
```

Change the annotation and the `return`:

```python
def _evaluate(
    enc_path: str, emoji_path: str, words: list[dict], counts: Counter
) -> tuple[list[dict], dict]:
    enc = _load(TextEncoder(), enc_path)
    head = _load(EmojiHead(), emoji_path)
```

Find the single `return <results>` at the end of `_evaluate` and change it to:

```python
    return results, {enc_path: enc._pt_meta, emoji_path: head._pt_meta}
```

(If the local variable is not named `results`, rename in the `return` accordingly — check with `sed -n '34,90p' test_emoji.py`.)

- [ ] **Step 3: `_render` — new signature + header**

NOTE: `_render`'s signature was collapsed to one line by a prior `ruff format`.
The current code is:

```python
def _render(results: list[dict], enc_path: str, emoji_path: str, stamp: str) -> str:
    scored_n, acc, mrr = _acc(results)
    lines = [
        f"# Emoji test - {stamp}",
        "",
        f"- model: `{enc_path}` + `{emoji_path}`",
        f"- words: {len(results)} ({scored_n} scored)",
        f"- MRR: {mrr:.3f}",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
```

Replace that with (let `ruff format` re-wrap the signature after):

```python
def _render(
    results: list[dict],
    enc_path: str,
    emoji_path: str,
    stamp: str,
    model_meta: dict | None,
    probe_meta: dict,
) -> str:
    scored_n, acc, mrr = _acc(results)
    lines = [f"# Emoji test - {stamp}", ""]
    lines += [f"- {ln}" for ln in stamp_lines(model_meta, enc_path, probe_meta)]
    if model_meta is None:
        lines.append(f"  - config: {' | '.join(probe_meta['config'])}")
    lines += [
        f"- words: {len(results)} ({scored_n} scored)",
        f"- MRR: {mrr:.3f}",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
```

- [ ] **Step 4: `test_emoji()` — write block**

Replace the whole body from `results = _evaluate(...)` through the end of the `if write_report:` block:

```python
    words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    results = _evaluate(enc_path, emoji_path, words, _emoji_counts())
    scored_n, acc, mrr = _acc(results)
    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{stamp}.md"
        path.write_text(_render(results, enc_path, emoji_path, stamp), encoding="utf-8")
        json_path = REPORT_DIR / f"{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    "stamp": stamp,
                    "enc": enc_path,
                    "emoji": emoji_path,
                    "summary": {"acc": acc, "mrr": mrr, "n": scored_n},
                    "words": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {path}")
        print(f"wrote {json_path}")
```

with:

```python
    words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    results, metas = _evaluate(enc_path, emoji_path, words, _emoji_counts())
    scored_n, acc, mrr = _acc(results)
    probe_meta = run_meta()
    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    enc_meta = metas.get(enc_path)
    emoji_meta = metas.get(emoji_path)

    if write_report:
        out_dir = REPORT_DIR / f"{stamp}-{model_slug(enc_meta)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "report.md").write_text(
            _render(results, enc_path, emoji_path, stamp, enc_meta, probe_meta),
            encoding="utf-8",
        )
        (out_dir / "report.json").write_text(
            json.dumps(
                {
                    "stamp": stamp,
                    "enc": enc_path,
                    "emoji": emoji_path,
                    "summary": {"acc": acc, "mrr": mrr, "n": scored_n},
                    "words": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        warnings = []
        if (
            enc_meta
            and emoji_meta
            and enc_meta.get("sha") != emoji_meta.get("sha")
        ):
            warnings.append(
                f"{enc_path} and {emoji_path} were saved from different commits"
            )
        doc = {
            "report_type": "test-emoji",
            "generated": probe_meta["generated"],
            "probe_commit": probe_meta["sha"],
            "probe_dirty": probe_meta["dirty"],
        }
        if warnings:
            doc["warnings"] = warnings
        doc["models"] = {enc_path: enc_meta, emoji_path: emoji_meta}
        doc["summary"] = {
            **{f"acc@{k}": acc[k] for k in TOP_K},
            "mrr": mrr,
            "n": scored_n,
        }
        write_meta_yml(out_dir, doc)
        print(f"wrote {out_dir}/")
```

- [ ] **Step 5: Run the probe (needs `enc.pt` + `emoji.pt`)**

```bash
bun run regen   # if not already fresh
uv run python test_emoji.py
```
Expected: `wrote report/test-emoji/<stamp>-<sha_or_nometa>/` then the `emoji test acc@1=... mrr=...` line.

Then:
```bash
D=$(ls -td report/test-emoji/*/ | head -1); ls "$D"
uv run python -c "import yaml; d=yaml.safe_load(open('${D}meta.yml')); print(list(d)); print(list(d['models'])); print(d['summary'])"
head -6 "${D}report.md"
```
Expected: `meta.yml`, `report.md`, `report.json` present; `meta.yml` top keys include `report_type`, `models`, `summary`; header shows the two `stamp_lines` bullets. If `enc.pt` is legacy, folder ends `-nometa` and the header shows the `config:` sub-bullet.

If no `.pt` files exist locally, skip the run and note that Task 15's `train-modal` path will exercise it; still do Step 6–7.

- [ ] **Step 6: Lint**

Run: `uv run ruff check test_emoji.py && uv run ruff format --check test_emoji.py`

- [ ] **Step 7: Commit**

```bash
git add test_emoji.py
git commit -m "$(printf 'test_emoji: per-run report folder with meta.yml\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 10: `test_color.py` — folder layout + `meta.yml` + header

**Files:**
- Modify: `test_color.py` — import line, `_evaluate` return, `_render_html` header, `test_color()` write block.

**Interfaces:**
- Consumes: `load_pt`, `model_slug`, `run_meta`, `stamp_lines`, `write_meta_yml`.
- Produces: `test_color(write_report=True)` writes `report/test-color/<stamp>-<enc_sha>/` containing `report.html`, `report.json`, `meta.yml`.
- Produces: `_evaluate(...) -> tuple[list[dict], dict[str, dict | None]]` mapping `enc_path` / `gen_path` to `_pt_meta`.

- [ ] **Step 1: Extend the import**

```python
from runmeta import load_pt, model_slug, run_meta, stamp_lines, write_meta_yml
```

- [ ] **Step 2: `_evaluate` returns metas**

```bash
sed -n '92,150p' test_color.py
```
Head is:

```python
def _evaluate(enc_path: str, gen_path: str, groups: dict[str, list]) -> list[dict]:
    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)
```

Change annotation to `-> tuple[list[dict], dict]:` and its final `return <groups_result>` to:

```python
    return groups_result, {enc_path: enc._pt_meta, gen_path: gen._pt_meta}
```

(Use the real local name from the `sed` output for the returned list.)

- [ ] **Step 3: `_render_html` — header**

Replace:

```python
def _render_html(groups: list[dict], summary: dict, meta: dict) -> str:
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Color test - {meta['stamp']}</title>",
```

Keep the signature; the extra fields ride on `meta`. Replace the `<p>model: ...` line:

```python
        f"<p>model: <code>{meta['enc']}</code> + <code>{meta['gen']}</code><br>",
```

with:

```python
        "<p>"
        + "<br>".join(html.escape(ln) for ln in meta["stamp_lines"])
        + "<br>",
```

(`html` is already imported in `test_color.py`.)

- [ ] **Step 4: `test_color()` — write block**

Current:

```python
    groups = _evaluate(enc_path, gen_path, _keyword_groups(train_path))
    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    ...
    meta = {"stamp": stamp, "enc": enc_path, "gen": gen_path}

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = REPORT_DIR / f"{stamp}.html"
        html_path.write_text(_render_html(groups, summary, meta), encoding="utf-8")
        json_path = REPORT_DIR / f"{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    **meta,
                    "summary": summary,
                    "keywords": [ ... ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {html_path}")
        print(f"wrote {json_path}")
```

Replace with (keep the `"keywords": [...]` list-comp exactly as it is now):

```python
    groups, metas = _evaluate(enc_path, gen_path, _keyword_groups(train_path))
    probe_meta = run_meta()
    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    enc_meta = metas.get(enc_path)
    gen_meta = metas.get(gen_path)
    ...
    meta = {
        "stamp": stamp,
        "enc": enc_path,
        "gen": gen_path,
        "stamp_lines": stamp_lines(enc_meta, enc_path, probe_meta),
    }

    if write_report:
        out_dir = REPORT_DIR / f"{stamp}-{model_slug(enc_meta)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.html").write_text(
            _render_html(groups, summary, meta), encoding="utf-8"
        )
        (out_dir / "report.json").write_text(
            json.dumps(
                {
                    "stamp": stamp,
                    "enc": enc_path,
                    "gen": gen_path,
                    "summary": summary,
                    "keywords": [ ... ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        warnings = []
        if enc_meta and gen_meta and enc_meta.get("sha") != gen_meta.get("sha"):
            warnings.append(
                f"{enc_path} and {gen_path} were saved from different commits"
            )
        doc = {
            "report_type": "test-color",
            "generated": probe_meta["generated"],
            "probe_commit": probe_meta["sha"],
            "probe_dirty": probe_meta["dirty"],
        }
        if warnings:
            doc["warnings"] = warnings
        doc["models"] = {enc_path: enc_meta, gen_path: gen_meta}
        doc["summary"] = summary
        write_meta_yml(out_dir, doc)
        print(f"wrote {out_dir}/")
```

Leave the `"keywords": [ ... ]` comprehension body verbatim from the current file — do not paraphrase it.

- [ ] **Step 5: Run the probe (needs `enc.pt` + `gen.pt` + fresh `train.jsonl`)**

```bash
bun run regen
uv run python test_color.py
D=$(ls -td report/test-color/*/ | head -1); ls "$D"
uv run python -c "import yaml; d=yaml.safe_load(open('${D}meta.yml')); print(list(d)); print(list(d['models']))"
```
Expected: `report.html`, `report.json`, `meta.yml`; `models` has `enc.pt` + `gen.pt`. Skip the run if the `.pt` files are absent locally.

- [ ] **Step 6: Lint**

Run: `uv run ruff check test_color.py && uv run ruff format --check test_color.py`

- [ ] **Step 7: Commit**

```bash
git add test_color.py
git commit -m "$(printf 'test_color: per-run report folder with meta.yml\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 11: `export_onnx.py` — `model_meta` in `web/public/meta.json`

**Files:**
- Modify: `export_onnx.py` — the `meta = {...}` dict and its consumers (`export()` loads `enc` before `meta` is built? check order).

**Interfaces:**
- Consumes: `enc._pt_meta` (set by the Task 8 `_load`).
- Produces: `web/public/meta.json` gains `"model_meta": <dict | null>` (the encoder's embedded meta).

- [ ] **Step 1: Inspect `export()` ordering**

```bash
sed -n '80,130p' export_onnx.py
```
Confirm whether the `meta = {...}` dict is built inside `export()` after `enc = _load(...)` or in a helper that doesn't see `enc`. Adjust Step 2 to pass `enc._pt_meta` into wherever `meta` is assembled.

- [ ] **Step 2: Add the key**

In the `meta = { ... }` dict (currently ending with `"exported_at": datetime.now(UTC).isoformat(timespec="minutes"),`), add:

```python
        "model_meta": getattr(enc, "_pt_meta", None),
```

If `meta` is built in a function without `enc` in scope, thread `model_meta` in as a parameter from `export()`:
```python
    meta_json(enc._pt_meta)          # or however the write is invoked
```
and add `model_meta` to that function's signature + dict.

- [ ] **Step 3: Verify (needs the four `.pt` files)**

```bash
bun run regen
uv run python export_onnx.py
uv run python -c "import json; d=json.load(open('web/public/meta.json')); print('model_meta' in d, type(d['model_meta']).__name__)"
```
Expected: `True dict` (or `True NoneType` for legacy `.pt`). Skip if `.pt` files absent; note it runs in Task 15's follow-up.

- [ ] **Step 4: Lint**

Run: `uv run ruff check export_onnx.py && uv run ruff format --check export_onnx.py`

- [ ] **Step 5: Commit**

```bash
git add export_onnx.py web/public/meta.json
git commit -m "$(printf 'export_onnx: record encoder model_meta in web/public/meta.json\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

(Only stage `web/public/meta.json` if you actually re-ran `export_onnx.py`; otherwise stage `export_onnx.py` alone.)

---

## Task 12: `stat.ts` → stdout; remove the PostToolUse hook

**Files:**
- Modify: `tools/data/stat.ts:1-2, 9, 195-200`
- Modify: `.claude/settings.json`

**Interfaces:**
- Produces: `bun run tools/data/stat.ts` prints the full report to stdout and writes no file.

- [ ] **Step 1: Inspect the tail of `stat.ts`**

```bash
sed -n '1,12p;190,205p' tools/data/stat.ts
```

- [ ] **Step 2: Edit `stat.ts`**

- Remove `mkdir` / `writeFile` from the `node:fs/promises` import on line 2. If nothing else from that module is used, delete the whole import line.
- Delete the `const REPORT_DIR = "report/data-stat"` line (line 9).
- Replace the write tail (currently roughly):

```typescript
  await mkdir(REPORT_DIR, { recursive: true })
  const dest = `${REPORT_DIR}/${file}.md`
  await writeFile(dest, doc.join("\n"))
  console.log(dest)
```

with:

```typescript
  console.log(doc.join("\n"))
```

If `file` (the timestamp var) is now unused, delete its declaration too (check `grep -n "file" tools/data/stat.ts`).

- [ ] **Step 3: Edit `.claude/settings.json`**

Current content:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path // .tool_response.filePath // empty' | { read -r f; case \"$(basename \"$f\")\" in data.jsonl) cd \"$CLAUDE_PROJECT_DIR\" && bun run tools/data/stat.ts ;; esac; } 2>/dev/null || true",
            "statusMessage": "Refreshing data stats report"
          }
        ]
      }
    ]
  }
}
```

Replace the entire file with:

```json
{}
```

- [ ] **Step 4: Verify**

```bash
bun run regen
bun run tools/data/stat.ts | head -5
ls report/data-stat 2>&1
git status --porcelain report/
```
Expected: report text on stdout; `ls` reports `report/data-stat` does not exist (or is unchanged from before if it lingers as an untracked leftover — it should not be freshly created); `git status` shows nothing new under `report/`.

- [ ] **Step 5: Commit**

```bash
git add tools/data/stat.ts .claude/settings.json
git commit -m "$(printf 'stat.ts prints to stdout; drop the data.jsonl PostToolUse hook\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 13: `color-analysis.ts` + `fails.ts` → stdout

**Files:**
- Modify: `tools/data/color-analysis.ts:1, 7, ~end`
- Modify: `tools/analysis/fails.ts:1, 9, ~100-105`

**Interfaces:**
- Produces: `bun run color-analysis` and `bun run tools/analysis/fails.ts` print their `.md` body to stdout and write no file.

- [ ] **Step 1: Inspect both tails**

```bash
grep -n "mkdir\|writeFile\|REPORT_DIR\|OUT_DIR\|console.log" tools/data/color-analysis.ts tools/analysis/fails.ts
sed -n '95,110p' tools/analysis/fails.ts
```

- [ ] **Step 2: `color-analysis.ts`**

- Trim `mkdir` / `writeFile` from the `node:fs/promises` import (keep `readFile` if used).
- Delete `const REPORT_DIR = "report/color-analysis"`.
- Replace the write tail (`await mkdir(...); const dest = ...; await writeFile(dest, ...); console.log(dest)`) with `console.log(<the doc string>)` — use the same variable that was passed to `writeFile`.

- [ ] **Step 3: `fails.ts`**

Same treatment: trim the import, delete `const OUT_DIR = "report/fails"`, replace:

```typescript
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${file}.md`
  await writeFile(dest, doc.join("\n"))
  console.log(dest)
```

with:

```typescript
  console.log(doc.join("\n"))
```

Remove `file` if now unused.

- [ ] **Step 4: Verify**

```bash
bun run regen
bun run color-analysis | head -5
bun run tools/analysis/fails.ts | head -5
git status --porcelain report/
```
Expected: text on stdout for each; nothing new under `report/`. (If `fails.ts` needs a `pred.jsonl` or other input it doesn't have, a clean "missing input" error is acceptable — the point is it no longer writes a report file. Note it in the commit if so.)

- [ ] **Step 5: Commit**

```bash
git add tools/data/color-analysis.ts tools/analysis/fails.ts
git commit -m "$(printf 'color-analysis.ts and fails.ts print to stdout\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 14: Preview tools → gitignored `preview/`

**Files:**
- Modify: `tools/data/preview.ts:6`, `tools/data/preview-pred.ts:8`, `tools/data/preview-labels.ts:7`, `tools/data/preview-model.ts:10`
- Modify: `.gitignore`

**Interfaces:**
- Produces: each preview tool writes under `preview/` (gitignored); `report/preview*` are never recreated.

- [ ] **Step 1: Change the four `OUT_DIR` constants**

| file | from | to |
| --- | --- | --- |
| `tools/data/preview.ts` | `const OUT_DIR = "report/preview"` | `const OUT_DIR = "preview"` |
| `tools/data/preview-pred.ts` | `const OUT_DIR = "report/preview"` | `const OUT_DIR = "preview"` |
| `tools/data/preview-labels.ts` | `const OUT_DIR = "report/preview-labels"` | `const OUT_DIR = "preview/labels"` |
| `tools/data/preview-model.ts` | `const OUT_DIR = "report/preview-model"` | `const OUT_DIR = "preview/model"` |

Leave the `mkdir(OUT_DIR, { recursive: true })` calls — they now create `preview/…`.

- [ ] **Step 2: `.gitignore`**

Append at the end:

```
# Browser-only preview pages, regenerated on demand.
preview/
```

Do **not** add `report/` — report folders stay tracked.

- [ ] **Step 3: Verify**

```bash
bun run regen
bun run tools/data/preview.ts | tail -1
bun run tools/data/preview-labels.ts | tail -1
ls preview preview/labels
git check-ignore preview/ ; echo "report ignored? $(git check-ignore report/ || echo no)"
git status --porcelain | grep -E 'preview|report' || echo "clean"
```
Expected: files land in `preview/` and `preview/labels/`; `git check-ignore preview/` prints `preview/`; `report/` is NOT ignored (`report ignored? no`); `git status` shows nothing for either path.

- [ ] **Step 4: Commit**

```bash
git add tools/data/preview.ts tools/data/preview-pred.ts tools/data/preview-labels.ts tools/data/preview-model.ts .gitignore
git commit -m "$(printf 'Preview tools write to gitignored preview/\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 15: Skill + docs + `loop_emoji.py` + `train-modal.py` comments

**Files:**
- Modify: `.claude/commands/data-quality.md`
- Modify: `CLAUDE.md`
- Modify: `loop_emoji.py`
- Modify: `train-modal.py` (comments/docstring-free — so only the module-level comment lines / the `--help` text if any)
- Modify: `Taskfile.yml` (verify only; edit only if broken)

**Interfaces:**
- Consumes: everything above.
- Produces: `loop_emoji.py` commits `data.jsonl` + `report/` between iterations so the next `train-modal.py` sees a clean tree.

- [ ] **Step 1: `data-quality` skill rewrite**

In `.claude/commands/data-quality.md`:
- Front-matter `description`: remove the clause "Writes one report to `report/data-quality/`," → "…palette quality on a 200-row sample in-session. Never modifies any data file."
- Section 1 ("Refresh distributions"): replace the block that does `mkdir -p report/data-quality`, `STAT=$(ls -t report/data-stat/*.md | head -1)`, `echo "distribution report: $STAT"`, "Read `$STAT`" with:

```bash
bun run tools/data/stat.ts | tee /tmp/emojic-datastat.txt
```
"Read `/tmp/emojic-datastat.txt` (the `stat.ts` stdout). Its numbers are lifted into section 1 by reference — do not recompute."

- Section 3 ("Draw the judging sample"): change `shuf -n 200 data.jsonl > "report/data-quality/$STAMP.sample.jsonl"` to `shuf -n 200 data.jsonl > /tmp/emojic-dq-sample.jsonl` and update the `wc -l` line.
- Section 5 ("Write the report"): replace "Write to `report/data-quality/$STAMP.md` **only**." with "Present the report **in this session** (chat) — do not write any file." Keep the markdown skeleton as the shape of the chat message. Change the skeleton header line to `# Data quality report — <YYYY-MM-DD HH:MM> · commit <short sha>`.
- Section 5 final paragraph: delete the sentence requiring `git status --short` to "show only new files under `report/`"; keep "`data.jsonl` and `labels.json` byte-identical to before this run".
- Section 6 stays.

- [ ] **Step 2: `loop_emoji.py` — commit between iterations**

Add a helper next to `_run` (after `def _run` / `def _lines`):

```python
def _commit(msg: str) -> None:
    subprocess.run(["git", "add", "data.jsonl", "report"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "commit", "-m", msg], check=True)
```

In `main()`, at the end of the per-iteration block, right after `_run(["bun", "run", "tools/data/regen.ts"])`:

```python
        _run(["bun", "run", "tools/data/regen.ts"])
        _commit(f"loop_emoji iter {i}: +{added} rows")
```

And before the closing color-GAN run:

```python
    if ok:
        print("\ngoal reached -- training the color GAN locally", flush=True)
        _commit("loop_emoji: commit reports before color GAN")
        _run(["uv", "run", "python", "train_gan.py"])
```

Also add near the top of `main()`, before the loop:

```python
    from runmeta import require_clean_tree

    require_clean_tree()
```

- [ ] **Step 3: `CLAUDE.md` edits**

Make these concrete changes (search for the quoted text):
- Line ~11 (`train-modal.py` bullet): after "…copies the artifacts back (`*.pt`, `runs/`, `report/`…" leave as is, but append a sentence: "`train-modal.py`'s local entrypoint now aborts on a dirty git tree and forwards `EMOJIC_GIT_SHA` / `EMOJIC_DISPATCH_CHECKED` so Modal-trained `.pt` files carry the dispatch commit SHA."
- Line ~13 (`test_emoji.py`): change "Writes `report/test-emoji/<YY-MM-DD-HH-MM>.{md,json}`." → "Writes `report/test-emoji/<YY-MM-DD-HH-MM>-<model-sha>/{report.md,report.json,meta.yml}` — `meta.yml` carries the encoder/head `.pt` embedded metadata (training SHA, `CONFIG_PARTS`, `train.jsonl` hash) plus the metric summary."
- Line ~14 (`test_color.py`): analogous — "Writes `report/test-color/<YY-MM-DD-HH-MM>-<model-sha>/{report.html,report.json,meta.yml}`."
- Line ~25 (`tools/data/stat.ts`): change "writes `report/data-stat/<MM-DD-HH-MM>.md`" → "prints the summary to stdout". Delete the sentence "Auto-run by the `.claude/settings.json` PostToolUse hook whenever `data.jsonl` is written."
- Line ~26 (`tools/data/color-analysis.ts`): change "Writes `report/color-analysis/<MM-DD-HH-MM>.md`; touches no data file." → "Prints the report to stdout; touches no data file."
- Line ~33 (Generated docs & reports): replace the sentence listing `report/` subdirs with: "`report/` holds git-tracked per-run folders `report/<type>/<timestamp>-<model-sha>/` for the two behavioural probes only (`test-emoji`, `test-color`), each with `meta.yml` + `report.md` (+ `report.json`/`report.html`). Every other generator (`stat.ts`, `color-analysis.ts`, `fails.ts`, the `data-quality` skill) prints to stdout / reports in-session; preview tools write to gitignored `preview/`."
- Line ~41 and ~54 (verification prose): change "`uv run python test_emoji.py` (→ `report/test-emoji/`)" mentions to "(→ `report/test-emoji/<ts>-<sha>/`)". Add to the "Environment & commands" section a line: "`uv run python train.py` / `train_gan.py` and `uv run modal run train-modal.py` abort immediately on an unclean git tree (`runmeta.require_clean_tree`, no override) — commit or stash first."
- Add a bullet under "Conventions": "`.pt` files are `{"state_dict", "meta"}` blobs — save with `runmeta.save_pt`, load with `runmeta.load_pt` (`(state_dict, meta|None)`; `None` = legacy bare file)."

- [ ] **Step 4: `train-modal.py` non-code text**

If `train-modal.py` has module-level comment lines listing what it collects / does, update them to mention the clean-tree gate and the SHA passthrough. No docstrings (repo convention) — comments only, and only if such comment lines already exist. Otherwise skip.

- [ ] **Step 5: `Taskfile.yml` verify**

```bash
sed -n '70,95p' Taskfile.yml
```
`modal-train:reports` does `find "${wt}report/test-emoji" -type f -name '*.md'`. `report.md` inside `<ts>-<sha>/` still matches `-name '*.md'` (find is recursive). The `sed -n 's/^- \(MRR:.*\)/…/p'` and `grep -E '^\| acc@'` extraction reads `report.md` body — `test_emoji._render` still emits `- MRR: …` and `| acc@k | … |` lines, so it still works. **No edit needed** unless the `sed`/`grep` output is empty when you test it against a real `report.md` from Task 9 — in that case update the `sed` pattern to also accept the new header. Confirm:

```bash
D=$(ls -td report/test-emoji/*/ | head -1)
sed -n 's/^- \(MRR:.*\)/  \1/p' "${D}report.md"; grep -E '^\| acc@(1|5|10) ' "${D}report.md"
```
Expected: non-empty. If empty, `test_emoji._render` no longer emits a bare `- MRR:` line (Task 9 kept it) — recheck Task 9.

- [ ] **Step 6: Lint + skill sanity**

```bash
uv run ruff check loop_emoji.py train-modal.py && uv run ruff format --check loop_emoji.py train-modal.py
```

- [ ] **Step 7: Commit**

```bash
git add .claude/commands/data-quality.md CLAUDE.md loop_emoji.py train-modal.py Taskfile.yml
git commit -m "$(printf 'Docs + loop_emoji commit step for the reports reorg\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h')"
```

---

## Task 16: Git history surgery (one-time, operational — needs explicit user go-ahead)

**This task rewrites every commit hash on `main` and force-pushes. Do not start it without the user's explicit confirmation in this session, and confirm the background auto-commit job is paused.**

**Files:** none committed by this task — it rewrites history and deletes the working-tree `report/` contents.

- [ ] **Step 1: Confirm preconditions**

- User has said "go" for the history rewrite in this session.
- The background pipeline auto-commit job is **paused**.
- Tasks 1–15 are committed and pushed; `git status --porcelain` shows only intentional leftovers (e.g. `todo.txt`).
- `git log --oneline origin/main..main` is the set of reorg commits you expect.

- [ ] **Step 2: Full backup**

```bash
cd /home/gilad
git clone --mirror /home/gilad/Work/emojic emojic-backup.git
cd /home/gilad/Work/emojic
```

- [ ] **Step 3: Clear worktrees**

```bash
git worktree prune
git worktree list
```
For each remaining non-main worktree that is not mid-run, `git worktree remove <path>` (or finish its run first). `emojic-runs/modal-train-26-09-04-09-07`, `…-08-09`, `.worktrees/rvr-impl` must be gone or idle before Step 5.

- [ ] **Step 4: Install `git-filter-repo`**

```bash
uv tool install git-filter-repo || pipx install git-filter-repo
git filter-repo --version
```

- [ ] **Step 5: Rewrite**

```bash
git filter-repo --path report/ --invert-paths --force
```
Expected: completes; `git log --stat -- report/ | head` shows nothing; `git log --oneline | wc -l` similar count, new hashes.

- [ ] **Step 6: Delete stale working-tree reports, re-add remote, push**

```bash
rm -rf report/
git remote add origin git@github.com:emojic-co/model
git push --force origin main
```

- [ ] **Step 7: Verify the clone shrank**

```bash
git gc --prune=now
du -sh .git
cd /tmp && rm -rf emojic-verify && git clone git@github.com:emojic-co/model emojic-verify
cd emojic-verify && git log --all --stat | grep -c 'report/' ; echo "^ expect 0"
cd /home/gilad/Work/emojic
```

- [ ] **Step 8: Recreate worktrees as needed, re-enable the background job**

Recreate any run worktrees from the new `main`. Tell the user to notify collaborators to re-clone or `git fetch && git reset --hard origin/main`. Re-enable the auto-commit job.

- [ ] **Step 9: First post-rewrite report is tracked**

```bash
bun run regen
uv run python test_emoji.py   # needs enc.pt/emoji.pt
git status --porcelain report/
```
Expected: a new `report/test-emoji/<ts>-<sha>/` folder shows as untracked → `git add report/ && git commit` lands it normally.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| A `runmeta.py` (`run_meta`, `save_pt`, `load_pt`, `require_clean_tree`, `write_meta_yml`, `stamp_lines`, `model_slug`) | 2, 3, 4, 5 |
| A — `EMOJIC_GIT_SHA` / `EMOJIC_DISPATCH_CHECKED` overrides | 2 (read), 7 (set) |
| B `config.CONFIG_PARTS` | 1 |
| C save sites (`train.py` ×2, `train_gan.py`) | 6 |
| C Modal SHA passthrough | 7 |
| D2 clean-tree gate call sites | 7 (train.py, train_gan.py, train-modal.py), 15 (loop_emoji.py) |
| D load sites (5 helpers) | 8 |
| E report folder layout | 9, 10 |
| F `report.md` / `report.html` header | 9, 10 |
| G `meta.yml` schema | 9, 10 (assembled per probe), 5 (`write_meta_yml`) |
| H `export_onnx.py` `model_meta` | 11 |
| I `stat.ts` stdout | 12 |
| J PostToolUse hook removal | 12 |
| K `color-analysis.ts` / `fails.ts` stdout | 13 |
| L preview tools → `preview/` | 14 |
| M `data-quality` skill | 15 |
| N `.gitignore` (`preview/` only) | 14 |
| O git surgery | 16 |
| P docs (`CLAUDE.md`, `train-modal.py`, `Taskfile.yml`, `loop_emoji.py`, `update-model-md` flagged out of scope) | 15 |

**Placeholder scan:** Task 9/10/11 contain `[ ... ]` only where the instruction is explicitly "leave this comprehension verbatim from the current file" — that is a copy directive, not an unfilled blank. Every code step has real code. `train-modal.py` exact line numbers for `main()` / `_run_env()` are resolved by an inspect step (7.3, 15.4) because the file wasn't fully read when writing this plan.

**Type consistency:**
- `run_meta()` keys `sha/dirty/generated/config/train_sha` — used consistently in Tasks 2, 5 (`stamp_lines` reads `probe_meta["sha"]`, `["dirty"]`, `["generated"]`, `["config"]`), 9, 10.
- `load_pt` returns `(state_dict, meta|None)` everywhere (Tasks 3, 8).
- `model_slug(meta|None) -> str` — Tasks 3, 9, 10.
- `stamp_lines(model_meta, model_name, probe_meta) -> list[str]` — defined Task 5, called Task 9 (`stamp_lines(model_meta, enc_path, probe_meta)`) and Task 10 (`stamp_lines(enc_meta, enc_path, probe_meta)`). Consistent 3-positional-arg call.
- `write_meta_yml(out_dir, doc)` — 2-arg, defined Task 2/5, called Tasks 9, 10.
- `_evaluate` return widened to `tuple[list, dict]` in both `test_emoji.py` and `test_color.py` (Tasks 9, 10); callers updated in the same task.
- `mod._pt_meta` attribute name — set in Task 8, read in Tasks 9, 10, 11.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-04-reports-reorg.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Tasks 1–15 are code; Task 16 is an operational runbook I'd walk through with you directly, not a subagent.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
