# Reports reorg — design

Date: 2026-09-04

## Problem

Three problems with `report/` today:

1. **Too many report types.** Eight subtrees, 215 files, 13 MB: `test-emoji/` (34),
   `test-color/` (14), `data-stat/` (34), `data-quality/` (3), `color-analysis/` (2),
   `preview/` (91), `preview-labels/` (33), `preview-model/` (4).
2. **No link from a report to the model that produced it.** Every report is named
   by a bare timestamp (`%y-%m-%d-%H-%M`). The `.pt` state dicts are anonymous bare
   `state_dict`s, overwritten every run; the only run identity anywhere is the
   gitignored `runs/<CONFIG_NAME>/` TensorBoard dir.
3. **Reports are git-tracked** (259 commits, 685 blobs ever) and therefore bloat
   the repo and every worktree with obsolete history.

## Decisions (settled with the user)

- **Keep only** `test-emoji` and `test-color` as report families. Every other
  generator stops writing report files.
- **Purge all existing `report/` files from git history** (`git-filter-repo`),
  then force-push `main`.
- **Future reports stay tracked in git**, but in a new layout: one folder per
  report run, `report/<type>/<timestamp>/` containing `meta.yml` + `report.md`
  (+ `report.html` / `report.json` where the probe produces them).
- **`meta.yml`** carries the provenance: the `meta` dict embedded in each model
  `.pt` file plus report-level metadata (type, probe commit, generated-at, a
  compact metric summary).
- **`.pt` files embed a `meta` dict.** Each is saved as
  `{"state_dict": ..., "meta": {...}}` (training commit SHA + dirty flag,
  timestamp, `CONFIG_PARTS`, stage, `train.jsonl` hash). `save_pt` / `load_pt`
  helpers live in a new `runmeta.py`; `load_pt` degrades to `(state_dict, None)`
  for legacy bare `.pt` files.
- **Preview tools** write to a gitignored top-level `preview/` folder. `stat.ts` /
  `color-analysis.ts` / `fails.ts` print to stdout. `data-quality` reports
  in-session (chat) only — no file.
- Worktrees are **not** required for run tracking; per-run isolation is not made a
  policy — the identity lives in the `.pt` file and its `meta.yml`.

## Design

### A. `runmeta.py` (new, repo root)

Comment/docstring-free per repo convention. Imports `torch`, `config`, `yaml`,
stdlib only — no import of `train` / `export_onnx` / lightning (avoids cycles;
`train.py`, `train_gan.py`, `export_onnx.py`, the probes all import `runmeta`).

```
run_meta() -> {
    "sha":       "<short sha>" | "unknown",
    "dirty":     bool,
    "generated": "<ISO-8601 seconds>",
    "config":    [<config.CONFIG_PARTS entries>],
    "train_sha": "<sha256(train.jsonl)[:12]>" | None,
}
```

- `sha` — `git rev-parse --short HEAD`; `"unknown"` on any failure
  (covers a git-less Modal image).
- `dirty` — `git status --porcelain` non-empty.
- `generated` — `datetime.now().isoformat(timespec="seconds")`.
- `config` — `config.CONFIG_PARTS` (see B).
- `train_sha` — `sha256` of `Path("train.jsonl")` bytes if present, else `None`
  (guards the eval-set reshuffle: runs on different `train.jsonl` snapshots stay
  distinguishable).

```
save_pt(state_dict, path, **extra) -> None
    torch.save({"state_dict": state_dict, "meta": run_meta() | extra}, path)

load_pt(path, *, map_location="cpu") -> (state_dict, meta | None)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"], blob.get("meta")
    return blob, None          # legacy / hand-made bare state_dict

write_meta_yml(dir, report_meta, models) -> None
    # dir/meta.yml  <-  yaml.safe_dump(..., sort_keys=False)
```

`weights_only=False` is explicit (the blob holds a plain-dict `meta`); these files
are already `torch.load`ed today at the same trust level.

`pyyaml` is currently a transitive dep — promote it to a direct one
(`uv add pyyaml`) since first-party code now imports it.

### B. `config.py`

Expose the section strings `CONFIG_NAME` already builds:

```
CONFIG_PARTS = [enc_str, emj_str, style_str, gan_str, train_str]
CONFIG_NAME  = " | ".join([f'TIME: {...}', *CONFIG_PARTS])
```

Pure data, no new imports.

### C. Save sites — `save_pt` instead of `torch.save`

| file:line | files saved | `extra` |
| --- | --- | --- |
| `train.py:329` | `enc.pt` `style.pt` `emoji.pt` | `stage="task"` |
| `train.py:370` | `gen.pt` `tst.pt` | `stage="gan"` |
| `train_gan.py:77` | `gen.pt` `tst.pt` | `stage="gan"` |

Each loop body: `save_pt(mod.state_dict(), f"{name}.pt", stage=...)`. `run_meta()`
is evaluated at save time (after the best-checkpoint reload in `train.py` — the
code state is unchanged across the fit, so the SHA is right).

### D. Load sites — `load_pt` instead of `torch.load` + `load_state_dict`

Five modules have a local `_load(mod, path)` / `load(mod, path)` helper doing
`mod.load_state_dict(torch.load(path, map_location="cpu"))`. Each becomes:

```
def _load(mod, path):
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    return mod
```

| file | helper | uses `meta` |
| --- | --- | --- |
| `test_emoji.py:20` | `_load` | yes — encoder + head meta → `meta.yml` |
| `test_color.py:40` | `_load` | yes — encoder + gen meta → `meta.yml` |
| `export_onnx.py:28` | `_load` | encoder meta → `web/public/meta.json` |
| `run.py:26` | `_load` | no (stash only) |
| `train_gan.py:30` | `load` | no (stash only) |

`LitTask.load_from_checkpoint` / `LitColorGAN.load_from_checkpoint` operate on
Lightning `.ckpt` files — untouched.

### E. Report folder layout

```
report/
  test-emoji/
    26-09-04-05-48-a1b2c3d/
      meta.yml
      report.md
      report.json
  test-color/
    26-09-04-05-48-a1b2c3d/
      meta.yml
      report.html
      report.json
```

- **Folder name** = `%y-%m-%d-%H-%M-<model_sha>`, e.g. `26-09-04-05-48-a1b2c3d`,
  where `<model_sha>` is the encoder's embedded `meta["sha"]` (or `nometa` for a
  legacy `.pt`). Timestamp-first so folders sort chronologically; SHA suffix for
  collision-safety and at-a-glance linkage to the model.
- `report.md` / `report.html` — the human report, unchanged in content except the
  header (see F).
- `report.json` — kept; the full per-word / per-keyword machine data.
- `meta.yml` — provenance + summary (see G). Canonical machine-readable link to
  the model.

`REPORT_DIR` constants change from `Path("report/test-emoji")` to the same, but
the write target becomes `REPORT_DIR / stamp / "report.md"` etc., with
`(REPORT_DIR / stamp).mkdir(parents=True, exist_ok=True)`.

### F. `report.md` / `report.html` header

Two provenance lines under the title (full detail is in `meta.yml`):

```
- model: `enc.pt` — trained <sha>[ dirty] @ <train generated>  ·  see meta.yml
- probe run: <probe generated> (code <probe sha>[ dirty])
```

Legacy `.pt` with no embedded meta:

```
- model: `enc.pt` — no embedded metadata (legacy .pt)
- probe run: <probe generated> (code <probe sha>[ dirty])
```

### G. `meta.yml` schema

```yaml
report_type: test-emoji
generated: 2026-09-04T05:48:12
probe_commit: b2c3d4e
probe_dirty: false
warnings:
  - enc.pt and emoji.pt were saved from different commits
models:
  enc.pt:
    sha: a1b2c3d
    dirty: false
    generated: 2026-09-04T05:12:03
    stage: task
    train_sha: 0011aabb2233
    config:
      - "ENCODER: ch=[...] k=..."
      - "EMOJI: ..."
      - "STYLE: ..."
      - "GAN: ..."
      - "TRAIN: ..."
  emoji.pt:
    sha: a1b2c3d
    ...
summary:
  acc@1: 0.446
  acc@5: 0.606
  acc@10: 0.651
  mrr: 0.521
  n: 249
```

- `models` — one entry per `.pt` the probe loaded, value = that file's embedded
  `meta` (or `null` for a legacy file).
- `warnings` — present only when non-empty. `test_emoji` warns if `enc.pt` /
  `emoji.pt` `sha` disagree; `test_color` warns if `enc.pt` / `gen.pt` `sha` or
  `config` disagree (`stage` is expected to differ).
- `summary` — the same numbers the probe already prints to stdout.
- Written with `yaml.safe_dump(..., sort_keys=False, allow_unicode=True)`.

### H. `export_onnx.py`

Folds the encoder's `meta` into `web/public/meta.json` as `"model_meta"` (or
`null`), so the deployed site records which model it runs. No folder/`meta.yml`
here — this isn't a report.

### I. `tools/data/stat.ts`

Drop `import { mkdir, writeFile }`, `REPORT_DIR`, `mkdir` + `writeFile`; write
`doc.join("\n")` to stdout.

### J. `.claude/settings.json`

Remove the whole `PostToolUse` hook block (its only case is `data.jsonl` →
`bun run tools/data/stat.ts`). Result: `{}`.

### K. `tools/data/color-analysis.ts`, `tools/analysis/fails.ts`

Drop `mkdir` / `writeFile` / `OUT_DIR`; print the `.md` body to stdout.

### L. Preview tools → gitignored `preview/`

Only the path constant changes; still `mkdir(recursive)` + `writeFile` +
`console.log(dest)`.

| tool | old `OUT_DIR` | new `OUT_DIR` |
| --- | --- | --- |
| `tools/data/preview.ts` | `report/preview` | `preview` |
| `tools/data/preview-pred.ts` | `report/preview` | `preview` |
| `tools/data/preview-labels.ts` | `report/preview-labels` | `preview/labels` |
| `tools/data/preview-model.ts` | `report/preview-model` | `preview/model` |

### M. `data-quality` skill (`.claude/commands/data-quality.md`)

Rewrite so it produces **no file** (unchanged from the earlier decision — it is
not a kept report type):

- Front-matter `description`: drop "Writes one report to `report/data-quality/`".
- Section 1: run `bun run tools/data/stat.ts`, read its **stdout**.
- Section 3: sample to the session scratchpad, not `report/data-quality/`.
- Section 5: "present the report **in this session** (chat)"; keep the skeleton as
  the message body; header gains
  `commit: <git rev-parse --short HEAD> · generated: <date -Iseconds>`.
- Section 5 final check: drop the `git status --short` "only new files under
  `report/`" assertion; keep the "`data.jsonl` / `labels.json` byte-identical"
  check.
- Section 6 stays.

### N. `.gitignore`

```
# Browser-only preview pages, regenerated on demand.
preview/
```

`report/` is **not** ignored — future report folders are committed.

### O. Git surgery

1. **Pause the background pipeline job** that auto-commits `fix` commits to `main`.
2. Land A–N on `main` as normal commits first (clean tree; new `.gitignore` in
   history; no `report/` files staged).
3. `git worktree prune` (clears the 15 `prunable` run worktrees). Finish or
   `git worktree remove` the 3 live ones
   (`emojic-runs/modal-train-26-09-04-09-07`, `…-08-09`, `.worktrees/rvr-impl`).
4. Backup: `git clone --mirror . ../emojic-backup.git`.
5. Install `git-filter-repo` (`uvx git-filter-repo …` or `pipx install`).
6. `git filter-repo --path report/ --invert-paths` — strips every historical
   `report/` blob. New report folders committed after this point are unaffected.
7. Re-add the remote (`filter-repo` drops it):
   `git remote add origin git@github.com:emojic-co/model`.
8. `git push --force origin main`.
9. Recreate any needed worktrees from the rewritten `main`.
10. Re-enable the background pipeline job.
11. Collaborators re-clone (or hard-reset to the new `main`).

### P. Docs

- `CLAUDE.md`: lines 11, 13, 14, 25, 26, 33, 41, 54 — drop `report/data-stat`,
  `report/data-quality`, `report/color-analysis`, `report/preview*`; note the
  PostToolUse hook is gone; document the new `report/<type>/<timestamp>/`
  layout with `meta.yml` + `report.md`, the `.pt` `{"state_dict","meta"}` format
  and `runmeta.save_pt` / `load_pt`, and that preview tools now write to
  gitignored `preview/`.
- `train-modal.py`: `report/test-emoji/<ts>/` is still produced on the Modal box
  and collected back (now tracked, so it also shows in `git status` there).
  Update any comment listing old report types. `runmeta.run_meta()` degrades to
  `sha="unknown"` when the Modal image has no git — already handled.
- `Taskfile.yml`: `modal-train:reports` — `find "${wt}report/test-emoji" -name
  '*.md'` now matches `.../<ts>/report.md` (recursive find, still fine); the
  printed relative path gains the `<ts>/` segment; `-newer "${wt}.git"` mtime
  check still works. Optionally switch it to read `meta.yml` `summary`. Minimal
  change: none strictly required — verify the `sed`/`grep` metric extraction
  still hits `report.md`'s `- MRR:` / `| acc@` lines.
- `.claude/commands/update-model-md.md`: already references a non-existent
  `report/model/*.md`. **Out of scope** — flag, don't fix here.

## Out of scope

- Reproduce-a-historical-report tooling (explicit: checkout + re-run by hand).
- Per-run output directories / run-ID registry / mandatory training worktrees.
- Fixing `update-model-md`'s stale `report/model/` reference.
- `model.md` / `data.md` regeneration.

## Testing / verification

- `uv run ruff check . && uv run ruff format --check .` — `runmeta.py`,
  `config.py`, `test_emoji.py`, `test_color.py`, `export_onnx.py`, `run.py`,
  `train.py`, `train_gan.py`.
- `uv run python -c "import runmeta; print(runmeta.run_meta())"` — real short SHA,
  `dirty` matches `git status`, `train_sha` non-null when `train.jsonl` exists.
- `load_pt` round-trip: `save_pt({'w': torch.zeros(1)}, '/tmp/x.pt', stage='task')`
  → `load_pt` returns `(sd, meta)` with `stage="task"`; `torch.save(bare)` →
  `load_pt` returns `(bare, None)`.
- With local `.pt` files: `uv run python test_emoji.py` — creates
  `report/test-emoji/<ts>/{meta.yml,report.md,report.json}`; `meta.yml` parses
  (`yaml.safe_load`) and has `models.enc.pt` + `models.emoji.pt` + `summary`;
  `report.md` header shows the two provenance lines. `uv run python
  export_onnx.py` — `web/public/meta.json` has `model_meta`. (Skip if no `.pt`.)
- `bun run tools/data/stat.ts` — stdout only; `git status` clean, no
  `report/data-stat/`.
- `bun run tools/data/preview.ts …` — writes `preview/<stamp>.html`; `git status`
  clean (`preview/` ignored).
- `git check-ignore preview/` → ignored; `git check-ignore report/` → **not**
  ignored.
- Post-rewrite: `git log --stat | grep -c 'report/'` is 0 for pre-rewrite
  history; a fresh clone has no old `report/` blobs; `du -sh .git` dropped;
  a subsequent `test_emoji.py` run + commit adds a tracked `report/test-emoji/
  <ts>/` folder.
- `web/` build untouched.
