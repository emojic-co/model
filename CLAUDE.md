# CLAUDE.md

Guidance for Claude Code in this repo. **Keep this file compact.** It is an index, not a spec — one line per file/dir pointing at the thing that actually defines behavior, never a restated number, formula, or algorithm. If code changes, this file should usually need no edit; if you're tempted to explain *how* something works, put that in the code (or its own doc) and link to it instead.

## Project

`emojic` trains a small multi-task char-level model that maps a short text string to emojis, a style label, and a color palette, and ships that model into a small web app for live inference in the browser.

Goals/status/plan (source of truth, not summarized here):

- `goals.yml` — long-term goals.
- `goal/<newest>.yml` — current iteration's targets (`goal/README.md` for schema; written by `planning-emojic-improvements`).
- `report/<newest>/report.html` — current measured state, written by `tools/report.py`.
- `plans/<newest>/plan.md` — derived plan, written by `planning-emojic-improvements`.

Architecture (read the code, not this file): `model/model.py` (encoder + heads + GAN), `model/train.py` (training/loss), `tools/data/regen.ts` (corpus → vocab/split), `tools/data/keywords.ts` (CLDR+EmojiLib merge → `data/keywords.jsonl`/`data/terms.jsonl` split).

## Project structure

- `model/` — trainable model stack (Python): `model.py`, `train.py`, `config.py`, `data.py`, `pred.py`, `export_onnx.py`, `runmeta.py`, `metrics.py`, `kwtokens.py`.
- `tools/` — tooling (Python + TS): `report.py`, `tools/data/` (growth/annotate/regen), `preview.ts`.
- `pt/` — `.pt` checkpoints, gitignored.
- `data/` — data files; `files.py`/`files.ts` list which are committed vs. gitignored-derived.
- `files.py` / `files.ts` — source of truth for every data/model/pt/goals path. Add paths here first.
- `report/`, `goal/`, `plans/`, `runs/`, `preview/`, `pr/`, `docs/`, `web/` — generated reports, iteration goals, derived plans, TensorBoard logs, previews, marketing, prose docs, web app.

## Environment & commands

- Model stack: `uv` (`uv add`/`uv sync`, never `pip install`). Data toolchain: `bun` (`bun install`).
- Loop: grow corpus (`bun run train` / `bun run upsample`) → `bun run regen` → `train --local` (see `model/train.py` for flags) → report auto-generates (or `tools/report.py` by hand). A single-head/single-stage train is scratch work, not the iteration result.
- `model/train.py` aborts on a dirty git tree — commit/stash first.
- Refresh web app without retraining: `model/export_onnx.py`. `.pt` is gitignored; commit `web/public/` instead.
- Web app: from `web/`, `npm install && npm run dev` / `npm test` / `npm run build`.
- `bun run regen` must run before any Python entry point that loads the label vocab.

## Conventions

- Colors are a GAN, not a head; style/emoji are the only classifier heads — see `model/model.py` / `model/train.py`.
- `model/data.py` normalize and `web/src/model.js` normalize must stay byte-identical (training vs. browser inference); changing either invalidates checkpoints.
- Fixed-length char indexing with an explicit pad index, shared by training and inference.
- No pytest suite, no CI gate on tests — `ruff` before committing; real verification is a full `train --local` + `tools/report.py` run, not the quick checks. Checkpoints get overwritten even when worse — trust the report, not file presence.
- Save/load checkpoints via `model/runmeta.py`, not raw `torch.save`/`load`.
