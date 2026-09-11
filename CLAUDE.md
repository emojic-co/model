# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emojic` trains a small multi-task char-level model that maps a short text string to emojis, a style label, and a color palette, and ships that model into a small web app for live inference in the browser.

Goals and status are not documented here — they live in code and generated artifacts, which are the source of truth:

- `goals.yml` — the long-term goals statement (accuracy/coverage/energy targets across the whole tree of metrics).
- `goal/<YYYY-MM-DD>-<short-sha>.yml` (newest file) — the current iteration's targets, written by the `planning-emojic-improvements` skill; schema in `goal/README.md`.
- `report/<stamp>/report.html` (newest folder) — the current measured state against both of the above, written by `tools/report.py`.
- `plans/<stamp>/plan.md` (newest folder) — the derived plan for closing the gap, written by the `planning-emojic-improvements` skill.

**Never restate a hyperparameter, path, or metric value in this file — name the symbol/file instead and read it directly.** `model/config.py` is authoritative for every hyperparameter; `files.py` / `files.ts` are authoritative for every path. Numbers here rot; symbol names don't.

### Architecture, at a glance

One linear pipeline, no forks — see the referenced modules for how each stage actually works:

- **Data → vocab**: `tools/data/regen.ts` (`bun run regen`) slices the append-only corpus master into a training vocab and train/eval split.
- **Shared trunk**: `model/model.py:TextEncoder` — a char-level convolutional encoder producing one embedding vector per text.
- **Heads trained on that trunk**: an emoji retrieval head, a style retrieval head, a keyword/model fusion combiner, and a color critic — see `model/model.py` and `model/train.py` for what each one is and how it's trained (loss functions, detach points, and co-training rules are implementation detail, not policy — read the code, this changes across iterations).
- **Non-learned keyword signal**: a hand-rolled inverted index (`data/ii.json` → `regen.ts` → `web/src/keywords.js`) runs identically offline and in the browser; the fusion head combines it with the learned emoji head.
- **Colors**: a conditional GAN (`ColorGen` / `ColorCritic` in `model/model.py`), trained separately from the classification heads on the frozen encoder.
- **Serving**: every training run refreshes `web/public/` (`model/export_onnx.py`) so the full pipeline always ships a working web app end to end — never a partial-heads checkpoint.

For the literal architecture (layer shapes, loss formulas, detach boundaries, dilation schedule, etc.), read `model/model.py` and `model/train.py` directly — don't rely on prose descriptions of them, here or elsewhere; they are actively iterated on and this file is not kept in sync with every change.

## Project structure

- `model/` — the trainable model stack (Python): encoder/heads/GAN (`model.py`), training loop (`train.py`), config (`config.py`), data loading (`data.py`), inference (`pred.py`), ONNX export (`export_onnx.py`), checkpoint I/O (`runmeta.py`), metrics (`metrics.py`), keyword tokenization (`kwtokens.py`). Internal imports are package-qualified (`from model.config import ...`); every directly-run entry point inserts the repo root onto `sys.path` before importing.
- `tools/` — every tooling script, Python and TypeScript: the eval/report generator (`report.py`), the data growth/annotation/regen toolchain (`tools/data/`), preview tooling (`preview.ts`), and misc scripts (`print_model_params.py`). Same `sys.path` convention as `model/` for the Python ones.
- `pt/` — every `.pt` checkpoint, gitignored, written by `model/train.py` and read by `model/pred.py` / `model/export_onnx.py` / `tools/report.py` via `files.py` path constants.
- `data/` — every data file. Some are committed (the append-only corpus master, vendored reference/index files, the Cards gold set); some are gitignored derived artifacts rebuilt by `bun run regen` (the train/eval split, the label vocab, prediction output). See `files.py` / `files.ts` for the authoritative list of what lives where.
- `files.py` / `files.ts` (repo root) — the single source of truth for every data/model/pt/goals path, one file per language. Add a new path here first; never hardcode a bare filename in a script.
- `report/`, `goal/`, `plans/`, `runs/`, `preview/`, `pr/`, `docs/`, `web/` — generated reports, iteration goals, derived plans, TensorBoard logs, preview output, marketing material, prose docs, and the web app, respectively. None of these are model code, tooling scripts, checkpoints, or training data in the sense above.

## Environment & commands

- Package management for the model stack is `uv` only (`uv add` / `uv sync`, never `pip install`); the data toolchain is `bun` (`bun install`).
- Quick verify (no training run): the ruff lint/format checks plus the plain-assert test scripts under `model/` and `tools/` — see their filenames for exact invocations, they change as tests are added/removed.
- Grow the corpus (`bun run train` and/or `bun run upsample`) → rebuild the training slice (`bun run regen`) → train (`train --local`, the standard full end-to-end iteration command — see `model/train.py`'s own CLI help/docstring for its stages and flags) → the run's report auto-generates, or re-run it by hand via `tools/report.py`. This is the loop every iteration runs; treat a single-stage/single-head training command as scratch work, not the iteration's real result.
- `model/train.py` aborts on a dirty git tree in every mode — commit or stash first.
- To refresh the shipped web app without retraining: `model/export_onnx.py` (reads existing checkpoints, rewrites `web/public/`). The `.pt` files are gitignored; commit the regenerated `web/public/` output instead — pushing it to `main` triggers the Pages deploy.
- Run the web app locally from `web/` (`npm install && npm run dev`; `npm test`; `npm run build`).
- Training writes TensorBoard logs under `runs/`.
- `bun run regen` must run before any Python entry point that loads the label vocab (training, prediction, export, report) — it's the thing that produces the gitignored data artifacts those depend on.

## Conventions

- Colors are a GAN, not a classifier head; style and emoji are the only two classifier heads. See `model/model.py` / `model/train.py` for how each is trained and scored — don't restate their loss functions or detach points here.
- Text normalization is duplicated by necessity between `model/data.py` and `web/src/model.js` (training vs. browser inference) — they must stay byte-identical, or the model sees a different input distribution than it was trained on. Changing normalization invalidates existing checkpoints.
- Char indexing reserves an explicit pad index; sequences are always encoded to a fixed length shared by training and inference.
- There is no pytest suite and no CI gate on the Python or web test scripts — lint/format with `ruff` before committing, and treat a full `train --local` + `tools/report.py` run as the real behavioral verification, not the quick checks. Checkpoints are overwritten even when a run is worse than the previous one — trust the logs/report, not the presence of a `.pt` file.
- Checkpoints are saved/loaded through `model/runmeta.py`'s helpers, not raw `torch.save`/`torch.load` — they carry metadata alongside the state dict.
