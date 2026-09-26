# CLAUDE.md

Guidance for Claude Code in this repo. **Keep this file compact.** It is an index, not a spec — one line per file/dir pointing at the thing that actually defines behavior, never a restated number, formula, or algorithm. If code changes, this file should usually need no edit; if you're tempted to explain *how* something works, put that in the code (or its own doc) and link to it instead.

## Project

`emojic` trains a small multi-task char-level model that maps a short text string to emojis, a style label, and a color palette, and ships that model into web (`web/`) and Android (`android/`) apps for live inference.

Goals/status/plan (source of truth, not summarized here):

- `goals.yml` — targets, long-term and current alike (single source, no per-iteration target file).
- `report/<newest>/report.html` — current measured state, written by `tools/report.py`.
- `plans/<newest>/plan.md` — derived plan.

Architecture (read the code, not this file): `model/model.py` (encoder + classifier heads + GAN), `model/color.py` (color-space math: `rgb_to_oklab`, `energy_distance`, used by the GAN loss), `model/train.py` (training/loss), `tools/data/regen.ts` (corpus → vocab/split), `tools/data/keywords.ts` (CLDR+EmojiLib+wa-keywords merge → `data/keywords.jsonl`/`data/terms.jsonl`/`data/flags.jsonl` split).

## Project structure

- `model/` — trainable model stack (Python): `model.py`, `train.py`, `config.py`, `data.py`, `color.py`, `pred.py`, `export_onnx.py`, `runmeta.py`, `metrics.py` (just `r2_score`), `metric.py` (metric-key enums used for logging — distinct file from `metrics.py`, easy to confuse), `kwtokens.py`. `test_*.py` files exist here (no CI gate, see Conventions).
- `tools/` — tooling (Python + TS): `report.py`, `test_report.py`, `preview.ts`, `print_model_params.py`, `block_capacity.py`, `analysis/` (coverage/audit scripts), `cli/` (image/font/card generation), `social/` (Pinterest), `data/` (corpus growth, annotation, regen, keyword/color extraction — see files there, e.g. `annotate.ts` for LLM-based annotation).
- `android/` — native app consuming the exported ONNX model + style assets; `StyleUpdater.kt` OTA-refreshes `app/src/main/assets/style.yml`.
- `pt/` — `.pt` checkpoints, gitignored. `best/`, `exp/`, `history/` hold other checkpoint snapshots (also gitignored).
- `data/` — data files; `files.py`/`files.ts` list which are committed vs. gitignored-derived. `archive/` holds gzipped historical snapshots of `data/data.jsonl`.
- `eval/` — color-conditioned eval jsonl sets. `pred/` — prediction output jsonl.
- `files.py` / `files.ts` — source of truth for every data/model/pt/goals path (kept as parallel Python/TS mirrors, not perfectly identical — check both when adding a path). Add paths here first.
- `report/`, `plans/`, `runs/`, `preview/`, `pr/`, `docs/`, `web/` — generated reports, derived plans, TensorBoard logs, previews, marketing, prose docs, web app.
- `play/` — Google Play publishing: `publish.md` (checklist), `assets/` (store listing graphics), plus gitignored signing/service-account files.

## Environment & commands

- Model stack: `uv`, entry points are the console scripts in `pyproject.toml` (`train`, `pred`, `report`), not direct `python model/train.py` invocation. Data toolchain: `bun` (`bun install`); script names live in `package.json` — key ones are `upsample`, `regen`, `export-style`, `export-style-samples`, `preview`; check `package.json` for the rest (keyword/color/CLDR build steps, `web`/`web:build`/`web:test` wrappers, `pinterest`, `imgs`, `serve`).
- Loop: grow corpus (`bun run upsample`, topic-rotation by default or a targeted mode) → `bun run regen` → `train --local` (see `model/train.py` for flags) → report auto-generates (or `tools/report.py`/`report` console script by hand). A single-head/single-stage train is scratch work, not the iteration result.
- `model/train.py` aborts on a dirty git tree via `runmeta.require_clean_tree()` — commit/stash first.
- Refresh web + Android app without retraining: `model/export_onnx.py` (`export_onnx` console-script-adjacent, run via `uv`) — writes `model.onnx`/`meta.json` into both `WEB_PUBLIC_DIR` and `ANDROID_ASSETS_DIR`. `.pt` is gitignored; commit `web/public/` instead.
- Refresh card visual style (fonts/weights/patterns/layout ratios) without retraining: `bun run export-style` — see `tools/data/export-style.ts`.
- Web app: from `web/`, `npm install && npm run dev` / `npm test` / `npm run build`.
- `bun run regen` must run before any Python entry point that loads the label vocab.

## Conventions

- Colors are a GAN (`ColorGen`+`Critic` in `model/model.py`, trained via `LitColorGAN`), not a head; style/emoji are classifier heads (`StyleHead`/`EmojiHead`, trained in `LitEncoder`) — see `model/model.py` / `model/train.py`.
- `model/data.py` `normalize()` and `web/src/model.js` `normalize()` must stay byte-identical (training vs. browser inference); changing either invalidates checkpoints.
- `web/src/feelings.js` + `web/src/patterns.js` are the source of truth for per-style visuals (font/weight/pattern/opacity); `web/public/style.yml` is generated from them by `tools/data/export-style.ts` and synced into the Android app (`android/app/src/main/assets/style.yml`, OTA-refreshed by `StyleUpdater.kt`) so both apps render identical cards. Edit the JS, then re-run `bun run export-style` — `tools/data/export-style.test.ts` fails the build if they drift.
- Fixed-length char indexing with an explicit pad index, shared by training and inference.
- No CI gate on tests (test files exist under `model/` and `tools/` but aren't enforced) — `ruff` before committing; real verification is a full `train --local` + `tools/report.py` run, not the quick checks. Checkpoints get overwritten even when worse — trust the report, not file presence.
- Save/load checkpoints via `model/runmeta.py` (`save_pt`/`load_pt`), not raw `torch.save`/`load` — no other file in `model/` bypasses this.
