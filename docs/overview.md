# emojic overview

`emojic` is a full pipeline for generating emoji cards from short text.
Given one text input, it predicts:

- emojis (multi-label)
- style labels (multi-label)
- a color palette (`bg1`, `bg2`, `fg`)

The same trained model is exported to ONNX and used in both the web app and Android app.

---

## What this repository contains

This repo combines four systems:

1. **Data generation and curation (TypeScript/Bun)**
2. **Model training and inference (Python/PyTorch Lightning)**
3. **Packaging for browser/mobile inference (ONNX export)**
4. **Evaluation + reporting (HTML report, goals, plans)**

It is not only a model repo; it is an end-to-end product/training loop.

---

## End-to-end flow

## 1) Build / grow data

Data starts from a large JSONL corpus (`data/data.jsonl`) and auxiliary sources (CLDR, emojilib, internal keyword/term files, color-focused slices, flags, etc.).

Main TypeScript scripts in `tools/data/` handle:

- corpus growth (`upsample.ts`)
- label/source merges (`keywords.ts`, `cldr.ts`, `emojilib.ts`)
- dedupe + split + vocab rebuild (`regen.ts`)

`regen.ts` is the important normalization step that:

- collapses near-duplicate texts
- caps per-emoji frequency dominance
- derives final emoji vocabulary (`data/labels.json`)
- writes `data/train.jsonl` and `data/eval.jsonl`

The Python model reads those outputs directly.

## 2) Train model

Training entrypoint: `model/train.py` (CLI script name `train`).

Two-stage training:

1. **Encoder stage (`LitEncoder`)**
   - trains text encoder + style head + emoji head
   - also trains a conditional color critic used for color-text compatibility
2. **GAN stage (`LitColorGAN`)**
   - freezes text encoder
   - trains color generator + critics for palette generation

Training is managed with PyTorch Lightning and early stopping/checkpoint callbacks.

## 3) Export artifacts

`model/export_onnx.py` loads `.pt` modules and exports:

- `web/public/model.onnx`
- `web/public/meta.json`
- `web/public/config.json`

It also copies model artifacts into Android assets.

## 4) Run inference in apps

- Web app loads ONNX with `onnxruntime-web`
- Text normalization in browser mirrors Python normalization logic
- Outputs are decoded into emoji/style choices + multiple color palettes

## 5) Evaluate/report/plan

`tools/report.py` creates a report folder (`report/<timestamp>/`) with metrics and analysis sections.

Planning/status files:

- `goals.yml` (targets)
- `plans/<timestamp>/plan.md` (derived next actions)
- `report/<timestamp>/report.html` (current measured state)

---

## Model architecture (current implementation)

Defined in `model/model.py` and configured by `model/config.py`.

### Text encoder

Character-level encoder:

- fixed vocab from `model/data.py` (`CHARS`, including English, Hebrew, digits, punctuation, space)
- fixed max length (`MAX_TEXT_LEN = 42`)
- embedding layer + dilated Conv1d blocks
- masked global max pooling
- projection to text embedding

No tokenizer/pretrained language model is used at inference.

### Emoji + style heads

Both are retrieval-style heads:

- project text embedding to a head-specific latent space
- score against learned label embeddings
- output one logit per label

Loss is multi-positive Log-Sum-Exp InfoNCE (`lse_infonce` in `model/train.py`).

### Color generator

Conditional generator (`ColorGen`) maps text embedding + random noise to 9 values (3 RGB colors as offsets around 127.5):

- `bg1` (3 channels)
- `bg2` (3 channels)
- `fg` (3 channels)

Color critics (`CondColorCritic`, `ColorCritic`) provide adversarial/compatibility training signals.

---

## Data interface expected by Python

`model/data.py` parses rows with fields like:

- `text`
- `emojis` (space-separated string)
- `styles` (string array)
- optional `colors` list with `{ bg: [hex, hex], fg: hex }`

Key preprocessing:

- lowercase
- collapse whitespace
- clip repeated chars
- drop out-of-vocab chars
- pad/truncate to fixed length

A critical project invariant is that normalization in Python and web must stay equivalent.

---

## Main repository areas

- `model/` — core ML code (config, model modules, train loop, export, prediction)
- `tools/data/` — dataset build/growth/merge scripts (Bun/TS)
- `tools/report.py` — metrics + report generation
- `data/` — corpora, labels, train/eval splits, keyword/term files
- `web/` — app consuming ONNX model in browser
- `android/` — Android integration/assets
- `pt/` — active checkpoints
- `history/` — archived model artifacts
- `report/`, `plans/`, `runs/` — generated outputs

---

## Runtime/tooling stack

- **Python**: 3.13+, PyTorch, Lightning, Typer, TorchMetrics
- **JS/TS**: Bun scripts for data/tooling
- **Inference packaging**: ONNX + `onnxruntime-web`
- **Lint/tooling**: Ruff (Python)

Dependencies and scripts are split across `pyproject.toml` and `package.json`.

---

## Typical iteration loop

1. Grow or target data slices (`bun run upsample`, related data scripts)
2. Rebuild labels/splits (`bun run regen`)
3. Train (`train --local` or remote flow)
4. Export ONNX + assets (`model/export_onnx.py`, usually part of training flow)
5. Generate report (`tools/report.py`)
6. Review goals vs report and plan next iteration

---

## Important project conventions

- `files.py` / `files.ts` are central path registries.
- Model checkpoints are saved/loaded with metadata (`model/runmeta.py`), not raw ad-hoc `torch.save` usage.
- Training command expects clean git state.
- Keyword/term sources are mixed into model training as sampled sources; runtime is model-first inference.

---

## In one sentence

`emojic` is a compact, character-level, multi-task emoji-card generation system with its own data factory, training pipeline, ONNX export path, and production-facing web/mobile inference targets in a single repo.
