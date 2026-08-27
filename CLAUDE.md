# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emojic` trains a small multi-task char-level LSTM (`main.py`) that maps a short text string to two labels: an emoji (`EMOJIS`, an 80-emoji palette, fully decoupled from feelings) and a feeling (`feeling`, the 7 that all appear in `data.jsonl`). Both label sets are loaded from `labels.json`. Colors are **not** learned: `feeling_colors(feeling)` in `main.py` looks up a fixed per-feeling Oklab palette (`FEELING_PALETTE`) for `bg1`/`bg2` (gradient background) and `text_color` (foreground), and `predict` merges that into its result so `server.py` / the web page still receive colors.

- `main.py` — data loading, model, training, eval, `predict`, and the `feeling_colors` palette.
- `config.py` — flat module of training hyperparameters (`LR`, `BATCH_SIZE`, `EMBED_SIZE`, `H_SIZE`, `NUM_LAYERS`, `MAX_TEXT_LEN`, `GRAD_CLIP`, `EPOCHS`); imported by `main.py` and `server.py`.
- `labels.json` — the `feelings` and `emojis` lists, shared by `main.py` and `gen_data.ts` (single source of truth for the label sets).
- `gen_data.ts` — Bun/TypeScript synthetic-data generator (`bun run gen_data.ts`). Uses the Vercel AI SDK + GPT-5.6 Luna via the AI Gateway to write short WhatsApp-style texts for a randomly chosen `(emoji, feeling)` pair and **appends** them to `data.jsonl` (20 texts/batch × 50 batches per run). Not deterministic; each run grows the corpus. Needs `AI_GATEWAY_API_KEY` (Bun auto-loads it from `.env`).
- `server.py` — stdlib `http.server` app: loads `model.pt` once and serves `GET /predict?text=...` plus the static page in `web/`. Run with `uv run server.py`.
- `data.jsonl` — one JSON sample per line: `text`, `emoji`, `feeling` (no color fields).

## Environment & commands

- Package management is `uv` only. Do not `pip install` — dependencies and the lockfile are managed with `uv add` / `uv sync`.
- `torch` is pinned to the PyTorch CPU wheel index (see `[tool.uv.sources]` in `pyproject.toml`). Keep that index config intact when editing dependencies.
- Quick verify (no training run): `uv run ruff check .` and `uv run ruff format --check .`.
- Full run: `uv run main.py` trains for `EPOCHS` epochs (see `config.py`), writes `model.pt`, and prints a sample inference. This is slow — don't use it as a smoke test.
- Training writes TensorBoard logs to `runs/<config-name>/` (run name, from `run_name()`, encodes embed/hidden/layers/lr/batch/epochs); view with `uv run tensorboard --logdir runs`.
- The generator toolchain is Bun, not `uv`: `bun install` then `bun run gen_data.ts`.
- Python 3.11.

## Conventions

- Oklab color values are `[L (0..1), a (-0.4..0.4), b (-0.4..0.4)]`. Colors are a fixed lookup, not a model head: `FEELING_PALETTE` in `main.py` holds one `(bg1, bg2, text_color)` triple per feeling (warm/bright for Happy/Excited, cool for Calm, muted/dark for Sad/Anxious, dark saturated red for Angry, neutral grey for Neutral). `web/app.js` renders them via CSS `oklab()`. The model has only two heads (`emoji`, `feeling`), both cross-entropy; there is no color loss.
- Char indexing reserves index 0 for padding (`PAD_IDX`); real characters in `CHARS` are numbered from 1, and `nn.Embedding` uses `padding_idx=0`. The LSTM is fed via `pack_padded_sequence`, so the classifier reads the hidden state at each row's last real character, not a trailing pad step. Sequences are always encoded to `MAX_TEXT_LEN` (train and inference must match).
- No test suite or CI yet — for code changes, lint/format with `ruff` (above) before committing. Behavioral verification is a full `uv run main.py`: loss should decrease, `feeling_acc` trains well, `emoji_acc` is weak at the current `H_SIZE`/`EMBED_SIZE` and thin/imbalanced `data.jsonl`. Grow the corpus with `bun run gen_data.ts`; edit `labels.json` if you touch the label sets.
