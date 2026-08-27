# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emojic` trains a small multi-task char-level LSTM (`main.py`) that maps a short text string to two labels: an emoji (`EMOJIS` in `main.py` — a fixed 60-emoji palette, fully decoupled from feelings) and a feeling (`feeling` list in `main.py` — the 7 that all appear in `data.jsonl`). Colors are **not** learned: `feeling_colors(feeling)` in `main.py` looks up a fixed per-feeling Oklab palette (`FEELING_PALETTE`) for `bg1`/`bg2` (gradient background) and `text_color` (foreground), and `predict` merges that into its result so `server.py` / the web page still receive colors.

- `main.py` — data loading, model, training, eval, `predict`, and the `feeling_colors` palette.
- `gen_data.py` — regenerates `data.jsonl` from scratch (`uv run gen_data.py`): 60 emojis × 7 feelings × 5 texts = 2100 rows, grouped emoji-major then feeling-minor. Deterministic. Each `text` is a short (~2-6 word) phrase joining an emoji-picture word (`EMOJI_WORDS`) with a feeling cue (`FEELING_MOODS`); edit those two dicts to change the corpus.
- `server.py` — stdlib `http.server` app: loads `model.pt` once and serves `GET /predict?text=...` plus the static page in `web/`. Run with `uv run server.py`.
- `data.jsonl` — one JSON sample per line: `text`, `emoji`, `feeling` (no color fields).

## Environment & commands

- Package management is `uv` only. Do not `pip install` — dependencies and the lockfile are managed with `uv add` / `uv sync`.
- `torch` is pinned to the PyTorch CPU wheel index (see `[tool.uv.sources]` in `pyproject.toml`). Keep that index config intact when editing dependencies.
- Run / verify a change: `uv run main.py` (trains for a few epochs, then prints a sample inference).
- Training writes TensorBoard logs to `runs/<config-name>/` (run name encodes embed/hidden/layers/lr/batch/epochs); view with `uv run tensorboard --logdir runs`.
- Lint / format: `uv run ruff check .` and `uv run ruff format .` (config in `pyproject.toml`).
- Python 3.11.

## Conventions

- Oklab color values are `[L (0..1), a (-0.4..0.4), b (-0.4..0.4)]`. Colors are a fixed lookup, not a model head: `FEELING_PALETTE` in `main.py` holds one `(bg1, bg2, text_color)` triple per feeling (warm/bright for Happy/Excited, cool for Calm, muted/dark for Sad/Anxious, dark saturated red for Angry, neutral grey for Neutral). `web/app.js` renders them via CSS `oklab()`. The model has only two heads (`emoji`, `feeling`), both cross-entropy; there is no color loss.
- Char indexing reserves index 0 for padding (`PAD_IDX`); real characters in `CHARS` are numbered from 1, and `nn.Embedding` uses `padding_idx=0`. The LSTM is fed via `pack_padded_sequence`, so the classifier reads the hidden state at each row's last real character, not a trailing pad step. Sequences are always encoded to `MAX_TEXT_LEN` (train and inference must match).
- No test suite or CI yet — verification is running `main.py` and checking the loss decreases and inference looks sane (feeling accuracy trains well; emoji accuracy is weak at the current `H_SIZE`/`EMBED_SIZE`). Regenerate the corpus with `uv run gen_data.py` if you touch the label sets. Lint/format with `ruff` before committing.
