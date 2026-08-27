# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emojic` trains a small multi-task char-level LSTM (`main.py`) that maps a short text string to: an emoji (one of a fixed 30-emoji palette, grouped 5-per-feeling), a feeling (`feeling` list in `main.py`; the 6 that appear in `data.jsonl` plus an inert `Neutral` slot), and three Oklab colors — `bg1`/`bg2` for a gradient background and `text_color` for the foreground text.

- `main.py` — data loading, model, training, eval, and `predict`.
- `server.py` — stdlib `http.server` app: loads `model.pt` once and serves `GET /predict?text=...` plus the static page in `web/`. Run with `uv run server.py`.
- `data.jsonl` — one JSON sample per line: `text`, `emoji`, `feeling`, `bg1`, `bg2`, `text_color`.

## Environment & commands

- Package management is `uv` only. Do not `pip install` — dependencies and the lockfile are managed with `uv add` / `uv sync`.
- `torch` is pinned to the PyTorch CPU wheel index (see `[tool.uv.sources]` in `pyproject.toml`). Keep that index config intact when editing dependencies.
- Run / verify a change: `uv run main.py` (trains for a few epochs, then prints a sample inference).
- Lint / format: `uv run ruff check .` and `uv run ruff format .` (config in `pyproject.toml`).
- Python 3.11.

## Conventions

- Oklab color values are `[L (0..1), a (-0.4..0.4), b (-0.4..0.4)]`. All three color targets and predictions are in Oklab; the color heads emit raw logits that `squash_oklab` maps into range (`sigmoid` for L, `0.4*tanh` for a/b), and the color losses are MSE in Oklab space (weighted 5x) to approximate perceptual distance.
- Char indexing reserves index 0 for padding (`PAD_IDX`); real characters in `CHARS` are numbered from 1, and `nn.Embedding` uses `padding_idx=0`. The LSTM is fed via `pack_padded_sequence`, so the classifier reads the hidden state at each row's last real character, not a trailing pad step. Sequences are always encoded to `MAX_TEXT_LEN` (train and inference must match).
- No test suite or CI yet — verification is running `main.py` and checking the loss decreases and inference looks sane. Lint/format with `ruff` before committing.
