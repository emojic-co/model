# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emojic` trains a small multi-task LSTM (`main.py`) that maps a short text string to: an emoji, a feeling (one of 6), and two Oklab colors (`bg1`, `bg2`) for a gradient background. All logic currently lives in `main.py`.

## Environment & commands

- Package management is `uv` only. Do not `pip install` — dependencies and the lockfile are managed with `uv add` / `uv sync`.
- `torch` is pinned to the PyTorch CPU wheel index (see `[tool.uv.sources]` in `pyproject.toml`). Keep that index config intact when editing dependencies.
- Run / verify a change: `uv run main.py` (trains for a few epochs, then prints a sample inference).
- Lint / format: `uv run ruff check .` and `uv run ruff format .` (config in `pyproject.toml`).
- Python 3.11.

## Conventions

- Oklab color values are `[L (0..1), a (-0.4..0.4), b (-0.4..0.4)]`. Background-color targets and predictions are in Oklab, and the bg losses are MSE in Oklab space (weighted 5x) to approximate perceptual distance.
- No test suite, linter, or CI yet — verification is running `main.py` and checking the loss decreases and inference looks sane.
