---
name: planning-emojic-improvements
description: Use when a training run has finished (train then tools/report.py) and the eval report needs to become a prioritized improvement plan toward the emoji / CLDR / style / color accuracy targets. Also use when asked to review the latest report/ folder or plan the next training iteration.
---

# Planning Emojic Improvements

## Overview

After each `train -> report` iteration, turn the newest `report/<stamp>-<sha>/report.json` into **one** prioritized, actionable plan spanning three pillars — dataset, model configuration, model architecture — measured against the hard targets below. Write the plan to `plans/<same-stamp>/plan.md` (git-tracked, mirrors `report/`). Optionally apply the single lowest-risk step; never retrain.

Core principle: **CLDR keyword parity is priority 1 and must never regress.** Every recommendation states its risk to the CLDR probe explicitly. A short-text-emoji gain that costs CLDR accuracy is not an improvement.

## Hard targets

| Target | Metric | Threshold |
|---|---|---|
| Short-text emoji Acc@1 | `emoji.eval` best fusion variant, k=1 | ≥ 0.80 |
| Short-text emoji Acc@5 | same, k=5 | ≥ 0.90 |
| Short-text emoji Acc@10 | same, k=10 | ≥ 0.95 |
| CLDR keyword Acc@1 (exact) | `cldr.acc_at_k[0]` † | ≥ 0.95 |
| CLDR keyword Acc@1 (fuzzy) | `cldr.acc_at_k[0]` † | ≥ 0.90 |
| Color palette | `cards.per_color.*` accuracy / OKLab distance | high (no fixed number) |
| Style | `cards.style_acc_at_k`, TB `MRR/s/val` | high (no fixed number) |

† **TODO — unresolved metric mapping.** `report.json` currently has a single CLDR probe (`cldr` section: `EmojiHead` scored over `data/cldr.jsonl`, no exact-vs-fuzzy split). The non-learned keyword predictor is only scored on eval short texts (`emoji.eval.keywords_acc_at_k`), not on CLDR. Until the split exists, report `cldr.acc_at_k` against **both** CLDR rows and flag the gap. If CLDR becomes the blocking target, the first recommended action is: extend `tools/report.py` to emit `kw`-predictor Acc@1 on `data/cldr.jsonl` with exact-postings-only vs. uFuzzy-enabled.

Priority order: **1** CLDR parity (no degradation) · **2** short-text emoji Acc@k · **3** color palette · **4** style.

## Inputs

| Source | Use |
|---|---|
| `report/<newest>-<sha>/report.json` | **Primary** — all Acc@k numbers. Confirm `provenance.issues == []` first; if not, stop and tell the user the report/`.pt` are inconsistent. |
| `runs/<CONFIG_NAME>/` (gitignored TensorBoard) | Optional — loss curves, `MRR/{e,s,fusion,kw}/val`, `auc/critic/val`, `energy/gan/val`, early-stop epoch. Parse with `uv run tensorboard` or `EventAccumulator`. |
| `model/config.py` + `uv run python model/config.py` | Current hyperparameters, param count, receptive field (15) vs `MAX_TEXT_LEN` (42). |
| `data/labels.json` | Current emoji vocab size (dynamic). |
| newest previous `plans/*/plan.md` | Regression gate — diff the scorecard. |
| `git log --oneline -15` | What changed since the last plan (data grow, config edit, arch). |

### report.json field map

- `emoji.eval.acc_at_k` — `EmojiHead` retrieval, eval short texts, k=1..10.
- `emoji.eval.fusion_{gate,gain,mix}_acc_at_k` — same via each detached combiner. Shipped variant = `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` (default `gate`). If another variant clearly wins, that is a free win.
- `emoji.eval.keywords_acc_at_k` — non-learned `kw` vector alone.
- `emoji.eval.baseline.acc_at_k` — `overlap` baseline.
- `emoji.keywords.acc_at_k` — `data/keywords.json` single-keyword probe.
- `cldr.acc_at_k`, `cldr.n`, `cldr.total` — `data/cldr.jsonl` keyword probe.
- `cards.emoji_acc_at_k`, `cards.style_acc_at_k` — shipped-graph end-to-end on the 125-row colours gold set.
- `cards.per_color.<red|green|blue|dark|bright|all>.{pure_accuracy,pure_mean_distance,gt_accuracy,gt_mean_distance}` — palette accuracy per colour.
- `keywords_flex.{candidates,missed,ranked}` — single-token keyword-vocab diagnostic; `ranked` = rank>10 misses, worst first.
- `data.records`, `data.length_distribution` — corpus size + text-length histogram.
- `labels.emojis`, `labels.styles` — integer **counts** in `report.json` (not lists); vocab size for the run. The full lists are in `data/labels.json`.

## Workflow

1. **Locate** the newest `report/` dir. Verify its `-<sha>` matches `git rev-parse --short HEAD` and `provenance.issues == []`. If stale or inconsistent, stop and say so (or run `uv run python tools/report.py` if the user wants a fresh one).
2. **Scorecard** — for every target: current value, threshold, pass/fail, and Δ vs the previous plan. Use the **best** fusion variant for the short-text emoji rows and name which one it is.
3. **Regression gate** — compare `cldr.acc_at_k[0]` (and `[4]`, `[9]`) to the previous plan. Any drop = priority-1 finding, called out at the top of the plan.
4. **Diagnose per pillar:**
   - *Dataset* — vocab size and drift since last run; most/least frequent kept emojis; `kw` mean-nonzero count; how many `keywords_flex` candidates miss (rank>10) and the worst offenders; slices where `cards`/`cldr` lag; suspected noisy annotations.
   - *Configuration* — best vs exported fusion variant; which head's val metric lags (`MRR/e` vs `MRR/s` vs `auc/critic` vs `energy/gan`); `INFONCE_TEMP`, `LR`, dropout, `EARLY_STOP_PATIENCE`, batch sizes; early-stop epoch (under- vs over-training).
   - *Architecture* — receptive field 15 vs 42 (partial coverage); channel chain / dilation / kernel; fusion combiner shape; **on-device budget**: param count and single-threaded wasm latency must stay viable — quantify any proposed size increase.
5. **Prioritize** — rank candidate actions by (expected gain toward the highest-priority *failing* target) then (cheapest / lowest-risk first: local > data-gen > Modal GPU). Respect the priority matrix. Never propose a change that risks CLDR parity without a risk line and a mitigation.
6. **Write** `plans/<report-stamp>/plan.md` from the template below (reuse the report dir's exact `<stamp>` so plan and report pair up).
7. **Optionally apply** at most one step — see next section. Record what was done (or "none") under *Applied this run*.

## Applying a low-risk step

Apply **at most one** — the top-ranked action that qualifies. Everything else is recommend-only.

**Qualifies:**
- Targeted data upsample for a clearly underperforming in-vocab emoji or keyword cluster: `bun run upsample --emojis "<e1>,<e2>"` / `--keywords "<k1>,<k2>"` / `--rare`, then `bun run regen`. Re-read the regen summary; if vocab size changed, record it (future Acc@k is no longer comparable).
- Switching `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` when the report shows another variant clearly wins, then `uv run python model/export_onnx.py`.
- One localized scalar nudge in `model/config.py` — a single knob, small step, only when TensorBoard clearly points to it: one of `LR`, `INFONCE_TEMP`, `DROPOUT_EMOJI`, `DROPOUT_STYLE`, `EARLY_STOP_PATIENCE`.

**Does NOT qualify (recommend-only):**
- Any edit to `ENCODER_CHANNELS` / `ENCODER_DILATION` / `ENCODER_KERNEL_SIZE` / `CHAR_EMBED_SIZE` or `model/model.py` structure.
- Anything touching `normalize` / `CHARS` (invalidates `.pt`, breaks web parity).
- Multi-knob sweeps, or anything needing a Modal GPU run to evaluate.
- Regenerating `data/keywords.json` — that is the `generating-emoji-keywords` skill.

**Guardrails:**
- A background job auto-commits "fix" commits to this branch (see `[[concurrent-training-pipeline-commits]]`): never `git add -A`; scope any commit to the files you touched.
- After editing a `.py`: `uv run ruff check .` and `uv run ruff format --check .`.
- After a data change: `bun run regen`, then re-read its summary.
- Never run `train` / `train --local` — see `[[dont-run-main-to-test]]`. The retrain is the user's to kick off.
- No comments or docstrings in any code touched — see `[[no-comments-or-docstrings]]`.

## Plan template

```markdown
# Improvement Plan — <stamp> · <sha>

Report: report/<stamp>-<sha>/report.html
Prev plan: plans/<prev-stamp>/plan.md   (or "none")
Changed since prev: <one line from git log>

## Scorecard

| Target | Path | Current | Threshold | Status | Δ prev |
|---|---|---|---|---|---|
| Short-text emoji Acc@1 | emoji.eval.<best>_acc_at_k[0] | 0.000 | 0.80 | FAIL -0.00 | +0.00 |
| Short-text emoji Acc@5 | ...[4] | | 0.90 | | |
| Short-text emoji Acc@10 | ...[9] | | 0.95 | | |
| CLDR Acc@1 (exact/fuzzy) † | cldr.acc_at_k[0] | | 0.95 / 0.90 | | |
| Color (all) | cards.per_color.all.pure_accuracy | | — | | |
| Style Acc@1 | cards.style_acc_at_k[0] | | — | | |

Best fusion variant: <gate|gain|mix> (exported: <...>)
† no exact/fuzzy split reported yet

## Priority-1 regression gate

CLDR Acc@1/@5/@10 vs prev: <PASS / FAIL + numbers>

## Diagnosis

### Dataset
### Model configuration
### Model architecture

## Recommended actions (priority order)

1. **[dataset|config|arch] <action>** — target: <which>, expected: <rough gain>. Risk to CLDR: <none|low|...>. Cost: <local|data-gen|Modal GPU>.
   `<command or diff>`
2. ...

## Applied this run

<what was executed, with verification output — or "none, all actions recommend-only">
```

## Common mistakes

- Comparing Acc@k across runs where the emoji vocab size changed — the dynamic vocab shifts the eval split; not comparable. Note vocab size in every plan.
- Optimizing short-text emoji at the cost of the `cldr` probe — violates priority 1.
- Treating `fusion_gate`/`fusion_mix` reading equal to `EmojiHead` as a bug — a combiner collapses to the raw model when its learned scalar is ~0; report the gain from the best variant instead.
- Reading an older `report/` dir than the latest train, or one with non-empty `provenance.issues`.
- Recommending architecture changes without quantifying the param-count / wasm-latency cost.
- Running training to "check" a recommendation — leave that to the user.

## When NOT to use

- Mid-training, or before any `report/` exists (run `uv run python tools/report.py` first).
- To build `data/keywords.json` — use `generating-emoji-keywords`.
