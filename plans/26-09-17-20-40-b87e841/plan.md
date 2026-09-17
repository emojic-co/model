# Improvement Plan — 26-09-17-20-40 · b87e841

Report: report/26-09-17-20-40-b87e841/report.html
Prev plan: plans/26-09-12-23-12-52017be/plan.md
Changed since prev: 139 commits (`git log --oneline 52017be..HEAD`) — almost entirely unrelated Android-port work plus an undirected, uncoordinated sweep of `model/config.py` (`ENCODER_CHANNELS` cycled through `[100,100,100,100]` → `[90,130,100,60]` → `[120,160,120,80]` → `[120,180,120,60]`; `DROPOUT_EMOJI` bounced `0.1→0.15→0.2→0.25→0.3→0.2→0.1` repeatedly across commits, not a single controlled step; `TASK_BATCH_SIZE`/`GAN_BATCH_SIZE` env defaults and `GRAD_CLIP_GEN` also changed). Emoji vocab grew 711 → 808 (plain `regen`, incidental). The one planned diagnostic from the last loop shipped as a permanent report section (`dd317ef`, `fecb45f`): `length_acc` bucketed by text length vs. `RECEPTIVE_FIELD`.
Best emoji variant: EmojiHead (unchanged)
Loop verdict: **actionable now** — found a concrete "wrong signal" bug in the checkpoint/early-stop monitor for the encoder stage that plausibly explains why full-text Acc@1 *regressed* (0.6148 → 0.5945) despite ~139 iterations since the last plan. Fix is a one-line, low-risk change; no new diagnostic needed.

## Abstract

Priorities 1–2 (keyword/term Acc@k) remain maxed out (0.98–1.0 vs. 0.90–0.99 targets). Priority 3, full-text Acc — this loop's focus per the user's request — is red on 2 of 3 metrics: Acc@1 = 0.5945 (target 0.65, and *worse* than the last plan's 0.6148), Acc@5 = 0.786 (target 0.75, now green), Acc@10 = 0.833 (target 0.85, essentially flat). The `length_acc` diagnostic shipped last loop confirms again this run that there is no accuracy cliff at the 31-char receptive-field boundary (Acc@1 0.606/0.573 for the 16-31/32-42 buckets — a mild, gradual decline, not a structural collapse), so an architecture change to widen the receptive field is still not supported by the evidence and stays recommend-only. The real finding this loop is in `model/train.py`: the encoder stage's checkpoint/early-stop monitor (`loss/e/val`, `ModelCheckpoint(monitor="loss/e/val", ...)`) is computed over `eval_data_loader()` with its default `mix_sources=True`, which stochastically replaces up to ~40% of the "validation" batch (two sources at `SAMPLING_BASE_RATE=0.2` each) with `data/keywords.jsonl`/`data/terms.jsonl` rows — tasks the model already solves at 0.98–1.0 accuracy. `tools/report.py`, by contrast, reads `EVAL_PATH` directly with no mixing, so its full-text Acc@k is pure. This means the signal used to pick/early-stop checkpoints during the 139-commit sweep was diluted toward an already-solved sub-task and was not sensitive to full-text quality — a plausible mechanism for why undirected config churn produced a full-text *regression* that the training-time metric (`MRR/e/val` plateaued healthy at 0.81–0.82, train/val gap ≈ 0) never flagged. Color energy (priority 4, below focus) got noticeably worse on 4/5 tags this loop (e.g. `dark` 0.210 → 0.346) but stays out of scope — the user asked specifically about full-text emoji prediction.

## Current status

| Goal (priority) | Target (goals.yml) | Current value | Status | Δ vs prev plan |
|---|---|---|---|---|
| Keyword acc@1 (1) | ≥0.97 | 0.9810 | 🟢 good | not comparable — vocab 711→808 |
| Keyword acc@5 (1) | ≥0.98 | 0.9992 | 🟢 good | not comparable |
| Keyword acc@10 (1) | ≥0.99 | 1.0000 | 🟢 good | not comparable |
| Term acc@1 (2) | ≥0.80 | 0.9943 | 🟢 good | not comparable |
| Term acc@5 (2) | ≥0.85 | 1.0000 | 🟢 good | not comparable |
| Term acc@10 (2) | ≥0.90 | 1.0000 | 🟢 good | not comparable |
| **Full-text acc@1 (3)** | **≥0.65** | **0.5945** | 🔴 **red** | **↓ from 0.6148 — regressed** |
| **Full-text acc@5 (3)** | **≥0.75** | **0.7860** | 🟢 **good** | **↓ from 0.7949, but crossed the target** |
| **Full-text acc@10 (3)** | **≥0.85** | **0.8330** | 🔴 **red** | **↓ from 0.8354, still just below** |
| Color energy · red (4) | ≤0.10 | 0.2698 | 🔴 red | worse — was 0.1948 |
| Color energy · green (4) | ≤0.10 | 0.1136 | 🔴 red | roughly flat — was 0.1200 |
| Color energy · blue (4) | ≤0.10 | 0.1642 | 🔴 red | worse — was 0.1470 |
| Color energy · dark (4) | ≤0.10 | 0.3458 | 🔴 red | much worse — was 0.2103 |
| Color energy · bright (4) | ≤0.10 | 0.0632 | 🟢 good | roughly flat |
| Style acc@1/@5/@10 (5) | ≥0.60/0.80/0.90 | 0.672/0.936/0.984 | 🟢 good | improved |
| Max text len (6) | ≥42 | 42 | 🟢 good | unchanged |
| Emoji vocab size (7) | ≥700 | 808 | 🟢 good | 711 → 808 |
| Vocab coverage (8) | all groups ≥ target | 100/100 | 🟢 good | 54/100 → 100/100 |

Priority-1 regression gate: keyword/term Acc@k all still 0.98–1.0, no regression (though not a clean comparison — vocab grew 711→808 since the last plan's checkpoint).

## Gaps

- **Full-text Acc@1 regressed despite ~139 commits of activity (priority 3, focus, actionable now)** — evidence: `model/train.py:535` (`_train_encoder`) calls `val_dl = eval_data_loader()`, and `model/data.py:261`'s `eval_data_loader(mix_sources=True)` default means the "val" set used for `ModelCheckpoint(monitor="loss/e/val", ...)` (line 541) and `EarlyStopping(monitor=monitor, ...)` (line 556) stochastically swaps in `data/keywords.jsonl`/`data/terms.jsonl` rows at up to ~40% of each batch (`SAMPLING_BASE_RATE=0.2` × 2 sources, `model/data.py:199-221`) in place of the real full-text eval row. Keyword/term rows are already at 0.98–1.0 accuracy, so mixing them in pulls `loss/e/val` toward an already-solved task and desensitizes it to full-text drift — while `tools/report.py:1048` reads `EVAL_PATH` directly with no mixing, so the report's full-text Acc@k is pure. TensorBoard confirms the disconnect: the (diluted) `MRR/e/val` on the current checkpoint's run is healthy and flat (0.81–0.82, matching `MRR/e/train`, no train/val gap), yet the report's pure full-text Acc@1 dropped 0.6148 → 0.5945 over the same period. This is a **wrong-signal bug**, not a capacity or data problem — the compass being used to pick checkpoints during 139 commits of churn was not measuring the thing being optimized for.
- **No receptive-field cliff (priority 3, re-confirmed, no action)** — this run's `length_acc` section: Acc@1 is 0.606 (16-31 chars, n=1346) vs. 0.573 (32-42 chars, n=634) — a mild decline, not the sharp drop a hard capacity ceiling would produce. `block_capacity` still shows the dilation-8 block's contribution collapsing on eval text (8.5%) vs. keywords (12.0%)/terms (7.4%) — real, but the previous loop's diagnostic already showed this doesn't translate into a length-driven accuracy cliff, and this run reconfirms it. Architecture change stays recommend-only, deprioritized behind the monitor fix.
- **Uncontrolled hyperparameter/architecture churn (process gap, not a single actionable item)** — `ENCODER_CHANNELS` and `DROPOUT_EMOJI` were changed dozens of times across the 139 commits without holding other variables fixed, so none of that churn can be individually credited or blamed for the Acc@1 regression. Once the monitor is fixed (action below), a single controlled re-run against the *current* config is needed before drawing conclusions about any specific config value.
- **Color energy regressed on 4/5 tags (priority 4, out of scope this loop)** — `dark` 0.210→0.346, `red` 0.195→0.270, `blue` 0.147→0.164. Noted for the next loop that focuses on color; not actioned here since the user asked specifically about full-text emoji prediction (priority 3).

## Proposed actions

### Metric Adjustments (this loop's real output)
- **Wrong signal — fix the encoder checkpoint/early-stop monitor's val set.** `model/train.py:535`, inside `_train_encoder`: change `val_dl = eval_data_loader()` → `val_dl = eval_data_loader(mix_sources=False)`, matching what `_train_gan` already does at `model/train.py:589`. This makes `loss/e/val` (the `ModelCheckpoint`/`EarlyStopping` monitor at lines 541/556) track pure full-text validation loss — the same distribution `tools/report.py` grades against — instead of a blend diluted ~40% by near-ceiling keyword/term rows.
  - **Expected impact**: does not itself change the model — it changes *which* checkpoint gets exported as "best" and when training stops. Direction: should make future runs' exported checkpoint track full-text Acc@1/@10 (currently red) more faithfully, likely recovering some or all of the recent regression once the next `train --local` completes. Magnitude: unverified until the next run — this is a measurement fix, not a capacity increase, so it should not be expected to blow past the 0.65/0.85 targets on its own if the model is genuinely near a real capacity ceiling once measured correctly.
  - **Risk to priority 1**: negligible. Keyword/term rows still appear in *training* batches (`train_data_loader`/`train_ds(mix_sources=True)` is untouched) at the same rate; only the val/checkpoint-selection distribution changes, and keyword/term are so far above target (0.98–1.0 vs. 0.90/0.80) that a checkpoint chosen for full-text quality is extremely unlikely to drop them below target.
  - **Directly implementable**: yes — a training-loop wiring change using an existing, already-used parameter (`mix_sources=False` is already exercised by `_train_gan`), not a model/loss-shape change.

### Model Architecture (avoid unless the above proves insufficient)
- **Wider receptive field** (add a 5th dilated block, or raise dilation/kernel to push `RECEPTIVE_FIELD` past `MAX_TEXT_LEN=42`): still **not supported** by this run's `length_acc` evidence (no cliff at the 31-char boundary, again). **Recommend-only — do not apply.** Re-evaluate only if, after the monitor fix and a clean re-run, full-text Acc@1 is still red *and* a re-run of `tools/analysis/length_vs_acc.py` starts showing a sharper length-correlated drop than today's mild decline.

### Model Configuration / Training Configuration
- No new knob change proposed this loop beyond the monitor fix above. `DROPOUT_EMOJI`, `ENCODER_CHANNELS`, etc. were already churned incoherently across the 139 commits (see Gaps); tuning any of them further before the monitor is fixed would just add another uncontrolled variable on top of an already-uninterpretable sweep. Re-open this section next loop once a clean, correctly-monitored baseline exists.

### Data
- No change proposed. Full-text Acc's gap is a signal/measurement problem this loop, not a coverage or volume one — vocab (808) and eval set (2000 rows) are healthy, and `length_acc`'s per-bucket `n` (20 / 1346 / 634) shows no starved bucket.

### `goals.yml` adjustments
- No change proposed — Acc@5 just crossed its target validating the targets are reachable in principle; Acc@1/@10 stay open and unchanged.

## Applied this run

User picked a more targeted variant of the Metric Adjustments option: rather than de-mixing the whole val loader, log a new `full_text/acc@1/val` metric filtered to `SRC_FULL` rows within the existing mixed val batch, and monitor *that* for checkpoint selection + early stopping (keeping `mix_sources=True` so `keyword/acc@1`/`term/acc@1` val diagnostics are unaffected).

**`model/train.py`**:
- `_step()`: inside the existing `if split == "val":` per-source loop (`SAMPLING_SOURCES`), added a parallel `SRC_FULL`-masked block that logs `full_text/acc@1/val` (via the existing `_log` helper, `on_epoch` aggregated) whenever the val batch contains full-text rows.
- `_train_encoder()`: `monitor` changed from `"loss/e/val"` (`mode="min"`) to `"full_text/acc@1/val"` (`mode="max"`) for both `ModelCheckpoint` and `EarlyStopping`.
- Updated the `train()` docstring's stale "Heads" note to name the new monitor and its dependency on `"emoji" in --heads` (the old note's "regardless of --heads" claim was already inaccurate for `loss/e/val`, which is also only logged when `"emoji"` is in `--heads`; the default/production pipeline always trains all heads, so this doesn't affect the real iteration loop, only ad-hoc single-head debug runs).

```
$ uv run ruff check model/train.py
All checks passed!
```
(`ruff format --check` skipped for `model/train.py` per the documented pre-existing-unformatted exception.)

Not yet evaluated — needs the next `train --local` (the user's to run, loop step 1). Expect the exported `enc.pt`/`emoji.pt` checkpoint to now be selected by pure full-text Acc@1 instead of a keyword/term-diluted loss, which should make the next report's full-text Acc@1/@10 a more faithful reflection of what the encoder can do — recovering some/all of this run's regression if the dilution theory is correct, or reproducing it (with a now-trustworthy signal) if the model is genuinely at a real ceiling.
