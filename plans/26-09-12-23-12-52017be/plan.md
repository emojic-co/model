# Improvement Plan — 26-09-12-23-12 · 52017be

Report: report/26-09-12-23-12-52017be/report.html
Goals file written: goal/2026-09-12-52017be.yml
Prev plan: plans/26-09-11-10-48-7ee7db3/plan.md
Changed since prev: 8 unplanned `train`/`fix` iterations since the last plan (fusion combiner (`FusionHead`) and the standalone kw-search path removed project-wide per `goal/README.md`; encoder swept through several shapes, currently `[100,140,120,80]` dilation `[1,2,4,8]`; emoji vocab grew 400 → 711 via plain `regen`); the tracked `goal/2026-09-11-d577316.yml` uses the old `exact keyword`/`fuzzy keyword` schema, which no longer matches `report.json`'s current `keyword`/`term`/`full text` tree (reads back unmeasured).
Best emoji variant: EmojiHead (no combiner — fusion architecture removed)
Loop verdict: **blocked — need diagnostic** — block-capacity evidence already points at a receptive-field/pooling limitation for full-text (priority 3), but the report can't yet show whether misses concentrate in texts longer than the 31-char receptive field. This loop's #1 action ships that diagnostic before recommending the Modal-GPU architecture change it would justify.

## Abstract

Priorities 1-2 (keyword/term Acc@k) are maxed out (0.99-1.0 vs 0.90/0.80 hard targets) — nothing to do there. Priority 3, full-text Acc, is the focus: Acc@1 is amber (0.615 vs 0.75 hard target), Acc@5/@10 are red but within 0.005-0.015 of their hard targets (0.795/0.835 vs 0.80/0.85). TensorBoard shows `MRR/e/val` has plateaued and gone noisy (0.758-0.773 over the last ~2k steps) while `MRR/e/train` keeps climbing toward 0.79 — a growing train/val gap, the signature of a model near its current ceiling rather than an undertrained one. The report's own `block_capacity` section explains a plausible mechanism: the deepest, most-dilated encoder block (dilation 8) supplies 40.1% of the pooled embedding on keyword rows but only 12.9% on full-text eval rows, and `RECEPTIVE_FIELD` (31 chars, from `model/config.py`) doesn't cover `MAX_TEXT_LEN` (42) — each conv position only ever sees a local window, so cross-sentence composition has to happen through max-pooling across positions rather than within one receptive field. That's a structural/architecture story, not a data-volume or LR one, but the report can't yet confirm the misses actually concentrate in longer texts — this plan's first move is a cheap, read-only diagnostic to check that before spending a GPU run on an architecture change. Color energy (priority 4, below focus) has a real gap on 4 of 5 color tags (red/green/blue/dark all ~1.2-2x over the 0.10 threshold) but stays deferred this loop since priority 3 is still open.

## Current goals & status

| Goal (priority) | Global target (goals.yml) | Current target (this iteration's goal/*.yml) | Current value | Status | Δ vs prev plan |
|---|---|---|---|---|---|
| Keyword acc@1 (1) | ≥0.90 | ≥0.90 | 0.9947 | 🟢 good | not comparable — fusion combiner removed + vocab 400→711 |
| Keyword acc@5 (1) | ≥0.95 | ≥0.95 | 0.9996 | 🟢 good | not comparable |
| Keyword acc@10 (1) | ≥0.99 | ≥0.99 | 1.0000 | 🟢 good | not comparable |
| Term acc@1 (2) | ≥0.80 | ≥0.80 | 1.0000 | 🟢 good | not comparable — new goal branch (`term`) post-refactor |
| Term acc@5 (2) | ≥0.85 | ≥0.85 | 1.0000 | 🟢 good | not comparable |
| Term acc@10 (2) | ≥0.90 | ≥0.90 | 1.0000 | 🟢 good | not comparable |
| Full-text acc@1 (3) | ≥0.75 | ≥0.56 (old, superseded → 0.65) | 0.6148 | 🟡 amber | not comparable — fusion removed, vocab changed |
| Full-text acc@5 (3) | ≥0.80 | — (new: 0.80) | 0.7949 | 🔴 red | not comparable |
| Full-text acc@10 (3) | ≥0.85 | — (new: 0.85) | 0.8354 | 🔴 red | not comparable |
| Color energy · global (4) | ≤0.01 | ≤0.05 (held) | n/a | ⚪ na | `cards.energy` still unwired |
| Color energy · red (4) | ≤0.10 | ≤0.19 (held) | 0.1948 | 🔴 red | new leaf split (was single `red` in old schema too, values not comparable — different vocab/checkpoint) |
| Color energy · green (4) | ≤0.10 | ≤0.12 (held) | 0.1200 | 🔴 red | not comparable |
| Color energy · blue (4) | ≤0.10 | ≤0.15 (held) | 0.1470 | 🔴 red | not comparable |
| Color energy · dark (4) | ≤0.10 | ≤0.21 (held) | 0.2103 | 🔴 red | not comparable |
| Color energy · bright (4) | ≤0.10 | ≤0.06 (held) | 0.0608 | 🟢 good | not comparable |
| Style acc@1 (5) | ≥0.60 | ≥0.60 (held) | 0.6720 | 🟢 good | not comparable |
| Style acc@5 (5) | ≥0.80 | ≥0.80 (held) | 0.9040 | 🟢 good | not comparable |
| Style acc@10 (5) | ≥0.90 | ≥0.90 (held) | 0.9760 | 🟢 good | not comparable |
| Max text len (6) | ≥42 | ≥42 (held) | 42 | 🟢 good | unchanged |
| Emoji vocab size (7) | ≥700 | ≥700 (held) | 711 | 🟢 good | 400 → 711 (+311, plain `regen`, not targeted upsampling) |
| Vocab coverage (8) | all ≥ target | not gated this loop | 54/100 groups | ⚪ deferred | 40/100 → 54/100 (incidental, from the vocab jump) |

Priority-1 regression gate — keyword Acc@1/@5/@10 vs prev plan's actual: **N/A, not a clean regression check** — the fusion combiner the priority-1 metric used to grade (`keyword.fusion.acc_at_k`) no longer exists; the current metric is the bare `EmojiHead` classifier over `data/keywords.jsonl`. Both the old and new numbers are near-ceiling (0.858 old fusion vs 0.995 now), so there's no evidence of regression, but they aren't the same measurement.

## Current gaps

- **Full-text retrieval capped below its architecture's current shape (priority 3, focus)** — `block_capacity` rows show the dilation-8 block contributing 40.1% of the pooled embedding on `keywords` but only 12.9% on `eval` (full text), and `w_norm`-normalized activation RMS falls ~9x from keywords to eval at that block (0.167 → 0.018) vs ~1.3x at block 0 (0.225 → 0.126) — the gap grows with receptive field, not text content. `model/config.py` confirms `RECEPTIVE_FIELD = 31` vs `MAX_TEXT_LEN = 42` ("partial coverage"). TensorBoard (`MRR/e/val` plateaued 0.758-0.773 over ~2k steps while `MRR/e/train` → 0.79) shows the model isn't undertrained — it's at a ceiling consistent with this structural read. **Not yet confirmed**: whether full-text *misses* specifically cluster in rows longer than ~31 chars (the report has no per-length breakdown of `emoji.eval.acc_at_k` yet) — that's the missing evidence before recommending a receptive-field-widening architecture change.
- **Color energy red on 4/5 tags (priority 4, below focus)** — `cards.per_color.{red,green,blue,dark}.gt_mean_distance` are 0.195/0.120/0.147/0.210, all ~1.2-2.1x over the 0.10 threshold, while `bright` passes at 0.061. TensorBoard's aggregate `energy/gan/val` (0.026-0.038) looks close to `energy/gan/ref` (0.016-0.04) — the aggregate metric masks a real per-tag skew. Deferred this loop (priority 3 still open), but flagged for next focus.
- **`keywords_flex` misses look like real vocabulary gaps, not retrieval failures (informational, no open goal)** — worst ranked misses (`rotary`→☎️ rank 608, `snare`→🥁 rank 548, `heeled`→👢 rank 539, `url`/`www`→🔗/🌐 rank 400+) are single-token keywords with no close semantic neighbor already in the eval-adjacent training distribution; only 27 of 1663 candidates miss the top-10 at all, so this isn't the priority-3 bottleneck.

## Proposed actions

### Data
- **Regen params**: no change proposed — vocab (711) and split sizes look healthy; no goal points at `--min-count`/`--max-count`/`--n`.
- **Upsampling**: no upsample proposed. The open goal (full-text Acc) is a retrieval-shape problem on text the corpus already covers (`data.records.eval` = 2000, full sentences), not a missing-concept coverage hole — ruled out before reaching for `bun run upsample`.
- **`KEYWORDS_SAMPLING_RATE` / `TERM_SAMPLING_RATE`**: current 0.2 / 0.05 → no change proposed. Priority 1-2 are already maxed (0.99-1.0); there's no headroom to chase there and no evidence these knobs affect priority 3 (they control how often keyword/term rows appear in *training* batches, not the eval mix).
- **Other data adjustments**: none this iteration.

### Model Configuration

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `EMOJI_EMBED_SIZE` | 60 | — | no evidence pointing here | — |
| `STYLE_EMBED_SIZE` | 16 | — | no evidence pointing here | — |
| `DROPOUT_EMOJI` | 0.1 | 0.15 | `MRR/e/val` plateaued while `MRR/e/train` keeps climbing (0.767→0.79 train/val gap growing over the last ~2k steps) — a small regularization bump is the cheapest lever to test whether the gap (not the ceiling) is limiting val. | Small, uncertain — likely closes some train/val gap; unlikely to fix the structural receptive-field cap on its own. |
| `DROPOUT_STYLE` | 0.5 | — | style already green (priority 5, below focus) | — |
| `DROPOUT_CRITIC` | 0.2 | — | no evidence pointing here this loop (color energy is below focus) | — |
| `RELU_SLOPE` | 0.1 | — | no evidence pointing here | — |
| `Z_WEIGHT` | 0.35 | — | no evidence pointing here | — |
| `GEN_CHANNELS` | [64, 32] | — | no evidence pointing here | — |
| `CRITIC_COLOR_CHANNELS` | [96] | — | no evidence pointing here | — |
| `CRITIC_TEXT_CHANNELS` | [64] | — | no evidence pointing here | — |

Priority-1 risk of the `DROPOUT_EMOJI` bump: **low** — keyword/term Acc are 0.995-1.0, far above their targets; a small regularization increase has essentially no realistic path to dropping them below the 0.90/0.80 hard targets in one step.

### Model Architecture  (avoid unless Model Configuration changes are insufficient)
- **Wider receptive field for full-text (priority 3, this loop's focus)**: e.g. add a 5th dilated block or raise dilation to `[1,2,4,8,16]`, or widen `ENCODER_KERNEL_SIZE` to 5, to push `RECEPTIVE_FIELD` past `MAX_TEXT_LEN` (42). Reason: Model Configuration has no knob that changes receptive field, and the `block_capacity` evidence points specifically at the deepest block's collapse on long text, not at overfitting alone. Expected impact: unverified — plausible but unproven that full receptive-field coverage improves full-sentence retrieval; needs a Modal GPU run. Param-count cost: adding a 5th `[1,2,4,8,16]` block at similar width (~80-100 channels) would grow `encoder conv params` from 126,440 by roughly 25-35% and `TEXT_EMBED_SIZE` (currently 440) proportionally, which also grows both heads — a real wasm-latency cost for an on-device model, should be sized precisely with `uv run python model/config.py` before committing to a shape. **Recommend-only — do not apply.** Gate this on the length-bucketed diagnostic below actually confirming the correlation first; don't spend a GPU run on a hypothesis that hasn't been checked against the report's own eval data yet.

### Training Configuration

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `LR` | 0.01 | — | no evidence pointing here | — |
| `GAN_GEN_LR` | 0.005 | — | color energy below focus this loop | — |
| `GAN_CRITIC_LR` | 0.02 | — | color energy below focus this loop | — |
| `GAN_GEN_MARGIN` | 1.0 | — | no evidence pointing here | — |
| `GRAD_CLIP_GEN` | 1.0 | — | no evidence pointing here | — |
| `GRAD_CLIP_CRITIC` | 1.0 | — | no evidence pointing here | — |
| `INFONCE_TEMP` | 0.7 | — | keyword/term already near-ceiling; no evidence full-text is temperature-limited rather than capacity-limited | — |
| `TASK_BATCH_SIZE` | 512 | — | no evidence pointing here | — |
| `GAN_BATCH_SIZE` | 1024 | — | no evidence pointing here | — |
| `EPOCHS_TASK` | 1500 | — | `MRR/e/val` is plateaued/noisy, not still climbing — more epochs unlikely to help without the dropout/architecture changes above | — |
| `EPOCHS_GAN` | 300 | — | no evidence pointing here | — |
| `VAL_CHECK_INTERVAL` | 100 | — | no evidence pointing here | — |
| `EARLY_STOP_PATIENCE` | 20 | — | checkpointing already looks like it's landing in the plateau region, not cutting off an still-improving run | — |

No Training Configuration change proposed this iteration beyond the Model Configuration `DROPOUT_EMOJI` nudge above.

### Metric Adjustments
- **Wrong signal**: none this iteration.
- **Missing signal**: build `tools/analysis/length_vs_acc.py` (read-only, mutates nothing) — loads `pt/enc.pt` + `pt/emoji.pt` and `data/eval.jsonl`, buckets eval rows by text length (e.g. ≤15 / 16-31 / 32-42 chars — the middle break at the current `RECEPTIVE_FIELD`), and reports Acc@1/@5/@10 per bucket. This directly tests the *Current gaps* "full-text retrieval capped" hypothesis: if accuracy drops sharply past 31 chars, that confirms the receptive-field story and de-risks the Model Architecture recommendation above before it costs a Modal GPU run; if it doesn't, the ceiling is from something else (e.g. overfitting alone, or a harder semantic-composition problem) and the architecture option should be shelved in favor of the `DROPOUT_EMOJI` path.

### Global Goal Adjustments  (goals.yml)
- No change proposed — every target still looks reachable in principle; full-text Acc@5/@10 are within 0.005-0.015 of their hard targets already, and Acc@1's 0.75 target is a real stretch but not implausible given the near-ceiling keyword/term numbers on the same encoder.

### Current Goal Adjustments  (goal/2026-09-12-52017be.yml)
- **Keyword / term acc@{1,5,10}** (priority 1-2, at/above focus, `met`) → held at their global hard targets (already cleared with large margin; no stretch needed, nothing to gain by inventing a higher number).
- **Full-text acc@1** (priority 3, focus, `unmet` but close — cleared the old 0.56 iter target) → stepped to `0.65` (modest step given TB shows a plateau, not fast movement; not stretched to the 0.75 hard target in one loop).
- **Full-text acc@5 / acc@10** (priority 3, focus, `unmet` but within 0.005-0.015 of the hard target) → set directly to the hard targets `0.80` / `0.85` — close enough that a real step-change loop should clear them outright.
- **Color energy, style, max text len, vocabulary** (priority 4-8, below focus) → held at current measured value (or the existing global target where already met), listed in `meta.deferred` — not this loop's fight while priority 3 is open.

## Applied this run

none yet — awaiting the user's pick from the options above
