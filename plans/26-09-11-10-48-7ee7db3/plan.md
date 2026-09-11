# Improvement Plan — 26-09-11-10-48 · 7ee7db3

Report: report/26-09-11-10-48-7ee7db3/report.html
Goals file written: goal/2026-09-11-7ee7db3.yml
Prev plan: plans/26-09-11-02-48-d577316/plan.md
Changed since prev: `d929270` regen restored the 400-emoji vocab (the keyword-forcing fix's actual data run) · `af286c2` reverted the encoder back to a single `[200]`/`[1]` block · `6dd47cc` skill update (upsampling last resort) · three unplanned single-commit training iterations (`06b0cf3` `CLDR_WEIGHT` 0.01→0.1, `3e6949c` encoder → two dilated blocks `[100,100]`/`[1,2]` + `CLDR_WEIGHT` 0.1→0.08, `7ee7db3` same config, first **full** `train --local` of the three — the other two were `--heads`-only dev runs, so their `.pt`/report don't reflect a shipped GAN/style).
Best emoji variant: Fusion (single detached per-emoji-diagonal `FusionHead` — no gate/gain/mix sweep anymore)
Loop verdict: **step-change** — priority 1 (exact-keyword fusion Acc@1) is amber and within 0.04 of its hard target; a single-variable `CLDR_WEIGHT` revert isolates a plausible cause of its recent drift, paired with a read-only diagnostic that explains the larger, more actionable finding: the fusion model gives up ~0.107 Acc@1 versus the raw keyword search it's built from.

## Abstract

The last three training runs (all after the previous plan) changed `CLDR_WEIGHT` and the encoder's channel/dilation shape in the same two commits, so their combined effect on exact-keyword fusion Acc@1 can't be cleanly attributed — the metric moved 0.896 → 0.885 → 0.858 across them, only the last of which was a full `train --local` (the other two were partial `--heads` dev runs with a stale GAN/style, confirmed by their `provenance.issues`). TensorBoard for the current run shows every head's val metric peaking cleanly around step 14,000 with the checkpoint correctly saved at that peak (`MRR/fusion/val` 0.653 → 0.640 by early-stop, `MRR/e/val` 0.636, `MRR/s/val` 0.520 → 0.492) — training dynamics look healthy, not under- or over-trained beyond normal early-stop drift. The most concrete, actionable finding in `report.json` is that the standalone keyword search already hits 0.965 Acc@1 on the CLDR probe, but the fusion model — the thing every emoji-prediction goal actually grades — only reaches 0.858, a 0.107 gap the report can't currently explain (no goal is graded on the standalone kw search, and `FusionHead`'s per-emoji weights are a single static compromise across full-text and keyword-style rows, `fusion/w_search_mean` ≈ 2.45 vs `fusion/w_dl_mean` ≈ 0.84). Full-text Acc@1 has stayed flat (~0.54–0.57) across every config combination tried so far, and the encoder's receptive field is only 7 of 32 `MAX_TEXT_LEN` chars — architecture headroom, but below this iteration's priority-1 focus. This plan proposes reverting `CLDR_WEIGHT` to isolate its effect on the priority-1 regression, and building a read-only fusion-vs-kw regression diagnostic to explain the 0.107 gap before proposing an architecture fix for it.

## Current goals & status

| Goal (priority) | Global target (goals.yml) | Current target (this iteration's goal/*.yml) | Current value | Status | Δ vs prev plan |
|---|---|---|---|---|---|
| Exact keyword acc@1 (1) | ≥0.90 | ≥0.87 | 0.8576 | 🟡 amber | 0.7064 → 0.8576 (+0.1512, vocab 287→400 — not fully comparable) |
| Exact keyword acc@5 (1) | ≥0.92 | ≥0.95 | 0.9510 | 🟢 good | 0.8506 → 0.9510 (+0.1004, same caveat) |
| Exact keyword acc@10 (1) | ≥0.95 | ≥0.96 | 0.9659 | 🟢 good | 0.8901 → 0.9659 (+0.0757, same caveat) |
| Full-text emoji acc@1 (2) | ≥0.80 | ≥0.55 | 0.5478 | 🔴 red | 0.5496 → 0.5478 (−0.0018) |
| Full-text emoji acc@5 (2) | ≥0.85 | ≥0.69 | 0.6921 | 🔴 red | 0.6994 → 0.6921 (−0.0073) |
| Full-text emoji acc@10 (2) | ≥0.90 | ≥0.74 | 0.7427 | 🔴 red | 0.7481 → 0.7427 (−0.0054) |
| Fuzzy keyword acc@1 (3) | ≥0.80 | ≥0.0 | n/a | ⚪ na | unwired, unchanged |
| Color energy · global (4) | ≤0.01 | ≤0.05 | n/a | ⚪ na | unwired, unchanged |
| Color energy · red/green/blue/dark/bright (4) | ≤0.01 each | held at current | 0.198/0.122/0.128/0.344/0.049 | 🔴 red | prev report had `cards` off (partial dev runs) — newly measured |
| Style acc@1 (5) | ≥0.80 | ≥0.49 | 0.4880 | 🔴 red | prev report had `cards` off — newly measured |
| Style acc@5 (5) | ≥0.85 | ≥0.73 | 0.7280 | 🔴 red | newly measured |
| Style acc@10 (5) | ≥0.90 | ≥0.87 | 0.9360 | 🟢 good | newly measured |
| Max text len (6) | ≥42 | ≥32 | 32 | 🔴 red | unchanged |
| Emoji vocab size (7) | ≥700 | ≥400 | 400 | 🔴 red | 287 → 400 (+113, from a plain `regen`, not new upsampling) |
| Vocab coverage — groups (8) | all ≥ target | held, not gated | 40/100 pass | ⚪ deferred | 18/100 → 40/100 (incidental, from the vocab-size jump, not targeted work) |

Priority-1 regression gate — fusion model on exact keywords vs prev plan's actual: **PASS** (Acc@1 0.7064 → 0.8576, net gain from the keyword-forcing fix, though not a clean before/after since the vocab also grew 287→400 in the same window). **Watch item**: within this window, three back-to-back unplanned iterations (all vocab=400, so *comparable to each other*) show Acc@1 declining 0.896 → 0.885 → 0.858 as `CLDR_WEIGHT` dropped 0.1→0.08 and the encoder changed shape — flagged below, not yet a confirmed cause.

## Current gaps

- **Config regression risk, confounded (priority 1)** — `CLDR_WEIGHT` and `ENCODER_CHANNELS`/`ENCODER_DILATION` both changed in the same commit (`3e6949c`), and the very next commit (`7ee7db3`) reran the identical config as a full pipeline vs. `3e6949c`'s partial `--heads` run — so three different exact-keyword Acc@1 readings (0.896/0.885/0.858) reflect at least two conflated variables plus a run-type difference. The report and TensorBoard can't isolate which change (or run-to-run noise from the unseeded CLDR mix) caused the drift. Evidence: `report/26-09-11-09-29-06b0cf3`, `-09-47-3e6949c`, `-10-48-7ee7db3` `status.goals` rows; `model/config.py` git history above.
- **Fusion underperforms its own kw input (priority 1, the clearest lever)** — `report.json.keyword.exact.acc_at_k[0]` = 0.9652 (standalone inverted-index search) vs `keyword.fusion.acc_at_k[0]` = 0.8576 (the graded metric) on the same 1552 CLDR rows — a 0.107 Acc@1 gap. `FusionHead` is a single static per-emoji diagonal (`w_dl`≈0.84, `w_search`≈2.45 mean, from TensorBoard `fusion/w_dl_mean`/`fusion/w_search_mean`) trained jointly across full-text and CLDR-mixed rows, so it can't be a pure keyword-passthrough where kw is confident and pure-DL where kw is absent — it's a global compromise. This is a diagnostic gap: no existing report field shows *which* CLDR rows the fusion gets wrong that kw alone gets right, or whether it's concentrated in a few emoji classes.
- **Full-text Acc@1 flat across every config tried (priority 2, below focus)** — 0.5496 → 0.5443 → 0.5652 → 0.5478 across four different `CLDR_WEIGHT`/encoder combinations, no clear direction. `uv run python model/config.py` shows `RECEPTIVE_FIELD` = 7 of `MAX_TEXT_LEN` = 32 chars (`partial coverage`) — the encoder can only see ~2 words of context at any position, which plausibly caps full-sentence understanding regardless of dropout/CLDR-mix tuning. This is architecture headroom, not this iteration's fight (priority 2, below the priority-1 focus).
- **Style Acc@1 red despite Acc@10 already green (priority 5, below focus)** — `cards.style_acc_at_k` = [0.488, 0.728, 0.808, 0.824, 0.872, 0.872, 0.896, 0.920, 0.936, 0.936]: the right style is almost always in the top few candidates (Acc@5 = 0.728) but rarely ranked first. TensorBoard shows `MRR/s/val` peaking at 0.520 (step 13763) then declining to 0.492 by early-stop — a mild late-training dip, but the checkpoint used for `cards` is the best-`MRR/fusion/val` one (step 14378), not the best-`MRR/s/val` one, so some of this is an artifact of checkpointing on the fusion metric, not necessarily style overfitting. Below focus this iteration.

## Proposed actions

### Data
- **Regen params**: no change proposed. `--min-count`/`--max-count`/`--n` weren't touched this iteration; the vocab jump 287→400 already came from a plain `regen` restoring the documented default, not a parameter change.
- **Upsampling**: no upsample proposed. Neither open goal (exact-keyword amber, full-text red) shows a coverage hole — the exact-keyword gap is the fusion combiner underperforming its own kw input (a combiner/config question, not a missing concept), and full-text's flatness tracks the encoder's small receptive field, not a labeling gap. Ruled out before reaching for `bun run upsample`.
- **`CLDR_WEIGHT`**: current `0.08` → proposed `0.10`. Evidence: of the three vocab=400 runs (the only mutually comparable set), the one with `CLDR_WEIGHT=0.1` (`06b0cf3`) had the highest exact-keyword Acc@1 (0.896); the two at `0.08` scored lower (0.885, 0.858) — though confounded with the same-commit encoder change and a dev-vs-full-run difference, so this is a hypothesis test, not a proven fix. Priority-1 risk: **low** — raising `CLDR_WEIGHT` increases exposure to exactly the CLDR keyword-style rows priority 1 is graded on; the risk is entirely one-sided (upside for priority 1, small downside risk to full-text training mix, which is already flat and below focus). Cost: one-line config edit, needs a full retrain to evaluate.
- **Other data adjustments**: none this iteration.

### Model Configuration

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `EMOJI_EMBED_SIZE` | 42 | — | no evidence pointing here | — |
| `STYLE_EMBED_SIZE` | 16 | — | no evidence pointing here | — |
| `DROPOUT_EMOJI` | 0.1 | — | no evidence pointing here | — |
| `DROPOUT_STYLE` | 0.5 | — | `MRR/s/val` shows a mild post-peak dip (0.520→0.492), but style is below this iteration's focus and the checkpoint is selected on `MRR/fusion/val`, not `MRR/s/val` — not enough evidence to act yet | — |
| `DROPOUT_CRITIC` | 0.2 | — | no evidence pointing here | — |
| `RELU_SLOPE` | 0.1 | — | no evidence pointing here | — |
| `Z_WEIGHT` | 0.35 | — | no evidence pointing here | — |
| `GEN_CHANNELS` | [64, 32] | — | no evidence pointing here | — |
| `CRITIC_COLOR_CHANNELS` | [96] | — | no evidence pointing here | — |
| `CRITIC_TEXT_CHANNELS` | [64] | — | no evidence pointing here | — |

No Model Configuration change proposed this iteration — the identified levers (`CLDR_WEIGHT`, the fusion combiner shape) live in Data and Model Architecture respectively.

### Model Architecture  (avoid unless Model Configuration changes are insufficient)
- **Wider receptive field for full-text (priority 2, not this iteration's focus)**: add a third dilated encoder block, e.g. `ENCODER_CHANNELS=[100,100,100]` / `ENCODER_DILATION=[1,2,4]`, raising `RECEPTIVE_FIELD` from 7 to 15 of 32 `MAX_TEXT_LEN` chars (`uv run python model/config.py` math: `1 + (kernel-1)*sum(dilation)`). Reason: Model Configuration has no knob that changes receptive field — this is a structural fix for the flat full-text Acc@1. Expected impact: unverified, needs a Modal GPU run to test — plausible but unproven that more context per position helps full-sentence retrieval. Param-count cost: encoder conv params go from 35,000 to 65,100 (+86%), `TEXT_EMBED_SIZE` grows 200→300 which also grows `emoji_head`/`style_head` params proportionally — a real wasm-latency cost for an on-device model. Recommend-only — do not apply; below this iteration's priority-1 focus, and needs a GPU run to validate.
- **Row-adaptive fusion (priority 1, the more direct fix for the 0.107 kw-vs-fusion gap)**: `FusionHead`'s per-emoji diagonal is a single static weight applied to every row, so it can't fully favor kw on keyword-style rows while favoring the DL signal on full-text rows — the current split (`w_search_mean`≈2.45 vs `w_dl_mean`≈0.84) is a compromise, not a per-row gate. **Confirmed this loop** (see `tools/analysis/fusion_vs_kw.py` output under *Applied this run*): a handful of classes (`⚙️` `w_search=-13.97`, `🔎` `w_search=-9.23`, `🏘️` `w_search=-8.76`) have gone strongly *negative*, actively inverting a maxed exact-kw hit and dropping a top-15 raw candidate to dead last — not a data-volume issue (156-383 training rows each). Two candidate fixes, both touch `model/model.py`'s loss shape: (a) a small L2 penalty or a hard clamp on `w_search`/`w_dl` per class to stop one noisy class from swinging that far from the ≈1.0 init; (b) a real row-level gate (e.g. conditioned on the row's max kw strength) so the combiner can trust kw on clean keyword-style rows without needing one global per-class weight to cover every row shape. Recommend-only — do not apply; needs a retrain to validate and is squarely in the guardrail's "loss shape" no-touch list.

### Training Configuration

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `LR` | 0.01 | — | no evidence pointing here | — |
| `GAN_GEN_LR` | 0.01 | — | no evidence pointing here | — |
| `GAN_CRITIC_LR` | 0.02 | — | no evidence pointing here | — |
| `GAN_GEN_MARGIN` | 1.0 | — | no evidence pointing here | — |
| `GRAD_CLIP_GEN` | 1.0 | — | no evidence pointing here | — |
| `GRAD_CLIP_CRITIC` | 10.0 | — | no evidence pointing here | — |
| `INFONCE_TEMP` | 0.7 | — | no evidence pointing here | — |
| `TASK_BATCH_SIZE` | 128 | — | no evidence pointing here | — |
| `GAN_BATCH_SIZE` | 512 | — | no evidence pointing here | — |
| `EPOCHS_TASK` | 1500 | — | no evidence pointing here | — |
| `EPOCHS_GAN` | 300 | — | no evidence pointing here | — |
| `VAL_CHECK_INTERVAL` | 100 | — | no evidence pointing here | — |
| `EARLY_STOP_PATIENCE` | 20 | — | checkpointing already lands correctly on the `MRR/fusion/val` peak (step 14378 of 16433) — no sign of under/over-training past that | — |

No Training Configuration change proposed this iteration.

### Metric Adjustments
- **Wrong signal**: none this iteration.
- **Missing signal**: build `tools/analysis/fusion_vs_kw.py` (read-only, mutates nothing) — for every `data/cldr.jsonl` row, compare the fusion model's rank of the target emoji against the standalone kw search's rank, using `pt/enc.pt`/`pt/emoji.pt`/`pt/emoji_embed.pt`/`pt/fusion.pt` + `web/public/kwproj.json`. Report: how many rows regress (fusion worse than kw alone) vs. improve, whether regressions cluster in specific emoji classes or in rows with weak/ambiguous kw matches, and the `kw`-strength distribution for regressed rows. This directly targets the *Current gaps* "fusion underperforms its own kw input" finding and is the evidence a row-adaptive-gate proposal (Model Architecture, above) would need before being worth building for real.

### Global Goal Adjustments  (goals.yml)
- No change proposed — every target in `goals.yml` still looks reachable in principle (exact-keyword is within 0.04 of 0.90; nothing suggests a target is structurally unreachable yet).

### Current Goal Adjustments  (goal/2026-09-11-7ee7db3.yml)
- **Exact keyword acc@1** (priority 1, focus, last iteration `met` its 0.72 target and rose to 0.8576) → stepped to `0.87` — a small step above current given the 0.90 hard target is close but the metric has been drifting down for two runs; not stretched to 0.90 directly.
- **Exact keyword acc@5/@10** (priority 1, focus, already clear their hard targets) → held just above current (0.95/0.96) rather than pushed further — no headroom concern, priority-1's real fight is acc@1.
- **Full-text acc@1/@5/@10** (priority 2, below focus, `unmet`) → held at current (0.55/0.69/0.74), each a hair above the measured value, never below it. Listed in `meta.deferred`.
- **Fuzzy keyword acc@1** (priority 3, below focus, unwired) → held at `0.0` until the uFuzzy `kw` sidecar exists. Listed in `meta.deferred`.
- **Color energy (global + per-color)** (priority 4, below focus) → global held at a placeholder `0.05` (unwired, `cards.energy` doesn't exist yet); per-color held at current measured `gt_mean_distance` (red 0.20, green 0.13, blue 0.13, dark 0.35, bright 0.05). Listed in `meta.deferred`.
- **Style acc@1/@5/@10** (priority 5, below focus) → held at current (0.49/0.73/0.87). Listed in `meta.deferred`.
- **Max text len** (priority 6, below focus) → held at current `config.MAX_TEXT_LEN` = 32. Listed in `meta.deferred`.
- **Vocabulary size** (priority 7, below focus) → held at current `400`. Listed in `meta.deferred`.
- **Vocabulary coverage** (priority 8, below focus, floor gate) → not gated this iteration (every higher-priority goal is still open); omitted from the goal file's `vocabulary.coverage` branch entirely, per `goal/README.md`'s "only gate the groups this loop targets."

## Applied this run

User picked **Both**.

- **`CLDR_WEIGHT` 0.08 → 0.1** (`model/config.py`). Verification: `uv run ruff check model/config.py` clean; `uv run ruff format --check model/config.py` reports the same pre-existing formatting diffs the file already had before this edit (unrelated to the one-line value change, left alone). `model/test_runmeta.py`, `model/test_train_cli.py`, `tools/test_report.py` all `ok`. Not evaluated yet — needs the next full `train --local`.
- **`tools/analysis/fusion_vs_kw.py`** (new, read-only — loads `pt/*.pt` + `data/cldr.jsonl` + `web/public/kwproj.json`, mutates nothing). Verification: `uv run ruff check` / `ruff format --check` clean; ran it against the current `pt/` (`uv run python tools/analysis/fusion_vs_kw.py --pt pt`):
  - `rows: 1552  fusion worse: 197  fusion better: 43  tied: 1312` — `acc@1  kw-exact: 0.9652  fusion: 0.8576` (matches `report.json`'s `keyword.exact`/`keyword.fusion`, confirming the script's ranking logic is consistent with `tools/report.py`'s).
  - The regression is **not spread evenly** — it's concentrated in a handful of emoji classes, several of which collapse to the *worst possible* fused rank (400 of 400) despite a maxed-out exact-kw hit (`kw_strength=1.15`, the `PRIMARY_BONUS`-boosted top posting): `⚙️` (`cog`/`cogwheel`/`gear`), `🔎` (`search`/`magnifying`/`contact`/`tilted`), `🏘️` (`houses`). A second tier collapses to double/triple-digit ranks: `⏱️` (`stopwatch`, rank 162), `😣` (`concentration`/`persevere`/`concentrate`, ranks 115-213), `🏡` (`ranch`, rank 79).
  - **Root cause confirmed, not just correlated**: this is **per-class miscalibration in `FusionHead`'s diagonal weights**, and not a training-data-volume problem — `data/train.jsonl` carries 156-383 rows for each of these emoji, not obviously under-represented. Inspecting `pt/fusion.pt` directly: `⚙️` has `w_search = -13.97`, `🔎` has `w_search = -9.23`, `🏘️` has `w_search = -8.76` (`w_dl` ≈ 0.8-0.9, unremarkable for all three). For `"cog"` → `⚙️`: the raw `EmojiHead` logit alone already ranks it 14th of 400 (a perfectly recoverable position), but the maxed exact-kw hit (`1.15`, from `PRIMARY_BONUS`) times `w_search = -13.97` subtracts ~16 from the fused score — the fusion model has learned to **actively invert** the kw signal for these specific classes, not just under-weight it. Plausible mechanism: in the full-text (non-CLDR) training rows that dominate each batch, an incidental kw hit on one of these words' tokens more often points at the *wrong* label than the right one (noisy/ambiguous `ii.json` postings for words like "search"/"tilted"/"houses" in ordinary sentences), and the shared per-emoji `w_search[e]` has no way to tell "this is a clean CLDR-style keyword row" from "this kw hit is incidental noise" — it optimizes one static weight against both. This sharpens the Model Architecture note above: the fix is not more data for these classes, it's giving `FusionHead` a way to trust kw conditionally (a row-level confidence gate, or at minimum an L2/clamp regularizer on `w_search` to keep a single noisy class from swinging that negative) — still recommend-only (touches `model/model.py`'s loss shape), but now backed by the exact mechanism instead of a guess.
