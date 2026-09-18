# Improvement Plan — 26-09-18-02-11 · 0b1e564

Report: report/26-09-18-02-11-0b1e564/report.html
Prev plan: plans/26-09-17-20-40-b87e841/plan.md
Changed since prev: 9 commits (`git log --oneline b87e841..HEAD`), all `train` runs sweeping the color GAN plus a final `BEST ENCODER` commit that re-tightened `goals.yml` (full-text emoji targets raised, style acc@10 lowered) and dropped `LR_GAN_GEN` 0.005→0.001 for the *next*, not-yet-trained run. Across this window: `ENCODER_CHANNELS` widened `[120,180,120,60]→[120,220,160,60]`, `Z_WEIGHT` 0.2→0.3, `CRITIC_EMBEDDING_SIZE` 64→96, `LOSS_WEIGHT_ENERGY` 0.1→0.9, `LR_GAN_GEN` 0.02→0.001, batch sizes 128/512→1024 fixed (env-driven before), `GRAD_CLIP_GEN` 10→2. All several knobs moved together each run — an uncontrolled sweep, not an ablation, so no single change can be individually credited. `LOSS_WEIGHT_COLOR_REG` was the one color-relevant knob **never touched** in this window.
Best emoji variant: EmojiHead (unchanged)
Loop verdict: **actionable now** — the last loop's checkpoint-monitor fix held (full-text emoji acc@1 0.5945→0.6675, acc@5 0.786→0.829, both now green; acc@10 0.833→0.870, red only because the target was just raised to 0.90). Color energy also improved substantially through the GAN sweep (red 0.270→0.198, green 0.114→0.080 now green, blue 0.164→0.132, dark 0.346→0.248, bright flat) but 3/5 categories are still red, worst on `dark` and `red`. A short diagnostic run against the current `pt/enc.pt`+`pt/gen.pt` (matching this report exactly) found a concrete, reproducible bias behind the remaining gap.

## Current status

| Goal (priority) | Target | Current | Status | Δ vs prev plan |
|---|---|---|---|---|
| Keyword/term emoji acc@1/5/10 (1–2) | see goals.yml | 0.997–1.0 / 0.999–1.0 | 🟢 good | unchanged, still maxed |
| **Full-text emoji acc@1 (1)** | ≥0.65 | **0.6675** | 🟢 good | ↑ from 0.5945 (monitor fix from last loop worked) |
| **Full-text emoji acc@5 (1)** | ≥0.75 | **0.8290** | 🟢 good | ↑ from 0.7860 |
| **Full-text emoji acc@10 (1)** | ≥0.90 (raised this run, was 0.85) | **0.8700** | 🔴 red | ↑ from 0.8330 in absolute terms; red only because target moved up |
| **Color energy · red (2)** | ≤0.10 | **0.1984** | 🔴 red | ↓(better) from 0.2698 — shortfall ratio 2.60× floor, worst gt_accuracy (0.0) |
| Color energy · green (2) | ≤0.10 | 0.0795 | 🟢 good | ↓(better) from 0.1136 — newly green |
| **Color energy · blue (2)** | ≤0.10 | **0.1315** | 🔴 red | ↓(better) from 0.1642 — shortfall ratio 1.99× |
| **Color energy · dark (2)** | ≤0.10 | **0.2479** | 🔴 red | ↓(better) from 0.3458 — still worst absolute distance, shortfall 1.92× |
| Color energy · bright (2) | ≤0.10 | 0.0642 | 🟢 good | flat (was 0.0632) — shortfall ratio only 1.04×, effectively at floor |
| Color energy · global (2) | ≤0.01 | n/a | ⚪ na | **still unwired** — `tools/report.py`'s `cards` dict never sets an `"energy"` key, so this top-line priority-2 goal has never once been graded |
| Style acc@1/5/10 (3) | ≥0.60/0.80/0.87 (target lowered this run, was 0.90) | 0.728/0.960/0.984 | 🟢 good | improved, well clear either way |
| Max text len / vocab size / coverage (4) | see goals.yml | 42 / ≥700 / 100% | 🟢 good | unchanged |

Priority-1 regression gate: clean — full-text emoji acc@1/@5 both improved and crossed target; acc@10 improved in absolute terms, red only due to a target bump made in this same commit.

## Gaps

- **`dark` and `red` are the worst color categories, and a diagnostic run against the exact checkpoints behind this report (`pt/enc.pt` + `pt/gen.pt`, both timestamped 02:11:21, matching the report stamp) reproduces a concrete, systematic bias, not a diversity problem.** Conditioning the frozen `ColorGen` on 25 real per-category texts × 32 fresh `z` draws each and comparing mean Oklab lightness/chroma to gold:

  | category | gen L | gold L | gen chroma | gold chroma | z-sample std (diversity) |
  |---|---|---|---|---|---|
  | red | +0.820 | +0.646 | 0.047 | 0.108 | 0.033 |
  | dark | +0.745 | +0.490 | 0.051 | 0.031 | 0.044 |
  | blue | +0.822 | +0.777 | 0.055 | 0.045 | 0.037 |
  | green | +0.867 | +0.836 | 0.050 | 0.051 | 0.029 |
  | bright | +0.834 | +0.897 | 0.069 | 0.105 | 0.040 |

  `dark` is generated far too **light** (L off by 0.255, the largest gap of any category — this alone explains it being the worst absolute distance). `red` is generated far too **desaturated** (chroma 0.047 vs. gold 0.108 — a >2x undershoot). `green`/`bright` — the two categories already at or near their goal — show close L/chroma match to gold. The per-text `z`-sample spread (last column) is roughly uniform across categories (0.029–0.044), ruling out "the 8-sample eval just isn't diverse enough to land near the gold example" as the explanation; this is a **mean-conditioning bias**, not a sampling-diversity problem.
- **The same bias, more severely, shows up in the plain MSE `ColorRegressor` head that shares the frozen encoder's embedding** (not shipped for inference, but trained jointly with the encoder in stage 1 and diagnostic of what the embedding carries): predicted chroma is low across **every** category (0.023–0.036 vs. gold 0.031–0.086, worst relative undershoot on `red`/`bright`), and predicted lightness barely varies across categories (0.628–0.652, a 0.024 range) versus gold's 0.10 range (dark 0.564 vs. bright 0.665). A plain regressor collapsing toward the conditional mean on a multimodal target is expected in isolation, but the *direction* matches the GAN's own bias (both undershoot `dark`'s lightness the same way), pointing at a shared root: the encoder embedding `ColorGen` conditions on doesn't carry much category-discriminating lightness/chroma signal for full sentences.
- **`block_capacity["ColorRegressor"]` (already in every report, previously unexamined for color) confirms the embedding is the likely bottleneck, specifically for the block a controlled architecture change would target.** Its `eval`-split contribution (real eval.jsonl full sentences, not keyword/term) collapses across depth: block 0 → 61.0%, block 1 → 86.0%, block 2 → 43.2%, **block 3 (the deepest, dilation-8, 60-channel block) → 16.1%** — versus 53.6%/37.8% for the same block on `keywords`/`terms` respectively. `act_rms` on `eval` drops to 0.011 at block 3, an order of magnitude below `keywords`' 0.140. The deepest, longest-context block is nearly dead for color on real full-sentence input specifically — plausibly why nuance-dependent full sentences (e.g. "paprika and cinnamon", "Boots soaked through; red clay") land on safe, desaturated/light colors regardless of category, while `green`/`bright` — categories whose color words tend to appear more literally in-sentence — don't need that deep integration and score fine.
- **`LOSS_WEIGHT_COLOR_REG` (the one knob that shapes how hard the shared encoder is pushed to carry color signal at all) has been left at `1` through the entire 9-commit GAN sweep**, while every other color-adjacent knob was moved multiple times. This is the natural next lever to test before considering a block-3 architecture change, per this skill's ordering rule (config-level knob first, quantified architecture change only if that proves insufficient).
- **`LOSS_WEIGHT_ENERGY` was already swept 0.1→0.9 in this exact window, with color energy improving on 4/5 categories over the same span.** Reversing that trend (e.g. lowering it to rebalance toward the critic's conditional score, which was this session's first hypothesis before checking history) is **not supported** by the evidence on hand and contradicts a real, if confounded, directional result already in motion. Dropped as an action.
- **Priority-2 "global" color-energy goal (≤0.01) has literally never been measured** — `tools/report.py::_section_cards` populates `cards["per_color"]` but never a top-level `cards["energy"]`, and `tools/report.py:635` reads exactly that missing key. This is a pure reporting gap (zero training risk to fix) that's been silently masking whether the aggregate goal is anywhere close.

## Proposed actions

Ranked; pick one or more via the menu below.

### 1. Bump `LOSS_WEIGHT_COLOR_REG` — directly implementable, primary recommendation
- **File/knob**: `model/config.py:76`, `LOSS_WEIGHT_COLOR_REG = 1` → **`3`**.
- **Reasoning**: this is the only loss term giving the shared encoder (trained in stage 1, then frozen for the GAN) explicit pressure to preserve Oklab lightness/chroma information. The diagnostics above show that signal is currently weak specifically on full sentences at the deepest block — a 3x weight increase is a direct, low-cost way to test whether more gradient pressure recovers it before reaching for an architecture change.
- **Estimated effect**: should sharpen the encoder embedding's color-discriminating signal for full sentences, which the frozen GAN generator conditions on — plausibly narrowing the `dark`/`red` lightness/chroma gap. Direction only; magnitude needs the next `train --local` + report.
- **Risk to priority 1**: moderate — this loss shares the same encoder trunk as emoji/style. A 3x (not 10x) step keeps it modest; watch `full_text/acc@1/val` (the checkpoint monitor) and `MRR/e/val` for any drop. If full-text emoji regresses, back off to `~1.5–2` next loop rather than reverting fully.
- Needs a real `train --local` run to evaluate (encoder stage retrains; GAN stage then reconditions on the new frozen embedding) — not evaluable from existing checkpoints.

### 2. Wire the missing color-energy report metrics — directly implementable, zero training risk
- **File**: `tools/report.py`, `_section_cards()` (~line 946) and `_stats()` (~line 926).
- **Change**: set `cards["energy"]` from the existing `per_color["all"]["gt_mean_distance"]` (or an equivalent unweighted-mean-of-`gt_mean_distance`-across-categories, whichever this skill's philosophy would call "global" — pick the one already implied by `goals.yml`'s single `global` scalar) so `tools/report.py:635`'s `energy_global = cards.get("energy")` stops being permanently `None`. Also add the gen-vs-gold mean Oklab `L`/chroma columns (computed the same way as this session's ad hoc diagnostic, reusing `rgb_to_oklab` and the existing per-row `palettes`/gold9 arrays already computed in `_section_cards`) to the per-color cards table, so the lightness/chroma bias found this loop is visible in every future report without re-deriving it by hand.
- **Estimated effect**: no model change — purely makes an already-defined priority-2 goal and this loop's root-cause finding visible going forward.
- **Risk to priority 1**: none — report-only, doesn't touch training.
- Needs: `uv run ruff check tools/report.py`, then a `tools/report.py --pt pt` re-run against the existing checkpoints to confirm it renders before trusting it in the next real report.

### 3. Recommend-only — encoder block-3 capacity (do not apply this loop)
- **If action 1 does not close the `dark`/`red` gap**, the next candidate is architecture-level: block 3 (`ENCODER_CHANNELS[3] = 60`, dilation 8) is the block whose `ColorRegressor` contribution collapses hardest on real full-sentence input (16.1%, see Gaps). Widening it, or adding a residual/skip path so deeper blocks retain more signal for the color-relevant subspace specifically, could plausibly help — but this needs a quantified param-count/latency cost via `model/config.py`, a GPU run to validate, and per this skill's rules must wait until the config-level lever (action 1) has been tried and shown insufficient.

## `goals.yml` adjustments

No change proposed. The `global` color-energy target (≤0.01) can't be judged reachable or not until action 2 wires it up; the five per-category targets are unevenly cleared (2/5 good) and none looks structurally unreachable — leave as is.

## Applied this run

User picked action 2 only (report wiring; the `LOSS_WEIGHT_COLOR_REG` config nudge was not applied).

**`tools/report.py`**:
- Added `_l_chroma(vals9)` next to `_card_distance` — mean Oklab lightness and chroma over a 3-swatch palette.
- `_section_cards()`: each output row now also carries `gen_l`/`gold_l`/`gen_chroma`/`gold_chroma` (from the existing `flat` top-`k` palette and `gold9`, no new model calls).
- `_stats()`: added `l_bias`/`chroma_bias` (mean signed `gen − gold`) per category and for `all`.
- The returned cards dict now sets `"energy"` to `per_color["all"]["gt_mean_distance"]`, which is exactly the value `tools/report.py`'s goal-grading code was already trying to read via `cards.get("energy")` — the "Color energy · global" goal is no longer permanently `na`.
- `_cards_html()`: added `ΔL`/`Δchroma` columns to the colour-distance table, with a one-line note on sign convention (positive = model runs lighter/more saturated than gold).

```
$ uv run ruff check tools/report.py
All checks passed!
$ uv run ruff format --check tools/report.py
(pre-existing unformatted lines elsewhere in the file, none touched by this change)
```

Verified end-to-end against the existing `pt/` checkpoints (`uv run python tools/report.py --pt pt`, output discarded after inspection — it wasn't produced by a real training run so shouldn't be kept as a dated report): `cards["energy"]` came out `0.1443` (matches `per_color["all"]["gt_mean_distance"]`, now graded red against the `≤0.01` global target instead of `na`), and the new bias columns reproduced this loop's diagnostic finding directionally — `dark` `l_bias = +0.058` (generates lighter than gold), `red`/`bright` `chroma_bias` negative (desaturated vs. gold), `green`/`blue` near zero. Smaller in magnitude than the ad hoc 32-`z`-sample diagnostic in the Gaps section since this uses the single displayed top-`k` sample per row rather than an average, but consistent in sign and ranking.

The `LOSS_WEIGHT_COLOR_REG` bump (action 1) and the block-3 architecture note (action 3) remain open for a future loop — re-open once the user wants to test them with a real `train --local` run.
