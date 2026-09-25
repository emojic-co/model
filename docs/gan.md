# GAN review (2026-09-25)

This note summarizes the current GAN state in this repo, why results are weak, and what to try next.

## 1) Current status

## Product-level color quality (latest report)

From [report/26-09-25-04-31-150c45e/report.json](../report/26-09-25-04-31-150c45e/report.json):

- Overall `cards.per_color.all.pure_accuracy`: **0.40**
- Overall `cards.per_color.all.pure_mean_distance`: **0.3597** (lower is better)

Per color (`pure_accuracy`):

- red: **0.60**
- green: **0.36**
- blue: **0.00**
- dark: **0.04**
- bright: **1.00**

So the model is currently good at `bright`, somewhat usable on `red`, weak on `green`, and effectively failing on `blue` and `dark`.

## Color keyword probe (latest report)

Also from [report/26-09-25-04-31-150c45e/report.json](../report/26-09-25-04-31-150c45e/report.json), `color_keywords.rows` (target is `0.1`, lower is better):

- red 0.653
- blue 0.271
- bright 0.365
- chocolate 0.392
- coffee 0.273
- dark 0.963
- gold 0.403
- green 0.264
- lemon 0.432

All are red-status against target.

## TensorBoard GAN logs (latest run)

From [runs/20260925-041836-92992b9/gan](../runs/20260925-041836-92992b9/gan):

- `gan/energy/val`: ~**0.5987 → 0.2177**
- `gan/energy/train`: ~**0.6766 → 0.2070**
- `gan/color_auroc`: ends high (~**0.881**)
- `gan/cond_auroc_gen`: moderate (~**0.663**)

Interpretation: critics separate real/fake reasonably, but that does not translate to good per-color card behavior.

## Recent instability

From recent local GAN runs under [runs](../runs):

- Early runs today reached low `gan/energy/val` (~0.037–0.057)
- Later runs degraded to ~0.18–0.22
- Corresponding report card quality for recent generator SHAs remains poor (`all.pure_accuracy` roughly 0.21–0.40)

So current training is unstable, and energy alone does not guarantee good semantic color control.

---

## 2) Main problems

## Problem A — Objective mismatch (distribution realism vs semantic color intent)

Training optimizes mostly/only adversarial critic score, while success is judged by semantic buckets (`red/green/blue/dark/bright`) and keyword prompts.

In current code:

- Generator loss in [model/train.py](../model/train.py) combines:
  - `GAN_LOSS_COND * loss_gen_critic`
  - `GAN_LOSS_ENERGY * loss_energy`
  - `GAN_LOSS_COLOR * loss_gen_color_critic`
- Current config in [model/config.py](../model/config.py) sets:
  - `GAN_LOSS_COND = 1`
  - `GAN_LOSS_ENERGY = 0`
  - `GAN_LOSS_COLOR = 0`

So generator is effectively trained only through the conditional critic term. That can produce globally plausible palettes but weak target-color steering.

## Problem B — Critic success is not enough for downstream goal

`gan/color_auroc` can be high while `blue`/`dark` accuracy is near zero. This is visible in the latest run/report pair.

This indicates critic discrimination is not aligned tightly enough with the end metric.

## Problem C — Systematic bias toward light/pastel outputs

`gen_sensitivity.swatches` in the latest report shows many outputs clustered around muted light palettes, even for prompts that should be dark/saturated.

Per-color bias also supports this:

- blue never hits threshold in latest report
- dark nearly always misses
- bright always hits

The model appears to have learned a safe high-luminance mode.

## Problem D — Training instability across nearby runs

Recent runs in [runs](../runs) show a clear shift from low val-energy to much worse val-energy in a short sequence of experiments. That suggests hyperparameter sensitivity and/or weak optimization dynamics.

## Problem E — Selection metric is incomplete

Early stopping/checkpointing for stage-2 is based on `gan/energy/val` only. But observed failures are color-target specific (`blue`, `dark`) and keyword-conditional. A single global energy score does not capture these failures reliably.

---

## 3) Likely technical causes in current implementation

1. **Generator objective is under-constrained** for color intent (only conditional critic term active).
2. **Small generator capacity** (`GEN_HIDDEN_SIZE = 32`) may limit modeling of multimodal conditional color distributions.
3. **Optimizer choice**: stage-2 currently uses SGD for generator and critics in [model/train.py](../model/train.py) (Adam blocks are commented out), which may hurt stability for this adversarial setup.
4. **Weak color-specific supervision signal** in stage-2 selection (global energy only, no per-color guardrails).
5. **Possible data imbalance / ambiguity** in color-text mapping (many prompts do not strongly constrain hue), encouraging collapse toward safe neutrals.

---

## 4) Recommended solutions

## Priority 0 (fast, high impact)

1. **Re-enable multi-term generator loss**
   - Set non-zero `GAN_LOSS_ENERGY` and `GAN_LOSS_COLOR`.
   - Start with conservative weights (example):
     - `GAN_LOSS_COND = 1.0`
     - `GAN_LOSS_COLOR = 0.1`
     - `GAN_LOSS_ENERGY = 0.01`
   - This restores pressure toward distribution realism and unconditional palette quality.

2. **Switch GAN optimizers to Adam/AdamW with TTUR**
   - Replace SGD in stage-2 with Adam/AdamW.
   - Typical start: generator lr slightly lower than critics, betas `(0.5, 0.999)`.
   - Expect smoother convergence and less run-to-run drift.

3. **Add checkpoint metric guardrails**
   - Keep `gan/energy/val`, but also require minimum `cards` quality on a fixed mini-probe (`blue`, `dark`, `green`) before accepting best checkpoint.
   - Prevent selecting “energy-good but semantics-bad” models.

## Priority 1 (medium effort)

4. **Color-focused curriculum / reweighting in stage-2 batches**
   - Increase sampling of hard slices (`blue`, `dark`, and low-hit keyword subsets).
   - Add batch-level balancing by color bucket.

5. **Add explicit semantic color auxiliary losses**
   - Build simple differentiable losses in OKLab:
     - hue-distance loss for hue-labeled slices (`red/green/blue`)
     - luminance target loss for `dark`/`bright`
   - Use small weights so GAN behavior stays primary, semantics become guided.

6. **Increase generator capacity modestly**
   - Try `GEN_HIDDEN_SIZE` 64 or 80.
   - Keep training budget similar; evaluate if blue/dark fidelity improves without hurting diversity.

## Priority 2 (larger changes)

7. **Train/predict in OKLab directly**
   - Output in perceptual space and convert to sRGB at decode time.
   - Usually improves stability of hue/lightness control vs raw sRGB offsets.

8. **Critic architecture/negative strategy upgrade**
   - Add harder negatives (e.g., semantically close but wrong-hue palettes) beyond random mismatch.
   - Consider spectral norm/gradient-penalty tuning if critic becomes unstable.

---

## 5) Suggested experiment plan (minimal)

Run three controlled GAN-only sweeps (same encoder checkpoint, same seed grid):

1. **Loss sweep**
   - A: current `(cond=1, energy=0, color=0)`
   - B: `(1, 0.01, 0.1)`
   - C: `(1, 0.05, 0.2)`

2. **Optimizer sweep**
   - SGD (baseline) vs AdamW

3. **Capacity sweep**
   - `GEN_HIDDEN_SIZE`: 32 vs 64

Track and compare:

- `cards.per_color.*.pure_accuracy` (especially blue/dark)
- `color_keywords.rows[*].value`
- `gan/energy/val`
- run-to-run variance over 3 seeds

Success criterion for next merge: blue and dark must improve materially without collapsing red/green/bright.

---

## Bottom line

Current GAN training is optimizing a signal that is too weakly tied to the actual product objective (semantic color control). The immediate fix is to restore a multi-term generator objective, move stage-2 optimization to Adam-style updates, and select checkpoints with per-color guardrails rather than global energy alone.