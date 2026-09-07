# Color GAN: why text→color is broken, and how to fix it

## Symptom

`pred/colors.jsonl` (GAN output for the 8227 corpus texts) has almost no
relationship to the text. On a 50-row sample and on the full file:

- Human annotations (`data/colors.jsonl`): ~80% of palettes match a clear
  color/object cue in the text (fruit→red, sky→blue, slate→gray, jacaranda→purple).
- GAN predictions: ~18% strong match, ~48% contradict a clear cue.
- Full file: predicted `bg` hue matches the human `bg` hue **13.5%** of the time
  — *below* the ~14.4% random baseline from the prediction's own hue histogram.
- Saturation/lightness *are* learned: pred mean sat/light 0.49 / 0.67 vs real
  0.45 / 0.66. Hue distribution has drifted and flattened (pred piles onto
  yellow/cyan; real leans blue/green) — partial mode collapse.

The GAN matches `P(color)` and ignores `P(color | text)`.

## Evidence from the last GAN run (`runs/TIME: 2026-09-06 21:36:32/gan/`)

| metric | start | end | reference |
|---|---|---|---|
| `energy/gan/val` (fake↔real, OKLab) | 0.287 | 0.097 (plateau) | `energy/gan/ref` real↔real = 0.0245 |
| `loss/gan/tst` (critic BCE, real+fake) | 0.53 | 1.33 | 2·ln2 ≈ 1.386 = "can't tell at all" |
| `loss/gan/gen` | 1.71 | 0.74 | ln2 ≈ 0.693 = same point |
| epochs actually run | | 20 | `EPOCHS_GAN=100`, early-stopped |

Today's `color-exp` run (`runs/TIME: 2026-09-07 09:50:21/color-exp/`, encoder
unfrozen, trained jointly 100 epochs): `energy/gan/train` 0.70 → 0.043 vs `ref`
≈ 0.010; `loss/gan/tst` 0.79 → 1.35, `loss/gan/gen` → 0.73. Identical picture.

**Diagnosis:** in both runs the critic converges to the trivial equilibrium
(outputs ≈0.5 for everything) while the energy distance is still ~4× the
real-vs-real floor. From ~epoch 15 there is no useful gradient, so early-stop
kills the run. Unfreezing the encoder and training 5× longer changes nothing —
the critic is the bottleneck.

## Root causes in the code

1. **The critic is almost blind to text.** `CRITIC_TEXT_CHANNELS = [8, 4]`
   (`model/config.py:63`) compresses the 416-d text embedding to **4 dims**
   before concatenating with 128-d color features (`model/model.py:209-217`).
   It cannot represent "ocean vs. fire vs. limes", so its real/fake decision is
   driven entirely by the color branch = "is this a plausible palette". The
   generator is only pushed toward the marginal.
2. **No term forces joint (text, color) matching.** Vanilla concat-cGAN with BCE
   is free to ignore the condition, and here it does. No mismatched-pair
   negatives.
3. **Generator bottleneck is tiny.** `GEN_CHANNELS = [32, 32]`
   (`model/config.py:61`): `Linear(416 → 32)` first layer (`model/model.py:160`)
   collapses the semantic vector before anything conditional can happen.
4. **Noise is added into the conditioning vector**, not concatenated:
   `seed = 0.8·normalize(cond) + 0.2·z` (`model/model.py:182`). Signal and noise
   share a subspace; `normalize()` also discards magnitude.
5. **Critic has BatchNorm and no spectral norm.** `_critic_branch`
   (`model/model.py:189-201`) uses bare `nn.Linear` + `BatchNorm1d` despite
   CLAUDE.md claiming "spectral norm". BN in a BCE critic couples samples and
   weakens conditional signal; nothing constrains the Lipschitz constant.
6. **Plain SGD, 1:1 steps, huge batch.** `optim.SGD` no momentum
   (`model/train.py:302`), `GAN_BATCH_SIZE=512` ≈ 13 steps/epoch ≈ 260 generator
   updates total. GANs need Adam with low β1 (TTUR) and more steps.
7. **The energy distance is monitored but never optimized.** It's the early-stop
   metric and it plateaus immediately because the critic saturates — nothing
   drives it down.

## Suggested changes, by expected impact

### 1. Make it actually conditional (highest leverage)

Add **mismatched-real negatives** (Reed et al. 2016, text-to-image). Three critic
terms instead of two:

```python
roll = torch.roll(colors, shifts=1, dims=0)          # real palettes, wrong text
tst_real  = self.tst(cond, colors)          # -> 1
tst_wrong = self.tst(cond, roll)            # -> 0   (NEW)
tst_fake  = self.tst(cond, fake.detach())   # -> 0
loss_tst = bce(tst_real, 1) + 0.5*bce(tst_wrong, 0) + 0.5*bce(tst_fake, 0)
```

~5 lines; directly teaches "plausible palette + wrong text = fake", the exact
failure mode.

Widen the critic's text branch and switch to a **projection discriminator**
(Miyato & Koyama) — standard fix for "cGAN ignores the label", much stronger than
concat:

```python
# CRITIC_TEXT_CHANNELS = [128, 64]
score = self.phi(color_feats) + (self.proj(cond) * color_feats).sum(-1, keepdim=True)
```

### 2. Fix the generator

- `GEN_CHANNELS = [256, 128, 64]` (or at least `[128, 64]`).
- Concatenate noise instead of blending:
  `seed = torch.cat([cond, Z_WEIGHT * z], dim=-1)`, first
  `Linear(TEXT_EMBED_SIZE + TEXT_EMBED_SIZE, ...)`. Keeps identity and sampling in
  separate subspaces.
- Consider not `normalize()`-ing `cond` (or keep its norm as an extra scalar
  feature).

### 3. Fix the optimizer / schedule

- `optim.Adam(lr=2e-4, betas=(0.0, 0.9))` for both (TTUR: keep critic LR ≥ gen LR).
- `GAN_BATCH_SIZE = 128`.
- `n_critic = 2-3` critic steps per generator step.
- Add **spectral norm** to `_critic_branch` linears (`sn` is already imported) and
  drop `BatchNorm1d` from the critic; optionally add an R1 penalty on real.

### 4. Put the energy distance back into the generator loss (small weight)

CLAUDE.md removed it fearing it kills diversity, but energy distance between
*distributions* (what `energy_distance` computes over the batch) is the correct
one-to-many objective — it does not penalize producing several valid palettes.
Add it as a gradient that survives critic saturation:

```python
loss_gen = bce(tst_fake, 1) + LAMBDA_ENERGY * energy_distance(
    rgb_to_oklab(fake), rgb_to_oklab(colors))
```

Start `LAMBDA_ENERGY ≈ 0.1`, watch `energy/gan/val` vs `energy/gan/ref`.

### 5. Cheap sanity experiments first

- **Overfit check:** train the GAN on 200 rows with the current architecture. If
  it still can't drive `energy/gan/val` near `ref`, architecture (not data/optim)
  is confirmed as the bottleneck.
- **k-NN color baseline:** for each eval text, retrieve the nearest train text by
  encoder embedding and copy its palette. If that beats the GAN on hue-match,
  ship the retrieval baseline for the site while iterating on the GAN.
- Log a **hue-match-rate** metric (fraction of eval rows where fake `bg` hue ==
  real `bg` hue) next to `energy/gan/val`, so the thing that's actually broken is
  tracked, not just aggregate energy.

## Notes on data (not the bottleneck)

- ~8% of corpus rows have no color cue in the text and get an arbitrary neutral
  from the annotator; the top-level `"color"` bucket tag is noisy and not ground
  truth (e.g. "Mind the fox near the shed" tagged `red`). The strong rows are
  ~80% consistent — enough signal. Fix the model/critic before touching data.
