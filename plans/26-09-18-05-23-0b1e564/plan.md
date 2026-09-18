# Improvement Plan — 26-09-18-05-23 · 0b1e564 (GAN-only re-run)

Report: report/26-09-18-05-23-0b1e564/report.html
Prev plan: plans/26-09-18-02-11-0b1e564/plan.md
Changed since prev: 1 commit (`e26ce87 train gan`), encoder untouched (same `enc.pt`, sha `0b1e564`) — a `gan`-only stage re-run with three GAN-loss knobs moved together: `LOSS_WEIGHT_COND_COLOR` 0.9→0.5, `LOSS_WEIGHT_ENERGY` 0.9→0.1, `LR_GAN_CRITIC` 0.01→0.05. This is a real reversal of the exact `LOSS_WEIGHT_ENERGY` trend the previous plan flagged as "already swept 0.1→0.9 with improvement, don't reverse without evidence" — the background sweep reversed it anyway, in the same commit as two other knobs, one loop later.
Best emoji variant: EmojiHead (unchanged; encoder wasn't retrained)
Loop verdict: **actionable now, and the finding is a regression, not a new gap.** This is the cleanest single-loop comparison available so far (encoder held fixed, only GAN-stage config moved) and it reproduces the user's "still way off" observation directly: color energy got **worse on 4 of 5 categories** versus the prior report.

## Current status

| Category | Prior report (02-11) | This report (05-23) | Δ |
|---|---|---|---|
| Color energy · **global** (now wired) | 0.144 | **0.179** | worse |
| Color energy · red | 0.198 | **0.270** | worse — shortfall 2.6×→3.5× floor |
| Color energy · green | 0.080 (good) | **0.100** (now red) | worse — crossed back over target |
| Color energy · blue | 0.132 | **0.140** | worse |
| Color energy · dark | 0.248 | **0.347** | worse — shortfall 1.9×→2.7× floor |
| Color energy · bright | 0.064 (good) | 0.036 (good) | better, only improvement |

Priority-1 regression gate: n/a this loop — encoder wasn't retrained, emoji/style numbers are unchanged from the prior report.

## Gaps

- **The three-knob reversal (`LOSS_WEIGHT_COND_COLOR` 0.9→0.5, `LOSS_WEIGHT_ENERGY` 0.9→0.1, `LR_GAN_CRITIC` 0.01→0.05) regressed color energy on 4/5 categories, including flipping `green` from good back to red.** With the encoder held fixed, this is a much cleaner comparison than the 9-commit sweep behind the previous plan (which moved 7+ knobs simultaneously across encoder *and* GAN stages) — it isolates the change to the GAN stage alone. Still 3 knobs at once, not a clean single-variable ablation, but the direction is unambiguous and matches the previous plan's explicit warning against lowering `LOSS_WEIGHT_ENERGY`.
- **The new Δ`L`/Δchroma columns (wired last loop) confirm the same bias shape persists, just larger**: `dark` `l_bias` +0.061 (was +0.058 — still generating lighter than gold, now with a worse `gt_mean_distance` on top), `red` `chroma_bias` −0.021 (was −0.043 — direction unchanged, still desaturated). The underlying conditioning-bias diagnosis from the previous loop stands; this loop's regression is additive on top of it, not a different failure mode.
- **`LOSS_WEIGHT_COLOR_REG` (the previous loop's primary, still-untested recommendation) remains untouched at `1`.** Nothing about this loop's regression bears on that hypothesis one way or the other — it's still open. **Superseded below** — the user does not want to retrain the encoder, so this option (which requires a full `train --local`, re-running stage 1) is off the table regardless of merit.
- **User-directed re-diagnosis: "produces all the colors but fails conditional rendering," encoder to stay frozen.** A read-only variance-decomposition diagnostic against the current frozen `enc.pt`/`gen.pt` (fixed set of texts × 16 shared `z` draws, per `CARD_COLORS` category) confirms this precisely and locates it in `ColorGen`, not the encoder:

  | category | gen within-category between-text var (conditioning) | gen within-text var (z-noise) | ratio noise/conditioning | gold within-category spread |
  |---|---|---|---|---|
  | red | 0.00218 | 0.00088 | 0.40 | 0.02033 (9.3× the model's conditioning signal) |
  | green | 0.00083 | 0.00068 | 0.81 | 0.00143 (1.7×) |
  | blue | 0.00114 | 0.00076 | 0.67 | 0.01641 (14.4×) |
  | dark | 0.00128 | 0.00082 | 0.64 | 0.02315 (18.1×) |
  | bright | 0.00066 | 0.00072 | 1.10 | 0.00751 (11.4×) |

  Two findings: (1) gold data has real, substantial *within-category* color diversity (different "dark" texts genuinely call for different specific dark shades) that the generator's text-conditioning barely reproduces — 2–18× under-modulated depending on category, worst exactly where gold diversity is highest (dark, blue, bright); (2) within a single category, the `z`-noise contributes **comparable or more** output variance than the true text-to-text conditioning signal (ratio 0.4–1.1) — noise isn't riding on top of a strong conditioning signal, it's competing with a weak one. `green` — the one category close to its goal — is the one where gold's own within-category diversity is lowest (0.00143, closest to what the model can express) and the noise/conditioning ratio is highest (0.81), i.e. the easiest case for this architecture, not a sign the model is doing anything qualitatively different there.

  **Mechanism**: `model/model.py:147-161`, `ColorGen.forward` — `seed = (1 - Z_WEIGHT) * normalize(cond) + Z_WEIGHT * normalize(z)`, both full `EMBED_SIZE_TEXT`-dim (460) unit vectors, summed in the *same* space before the first `Linear(EMBED_SIZE_TEXT, GEN_HIDDEN_SIZE)`. Two random unit vectors in 460-D are ~orthogonal, so this isn't noise "added around" the conditioning direction — it's a second, unrelated direction blended additively into every one of the same 460 dimensions the network must read fine color signal from, with a single global scalar (`Z_WEIGHT`) controlling the blend regardless of which dimensions actually carry color information. Sweeping `Z_WEIGHT` (already tried: 0.2→0.3) or the loss weights can shift the balance but can't stop the noise from being mixed into the same coordinates as the signal — that's a structural property of the blend, not a magnitude the config can fix around.

## Proposed actions

### 1. Revert the three GAN-loss knobs to their prior (better) values — directly implementable, primary recommendation
- **File**: `model/config.py`. `LOSS_WEIGHT_COND_COLOR` 0.5 → **0.9**, `LOSS_WEIGHT_ENERGY` 0.1 → **0.9**, `LR_GAN_CRITIC` 0.05 → **0.01** — restoring exactly the config that produced the 02-11 report (the best color-energy numbers seen across both loops).
- **Reasoning**: direct, encoder-held-fixed before/after comparison shows this exact change regressed 4/5 categories. Reverting isn't new exploration, it's undoing a measured regression.
- **Estimated effect**: should recover the 02-11 report's color-energy numbers (global 0.179→~0.144, `green`/`bright` back to good) once the GAN stage is retrained from this config.
- **Risk to priority 1**: none — GAN stage only, encoder frozen.
- Needs a real `train gan --local` (or full `train --local`) to evaluate — config-only change, not evaluable from existing checkpoints.

### 2. ~~Apply `LOSS_WEIGHT_COLOR_REG` 1→3~~ — dropped per user constraint (no encoder retrain)
- Requires a full `train --local` (re-runs stage 1). Off the table now that the encoder is to stay frozen, regardless of merit.

### 3. Architecture — decouple `z` from `cond` in `ColorGen` — recommend, GAN-stage only, no encoder retrain
- **File**: `model/model.py`, `ColorGen.__init__`/`forward` (~138-161). Replace the additive blend with concatenation: add a new `NOISE_DIM` constant (e.g. 32, well under `EMBED_SIZE_TEXT`'s 460), change the first layer to `nn.Linear(EMBED_SIZE_TEXT + NOISE_DIM, GEN_HIDDEN_SIZE)`, and build the input as `torch.cat([normalize(cond, dim=-1), z], dim=-1)` with `z ~ N(0, I)` at `NOISE_DIM` width (drop the `Z_WEIGHT` scalar blend — let the network's own weights decide how much to lean on noise vs. conditioning, per output dimension, instead of one fixed global mix ratio).
- Also needs: `model/train.py`'s `LitColorGAN.z_bank` (currently `(ENERGY_Z_SAMPLES, EMBED_SIZE_TEXT)`) resized to `(ENERGY_Z_SAMPLES, NOISE_DIM)`, and `model/export_onnx.py`'s `CONST_Z` likewise — both currently assume `z` is `EMBED_SIZE_TEXT`-wide.
- **Reasoning**: directly targets the mechanism identified above — noise and conditioning currently share all 460 dimensions with a single global mix scalar; concatenating them gives the generator a clean, always-fully-present `cond` input plus a separate, much narrower noise channel it can learn to use non-uniformly (e.g. leaning on it more for legitimately multimodal texts, less for texts with a strong, specific color implication).
- **Confirms the user's constraint**: `TextEncoder`/`enc.pt` is not touched or retrained — only `ColorGen`/`ColorCritic` (already retrained from scratch each GAN stage run) are affected. `ColorCritic` itself needs no change (it already takes `cond` and `colors` as separate, unblended inputs).
- **Param/latency cost**: negligible — first-layer input width goes from 460 to ~492 (460 + 32), a <7% parameter increase in `ColorGen`'s first layer only.
- **Risk**: none to priority 1 (encoder frozen). To priority 2 itself: this is a real structural change to a model that's already been trained several times at the old shape, so it needs a fresh `train gan --local` to evaluate, and could plausibly need re-tuning of `LOSS_WEIGHT_ENERGY`/`LOSS_WEIGHT_COND_COLOR`/`LR_GAN_*` around the new architecture (the old sweep's conclusions were all found around the old blend mechanism, not guaranteed to transfer) — recommend re-running with the *known-good* 02-11 values first (see action 1) rather than re-opening that whole sweep blind.
- **Sequencing**: do this together with action 1 (revert the regressed loss knobs) in the same `train gan --local` — they're not confounded with each other the way action 1 vs. the old action 2 would have been, since this is a shape change and action 1 is restoring known-good loss weights for that new shape's first real run.

## `goals.yml` adjustments

No change proposed.

## Root-cause investigation — 2026-09-18, user-directed (systematic-debugging)

User asked to stop proposing the loss-knob revert (fine-tune only, not the fix) and instead verify (1) whether `ColorCritic` sees real and generated colors in the same format, and (2) whether there's a bug in the GAN training step. Two more background `train gan` runs landed mid-investigation (`e26ce87`→`14ce337`): the auto-tuner independently zeroed `Z_WEIGHT` (0.3→0) and added a `normalize(seed, dim=-1)` in `ColorGen.forward` — i.e. it's also poking at the exact cond/z blend mechanism flagged in this plan's action 3, by a different route. Result (`report/26-09-18-07-45-0b1e564`): global color energy 0.174, barely moved from 0.179, still far worse than the 02-11 report's 0.144 — **removing `z` entirely did not fix conditional rendering**, which is evidence against "noise-vs-signal blending" being the *dominant* cause (though it doesn't rule out the compression bottleneck in `ColorGen`'s first layer as a contributor).

**Phase 1 — traced the full real/fake data path into `ColorCritic`:**
- `model/data.py:colors2tensor`: real colors are `int(hex_byte, 16) - 127.5` per channel — always lands exactly on a half-integer (fractional part `.5`), by construction of parsing an 8-bit hex value.
- `model/model.py:ColorGen.forward`: fake colors are `tanh(net(seed)) * 127.5` — a continuous float, never on that lattice except by ~0 probability.
- `model/train.py:LitColorGAN.training_step`: `pair = torch.cat([colors, fake.detach()], dim=0)` then `self.critic(cond_pair, pair)` — **both go through the identical `nn.Linear(COLOR_DIM, CRITIC_EMBEDDING_SIZE)`, same dtype/shape/scale, no code-path divergence.** Confirmed no dataset filtering issue either — `data/train.jsonl` has `bg`/`fg` on all 213,367 records, so no zero-filled/placeholder rows silently entering the "real" colors tensor.
- **But empirically, on the current `enc.pt`+`gen.pt` against a 2000-row real sample**: `100.0%` of real color values have fractional part exactly `0.5`; only `0.03%` of generated values do. Aggregate mean/std/range otherwise closely match (real std 79.5 vs fake 77.8, both spanning close to the full ±127.5), so this isn't a scale bug — it's a **support/quantization mismatch**: real is a lattice, fake is continuous. This is the textbook "discrete real vs. continuous generated data" discriminator shortcut (the same failure mode addressed by pixel dequantization/instance-noise in image GANs) — a critic can win near-perfectly on this cue alone, independent of whether the color is actually right for the text, which starves `cond_score`/`color_score` (the only loss terms that tie a *specific* color to a *specific* text) of a meaningful gradient while the direct `energy_distance` term (computed in continuous Oklab space, insensitive to the RGB-integer lattice) still pulls the aggregate distribution into roughly the right shape — matching exactly what was observed ("produces all the colors" in aggregate, "fails conditional rendering" per-text).
- **Gradient-flow/detach/batching review of `training_step`**: no bug found. `cond`/`fake` pairing in `torch.cat([colors, fake.detach()])` / `torch.cat([cond, cond])` is correctly index-aligned; critic step correctly detaches `fake`; generator step correctly re-forwards `self.critic(cond, fake)` (not detached) after `opt_critic.step()`, so gradient flows into `self.gen` through the just-updated critic, which is standard and correct; `opt_gen`/`opt_critic` zero_grad/backward/clip/step ordering is standard hinge-loss GAN training with no reuse-of-freed-graph issue. `self.enc.eval()` is set each `on_train_epoch_start`; `gen`/`critic` correctly stay in `.train()` mode (needed for spectral-norm's power iteration to update). **No code-logic bug identified in the training step itself** — the issue is the *data format* reaching the critic, not the training loop's mechanics.

**Root cause (single hypothesis, per Phase 3)**: the critic can trivially separate real from fake using the real-data quantization artifact (exact half-integer values) rather than being forced to learn genuine text-color correspondence, which weakens the one loss component (`cond_score`) designed to teach *conditional* accuracy.

**Proposed minimal test (not yet applied — one variable, per debugging discipline)**: dequantize the real `colors` batch before it reaches the critic — `colors_dq = colors + (torch.rand_like(colors) - 0.5)` (uniform dither of ±0.5, matching the lattice spacing) — applied only where `colors` feeds `self.critic(...)` in `training_step` (both the critic-update pairing and nowhere else — `loss_energy`'s `rgb_to_oklab(colors)` should stay on the true values, dequantizing only where the critic could exploit the artifact). This is a training-step data-formatting fix, not a loss-weight change, not a `model.py` architecture change, and does not touch `TextEncoder`. Needs a `train gan --local` to evaluate.
