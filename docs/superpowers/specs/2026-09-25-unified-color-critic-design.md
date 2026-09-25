# Unified color Critic

## Goal

Replace `ColorEmbedding`, `ColorCritic`, and `CondColorCritic`
(`model/model.py`) with a single `Critic` module that always conditions
on text. During encoder training (`LitEncoder`) and the standalone probe
(`LitCondCriticProbe`), each step draws three equal-sized groups from
the color batch:

- 1/3 real `(text, color)` pairs — positive
- 1/3 shuffled `(text, color)` pairs — real color, mismatched text
  (same failure mode the old `CondColorCritic` targeted)
- 1/3 `(real text, random color)` pairs — a color that isn't from the
  dataset at all (the failure mode the old unconditional `ColorCritic`
  targeted)

and logs two AUROCs: real-vs-shuffled (pairing sensitivity) and
real-vs-random (realism sensitivity), replacing the single combined
`.../color_auroc`.

## Current state

Three classes in `model/model.py`:

- `ColorEmbedding` — `colors -> embedding` (2×`cblk`, spectral-norm).
- `ColorCritic` — `embedding -> scalar`, unconditional.
- `CondColorCritic` — `(cond, color_embedding) -> scalar` via a
  text-side projection dotted with the (externally computed) color
  embedding.

Three call sites in `model/train.py`:

- `LitEncoder._step` — owns one `CondColorCritic` + `ColorEmbedding`
  pair, trains it jointly with the style/emoji heads against real vs.
  same-batch-shuffled colors. Checkpointed as `cond_critic.pt` /
  `cond_color_embed.pt`.
- `LitCondCriticProbe` — a standalone CLI stage (`train cond`) that
  re-trains a *fresh* `CondColorCritic` + `ColorEmbedding` against a
  frozen encoder, real vs. shuffled only. Not checkpointed into the
  main pipeline; exists to measure text/color pairing capacity in
  isolation.
- `LitColorGAN` — uses the warm-started `CondColorCritic` (`self.critic`)
  adversarially against the generator (real vs. `gen(cond)` vs.
  shuffled), **and** a second, ephemeral, never-checkpointed
  `ColorCritic` + `ColorEmbedding` pair trained unconditionally on real
  vs. `gen(cond)`. The unconditional critic's output currently feeds the
  generator loss with weight `GAN_LOSS_COLOR = 0` — i.e. it already has
  zero effect on the generator today.

## New `Critic` module

`model/model.py`, replacing all three classes. No dual-tower
dot-product: concatenate the text embedding and the raw color vector
into one vector and run it through a single deep net down to a scalar.

```python
CRITIC_HIDDEN_SIZE = 128  # model/config.py


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *cblk(EMBED_SIZE_TEXT + COLOR_DIM, CRITIC_HIDDEN_SIZE),
            *cblk(CRITIC_HIDDEN_SIZE, CRITIC_HIDDEN_SIZE),
            *cblk(CRITIC_HIDDEN_SIZE, CRITIC_HIDDEN_SIZE),
            sn(nn.Linear(CRITIC_HIDDEN_SIZE, 1)))

    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        assert torch.all((colors >= -COLOR_SHIFT) & (colors <= COLOR_SHIFT)), \
            f"colors must be in [-{COLOR_SHIFT}, {COLOR_SHIFT}], " \
            f"got min={colors.min().item()} max={colors.max().item()}"
        return self.net(torch.cat([cond, colors], dim=-1))
```

Three `cblk` blocks (spectral-norm linear + LayerNorm + LeakyReLU,
same block style the old critics used) then a spectral-norm linear
readout — "deep" relative to the old two-block towers, since the net
now has to do the text/color interaction work that used to be a dot
product. `CRITIC_HIDDEN_SIZE` is a new `model/config.py` constant.
`EMBED_SIZE_COLOR` (only ever used by the three deleted classes — not
by `ColorGen`, which sizes off `Z_SIZE`/`GEN_HIDDEN_SIZE`/`COLOR_DIM`
directly) becomes dead and is deleted along with them.

There's no more separate `embed`/`score` split: `forward` is the only
entry point, and every score requires a full forward pass over the
concatenated vector — a call site can no longer compute one color
embedding once and cheaply re-score it against two different `cond`s.
`LitColorGAN`'s `real`/`wrong_score` reuse (see below) loses that
reuse and calls `self.critic(...)` twice instead. There is also no
separate unconditional scalar head — realism judgment is folded into
the same score by training it against random-color negatives, which is
what makes this module a genuine replacement for `ColorCritic` rather
than just a rename of `CondColorCritic`.

## Random-color negative

`model/data.py` already has an unused `rnd_color_tensor()`:

```python
def rnd_color_tensor() -> torch.Tensor:
    return torch.randint(0, 256, (COLOR_DIM,), dtype=torch.float32) - 127.5
```

Generalize it to a batched form and use it as the random-color source
everywhere one is needed:

```python
def rnd_color_tensor(n: int, device=None) -> torch.Tensor:
    return torch.randint(0, 256, (n, COLOR_DIM), device=device, dtype=torch.float32) - 127.5
```

This keeps the random negative on the same discrete byte-quantized
distribution as real colors (`colors2tensor`), rather than a
continuous float uniform, since that's the format the critic sees
everywhere else.

## `LitEncoder` / `LitCondCriticProbe`: shared three-way step

Both currently duplicate the same real-vs-shuffled logic. Factor the
new real/shuffled/random step into one module-level helper in
`model/train.py`, used by both:

```python
def _critic_probe_step(
    critic: Critic, cond: torch.Tensor, colors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = colors.shape[0]
    shuffled = colors[torch.randperm(n, device=colors.device)]
    random_colors = rnd_color_tensor(n, device=colors.device)

    pair = torch.cat([colors, shuffled, random_colors], dim=0)
    cond_pair = torch.cat([cond, cond, cond], dim=0)
    score = critic(cond_pair, pair)
    real, shuf_score, rand_score = score.chunk(3, dim=0)

    loss = relu(1 - real).mean() + relu(1 + shuf_score).mean() + relu(1 + rand_score).mean()

    target = torch.cat([score.new_ones(n), score.new_zeros(n)])
    auroc_shuf = binary_auroc(
        torch.cat([real, shuf_score], dim=0).detach().squeeze(-1), target.long())
    auroc_rand = binary_auroc(
        torch.cat([real, rand_score], dim=0).detach().squeeze(-1), target.long())

    return loss, auroc_shuf, auroc_rand
```

- `LitEncoder.__init__`: `self.cond_critic` + `self.color_embed` ->
  `self.critic = Critic()`. `_step`'s color block calls
  `_critic_probe_step(self.critic, cond_c, colors_c)` and logs both
  AUROCs (`Metric.AUROC_SHUF`, `Metric.AUROC_RAND`) under
  `LogStage.ENC`/`Source.COLOR`, in place of the single `Metric.AUROC`.
  `configure_optimizers` drops `color_embed.parameters()` (now part of
  `critic.parameters()`).
- `LitCondCriticProbe.__init__`: `self.critic = Critic()`.
  `_step` becomes a thin wrapper over `_critic_probe_step`; both AUROCs
  logged under `LogStage.COND`/`Source.COLOR` in `training_step` /
  `validation_step`, same on_step/on_epoch/prog_bar behavior as today.

## `LitColorGAN`: one critic, four groups

The unconditional `ColorCritic`/`ColorEmbedding` pair, its optimizer,
and `loss_gen_color_critic` (already zero-weighted) are deleted
outright — merged into the single conditional `Critic`, per the
"replaces both" instruction. `self.critic: Critic`, warm-started same
as today.

Per step, the critic now sees real + three negative groups. Each score
is its own forward pass (no more embed-once-score-twice: see above) —
`real` and `wrong_score` each run `colors` through the full net rather
than sharing one color embedding:

```python
random_colors = rnd_color_tensor(colors.shape[0], device=colors.device)
cond_wrong = cond[torch.randperm(cond.shape[0], device=cond.device)]

real = self.critic(cond, colors)
fake_score = self.critic(cond, fake.detach())
wrong_score = self.critic(cond_wrong, colors)
random_score = self.critic(cond, random_colors)

loss_critic = relu(1 - real).mean() \
    + COND_CRITIC_MISMATCH_WEIGHT * relu(1 + fake_score).mean() \
    + (1 - COND_CRITIC_MISMATCH_WEIGHT) * relu(1 + wrong_score).mean() \
    + COND_CRITIC_RANDOM_WEIGHT * relu(1 + random_score).mean()
```

`COND_CRITIC_MISMATCH_WEIGHT` (0.9/0.1 fake/wrong split) is unchanged
from today, to avoid perturbing the already-tuned adversarial dynamics
(see `docs/gan.md`'s open instability notes). `COND_CRITIC_RANDOM_WEIGHT`
is a new, independent, low-weight term (default `0.1`, same order as
the wrong-pairing term, since it's an auxiliary sanity check rather
than the primary adversarial signal) — a knob for the next
report-driven tuning pass, not something to over-fit in this change.

Three AUROCs logged (`gan/cond_auroc_gen`, `gan/cond_auroc_shuf`
unchanged; new `gan/cond_auroc_rand` replacing `gan/color_auroc`), each
`real` vs. one negative group.

The generator step is unaffected in structure — `gen_score =
self.critic(cond, fake)`, `loss_gen = GAN_LOSS_COND * loss_gen_critic +
GAN_LOSS_ENERGY * loss_energy`. `GAN_LOSS_COLOR` and the
`loss_gen_color_critic` term are removed (dead: `GAN_LOSS_COLOR` is `0`
today, so this changes no numbers, only removes the vestigial critic).

`configure_optimizers` returns `[opt_gen, opt_critic]` (was
`[opt_gen, opt_critic, opt_color_critic]`); `opt_critic` now optimizes
just `self.critic.parameters()` (`LR_GAN_COND_CRITIC`).
`training_step` unpacks `opt_gen, opt_critic = self.optimizers()`.

`_train_gan(enc, enc_path, critic: Critic, ds, val_ds, out_dir)` takes
one `Critic` instead of `(critic, color_embed)`; same for
`LitColorGAN.__init__(self, critic: Critic)` and the
`load_from_checkpoint(..., critic=Critic())` call.

## Metrics (`model/metric.py`)

- `Metric.AUROC` -> `Metric.AUROC_SHUF` ("auroc_shuf") and
  `Metric.AUROC_RAND` ("auroc_rand"). (Confirmed unused anywhere else —
  `tools/report.py`, `model/metrics.py`, `goals.yml` don't reference
  `auroc`/`color_auroc`/`cond_auroc` by string.)
- `GanMetric.COLOR_LOSS`, `GanMetric.COLOR_AUROC` deleted.
- `GanMetric.COND_AUROC_RAND = "gan/cond_auroc_rand"` added.
  `COND_AUROC_GEN`, `COND_AUROC_SHUF` unchanged.
- `GanMetric.COND_MEAN_SCORE_RANDOM` added (mirrors the existing
  `_REAL`/`_FAKE`/`_WRONG` mean-score debug logs, `prog_bar=False`).

## Checkpoints (`files.py`)

`PtFile.COND_CRITIC` + `PtFile.COND_COLOR_EMBED` ("cond_critic.pt" +
"cond_color_embed.pt") collapse into one `PtFile.CRITIC = "critic.pt"`.
`COND_CRITIC_PT` / `COND_COLOR_EMBED_PT` -> `CRITIC_PT`. This is a
breaking checkpoint rename — old `pt/cond_critic.pt` /
`pt/cond_color_embed.pt` files are orphaned; per project convention
(checkpoints get overwritten / regenerated from a full `train --local`
run, never hand-migrated), no migration path is provided.

`model/train.py`:
- `_load_cond_critic(pt_dir) -> (CondColorCritic, ColorEmbedding)`
  becomes `_load_critic(pt_dir) -> Critic`, loading `PtFile.CRITIC`,
  same fresh-fallback-on-mismatch behavior.
- `_train_encoder` saves `mod.critic.state_dict()` to
  `PtFile.CRITIC.in_dir(out_dir)` once, instead of two separate saves.
- `_pt_files_ok` is unaffected (it doesn't check the critic files
  today).

## Config (`model/config.py`)

- `EMBED_SIZE_COLOR` deleted (dead — see above); `CRITIC_HIDDEN_SIZE = 128`
  added in its place.
- `LR_GAN_CRITIC` deleted (was the now-gone unconditional critic's LR).
  `LR_GAN_COND_CRITIC` stays, now the merged GAN-stage critic's LR.
- `GAN_LOSS_COLOR` deleted (dead generator-loss weight, see above).
- `COND_CRITIC_RANDOM_WEIGHT = 0.1` added, next to
  `COND_CRITIC_MISMATCH_WEIGHT`.
- `gan_str` (config-name fingerprint string) drops `LR_GAN_CRITIC`,
  `EMBED_SIZE_COLOR`, and `GAN_LOSS_COLOR`, gains `CRITIC_HIDDEN_SIZE`
  and `COND_CRITIC_RANDOM_WEIGHT`.

## Test (`model/test_train_cli.py`)

`test_colorcritic_forward_shape` updates to the unified class:

```python
def test_colorcritic_forward_shape():
    from model.model import Critic

    score = Critic()(torch.zeros(5, EMBED_SIZE_TEXT), torch.zeros(5, 9))
    assert score.shape == (5, 1)
```

## Files touched

- `model/model.py` — delete `ColorEmbedding`, `ColorCritic`,
  `CondColorCritic`; add `Critic`.
- `model/data.py` — generalize `rnd_color_tensor` to batched.
- `model/train.py` — `LitEncoder`, `LitCondCriticProbe`, `LitColorGAN`,
  `_load_cond_critic` -> `_load_critic`, `_train_encoder`, `_train_gan`,
  new `_critic_probe_step` helper.
- `model/metric.py` — `Metric.AUROC` -> `AUROC_SHUF`/`AUROC_RAND`;
  `GanMetric` additions/removals above.
- `files.py` — `PtFile.CRITIC` replacing `COND_CRITIC`/`COND_COLOR_EMBED`.
- `model/config.py` — weight/LR constants above.
- `model/test_train_cli.py` — updated smoke test.

Out of scope: `web/`, `model/export_onnx.py`, `model/pred.py`,
`model/runmeta.py`, `tools/report.py` — none reference the critic
directly (the critic never ships to the browser; only `gen.pt` does).

## Verification

- `uv run ruff check .`
- `bun run regen` (label vocab, required before any Python entry point).
- `uv run python model/test_train_cli.py` (the `main` smoke-test
  command — not pytest, per repo convention).
- `train enc --local` — watch `enc/color_auroc_shuf`,
  `enc/color_auroc_rand` in TensorBoard; confirm `pt/critic.pt` written
  (and `cond_critic.pt`/`cond_color_embed.pt` no longer produced).
- `train cond --local` (the `LitCondCriticProbe` stage) — same two
  AUROCs under `cond/color_auroc_{shuf,rand}`.
- `train gan --local` — confirm only two optimizers, watch
  `gan/cond_auroc_gen`, `gan/cond_auroc_shuf`, `gan/cond_auroc_rand`;
  confirm `gan/color_auroc` and `gan/color_loss` no longer logged.
- `train --local` full pipeline end to end, then `tools/report.py`
  (auto-runs) to confirm nothing downstream broke.
