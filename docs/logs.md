# TensorBoard log names

Every scalar tag follows one template: `<stage>/<metric>[/<train|val>]`. `stage` is
`enc` / `gan` / `cond` — the same `LogStage` value used as the TensorBoard run version
(`runs/<CONFIG_NAME>/<stage>`). `metric` is a single name (may itself contain
underscores, e.g. `style_loss`, `cond_mean_score_real`). The `/<train|val>`
segment is present only for metrics logged on both splits — a metric logged on one split
only (or with no train/val distinction) omits it.

Within a metric name, `cond` means "scored by the *conditional* critic" (`CondColorCritic`
in `model/model.py`, scored against the text embedding) and `color` alone means "scored by
the *unconditional*, color-only critic" (`ColorCritic`) — `cond_*` vs. plain `color_*`/
`*_color` is the load-bearing distinction, not a shortening of the same thing.

`model/metric.py` is the source of truth. `named_metric(stage, source, metric, split)`
(`model/metric.py:34-40`) builds the `source/metric`-style enc/cond tags by joining
`Source` + `Metric` enum values into one `metric` segment, e.g.
`named_metric(LogStage.ENC, Source.EMOJI, Metric.MRR, Split.VAL)` → `enc/emoji_mrr/val`.
GAN tags don't fit the source+metric shape (asymmetric generator/critic naming) and are
instead flat string constants on `GanMetric` (`model/metric.py:43-55`), already written in
final `gan/<metric>[/<split>]` form. This doc is a reading guide, not a new spec.

## `enc` — encoder stage (`LitEncoder`, `model/train.py:151-283`)

Trained jointly: style head, emoji head, and the conditional color critic (co-trained here
so the encoder learns color-relevant structure, even though the critic/GAN proper trains
later in the `gan` stage).

| Tag | Split | Meaning |
|---|---|---|
| `enc/style_loss/train`, `enc/style_loss/val` | both | `lse_infonce` loss, style head |
| `enc/style_mrr/train`, `enc/style_mrr/val` | both | mean reciprocal rank, style head |
| `enc/emoji_loss/train`, `enc/emoji_loss/val` | both | `lse_infonce` loss, emoji head |
| `enc/emoji_mrr/train`, `enc/emoji_mrr/val` | both | mean reciprocal rank, emoji head. **`enc/emoji_mrr/val` is the checkpoint/early-stop monitor for this stage** (`model/train.py:554-576`) |
| `enc/full_text_acc@1` | val only | acc@1 over the full eval set, no per-source split |
| `enc/keyword_acc@1`, `enc/term_acc@1` | train only | acc@1 restricted to rows sampled from that source (`SAMPLING_SOURCES`, `model/config.py:129-140`); drives the adaptive per-source sampling rate, not just diagnostics |
| `enc/keyword_rate`, `enc/term_rate` | train only | current per-source sampling rate set by `on_train_epoch_start` (`model/train.py:251-267`) from the `acc@1` tags above |
| `enc/color_auroc/train`, `enc/color_auroc/val` | both | conditional color critic AUROC (real vs. shuffled-color pairs), co-trained with the encoder |

## `gan` — GAN stage (`LitColorGAN`, `model/train.py:286-452`)

Color generator vs. two critics (a *conditional* critic scored against the text
embedding, and an *unconditional* color-only critic).

| Tag | Meaning |
|---|---|
| `gan/energy/train`, `gan/energy/val` | Sinkhorn/energy-distance between real and generated color distributions (Oklab space). **`gan/energy/val` is the checkpoint/early-stop monitor for this stage**, mode `min` (`model/train.py:621-650`) |
| `gan/cond_loss` | conditional critic hinge loss |
| `gan/cond_auroc_gen` | conditional critic AUROC, real vs. generator-fake |
| `gan/cond_auroc_shuf` | conditional critic AUROC, real vs. shuffled-condition (mismatch) pairs |
| `gan/cond_mean_score_real`, `..._fake`, `..._wrong` | mean conditional-critic score by pair type (real, generator-fake, mismatched-condition) |
| `gan/color_loss` | unconditional color critic hinge loss |
| `gan/color_auroc` | unconditional color critic AUROC — **not** the same critic as `enc/color_auroc` / `cond/color_auroc` below, despite the shared `color_auroc` metric name; this one is unconditional (`ColorCritic`), those are conditional (`CondColorCritic`) |
| `gan/gen_loss_cond` | generator loss term from the conditional critic |
| `gan/gen_loss_color` | generator loss term from the unconditional critic |

## `cond` — conditional critic probe stage (`LitCondCriticProbe`, `model/train.py:455-496`)

Pretrains the conditional color critic alone (encoder frozen), ahead of the `gan` stage.
Same underlying metric as `enc/color_auroc/*`, distinguished by the `cond` stage prefix
since it's a different training context (encoder frozen here, co-trained there):

| Tag | Split |
|---|---|
| `cond/color_auroc/train` | train |
| `cond/color_auroc/val` | val |

No checkpoint/early-stop monitor here — this stage is a fixed-epoch probe
(`EPOCHS_COND_PROBE`), not an early-stopped fit.
