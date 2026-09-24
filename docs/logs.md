# TensorBoard log names

Every scalar tag is built from `model/metric.py`: `named_metric(source, metric, split)`
joins `Source`/`Metric`/`Split` enum values with `/` (`model/metric.py:30-36`), e.g.
`emoji/mrr/val`. GAN tags don't fit that `source/metric/split` shape (asymmetric
generator/critic naming) and are instead flat string constants on `GanMetric`
(`model/metric.py:39-51`). Both are the source of truth — this doc is a reading guide, not
a new spec.

Each training stage writes its own TensorBoard run directory,
`runs/<CONFIG_NAME>/<version>`, where `<version>` is `enc` / `gan` / `cond` (matches
`Stage` in `model/train.py:145-148`, set per `TensorBoardLogger(..., version=...)` call).
Tags below are grouped by which stage writes them.

## `enc` — encoder stage (`LitEncoder`, `model/train.py:151-277`)

Trained jointly: style head, emoji head, and the conditional color critic (co-trained here
so the encoder learns color-relevant structure, even though the critic/GAN proper trains
later in the `gan` stage).

| Tag | Split | Meaning |
|---|---|---|
| `style/loss/train`, `style/loss/val` | both | `lse_infonce` loss, style head |
| `style/mrr/train`, `style/mrr/val` | both | mean reciprocal rank, style head |
| `emoji/loss/train`, `emoji/loss/val` | both | `lse_infonce` loss, emoji head |
| `emoji/mrr/train`, `emoji/mrr/val` | both | mean reciprocal rank, emoji head. **`emoji/mrr/val` is the checkpoint/early-stop monitor for this stage** (`model/train.py:564-581`) |
| `emoji/avg_logits/pos/train`, `.../val` | both | mean emoji-head logit over true-label positions (sanity signal, not a loss) |
| `emoji/avg_logits/neg/train`, `.../val` | both | mean emoji-head logit over non-label positions |
| `full_text/acc@1` | val only | acc@1 over the full eval set, no per-source split |
| `keyword/acc@1`, `term/acc@1` | train only | acc@1 restricted to rows sampled from that source (`SAMPLING_SOURCES`, `model/config.py:129-140`); drives the adaptive per-source sampling rate, not just diagnostics |
| `keyword/rate`, `term/rate` | train only | current per-source sampling rate set by `on_train_epoch_start` (`model/train.py:261-277`) from the acc@1 tags above |
| `color/auroc/train`, `color/auroc/val` | both | conditional color critic AUROC (real vs. shuffled-color pairs), co-trained with the encoder |

## `gan` — GAN stage (`LitColorGAN`, `model/train.py:607-462` region)

Color generator vs. two critics (a *conditional* critic scored against the text
embedding, and an *unconditional* color-only critic). All `gan/*` tags are `GanMetric`
constants, not `named_metric` tuples.

| Tag | Meaning |
|---|---|
| `gan/energy/train`, `gan/energy/val` | Sinkhorn/energy-distance between real and generated color distributions (Oklab space). **`gan/energy/val` is the checkpoint/early-stop monitor for this stage**, mode `min` (`model/train.py:630-654`) |
| `gan/cond_color_critic/loss` | conditional critic hinge loss |
| `gan/cond/auroc/gen` | conditional critic AUROC, real vs. generator-fake |
| `gan/cond/auroc/shuf` | conditional critic AUROC, real vs. shuffled-condition (mismatch) pairs |
| `gan/cond_color_critic/mean_score/real`, `.../fake`, `.../wrong` | mean conditional-critic score by pair type (real, generator-fake, mismatched-condition) |
| `gan/color_critic/loss` | unconditional color critic hinge loss |
| `gan/color_critic/auroc` | unconditional color critic AUROC |
| `gan/gen/loss/cond_color_critic` | generator loss term from the conditional critic |
| `gan/gen/loss/color_critic` | generator loss term from the unconditional critic |

## `cond` — conditional critic probe stage (`LitCondCriticProbe`, `model/train.py:465-506`)

Pretrains the conditional color critic alone (encoder frozen), ahead of the `gan` stage.
Reuses the same tag names as the encoder stage's color critic:

| Tag | Split |
|---|---|
| `color/auroc/train` | train |
| `color/auroc/val` | val |

No checkpoint/early-stop monitor here — this stage is a fixed-epoch probe
(`EPOCHS_COND_PROBE`), not an early-stopped fit.

## Reading tags across stages

`color/auroc/{train,val}` appears in both `enc` and `cond` runs (same metric, two
different training contexts — co-trained-with-encoder vs. standalone probe); compare them
within a run, not across, since the encoder is frozen in one and not the other.
