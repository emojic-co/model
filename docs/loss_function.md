# Loss functions for sparse, missing-label emoji annotation

Current loss is `lse_infonce` (`model/train.py:105-119`), used identically for both
classifier heads — `StyleHead` and `EmojiHead` (`model/model.py`). This doc describes the
dataset shape that constrains the choice of loss, then surveys several loss functions
suited to it — not only the one currently implemented.

## Dataset shape

- `EMOJI_COUNT = 911`, `STYLE_COUNT = 21` (`model/config.py:21-24`).
- Labels are multi-hot float vectors (`model/data.py:82-92`, `emoji2idx` at
  `model/data.py:40`), parsed from a space-separated `"emojis"` field per record
  (`model/data.py:107-125`).
- `data/train.jsonl` has 380,812 rows, `data/eval.jsonl` has 2,000. Measured directly:
  ~1.93 positive emoji labels per example out of 911 possible (~0.2% positive rate).

## Two governing assumptions

**(a) Extreme sparsity.** With ~1.93 positives out of 911 classes per row, any loss that
sums or averages a per-class term uniformly over all 911 classes is dominated by the ~909
negative terms unless corrected. A trivial all-zero-logit model already gets ~99.8%
per-class "accuracy" while carrying no signal.

**(b) Missing labels are not confirmed negatives.** Labels come from a union of
independently-curated keyword sources — `data/keywords.jsonl`, `data/terms.jsonl`,
`data/cldr.jsonl`, `data/emojilib.jsonl` — merged by `mergeKeywords`
(`tools/data/keywords.ts:37-70`). Each source lists only a handful of keywords per emoji,
not an exhaustive relevance judgment, so for a given text an absent emoji is *unknown*,
not *irrelevant*. This is a positive-unlabeled (PU) setting, not label noise.

A loss suited to this dataset should correct for (a) and, ideally, not penalize (b) as
hard as a genuine negative.

## Loss function survey

### Binary cross-entropy (baseline, not used in this repo)

$$\mathcal L=-\frac1C\sum_{i=1}^C\big[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\big]$$

Reference point every other entry below is a correction of. At ~0.2% positive rate the
$(1-y_i)$ term is summed over ~909 of 911 classes per row, so the gradient is
overwhelmingly negative-driven and carries little ranking signal for the ~2 true classes.
Under assumption (b), plain BCE pushes every unlabeled-but-plausibly-true emoji toward 0
with the same weight as a genuine negative — it does not distinguish "confirmed negative"
from "unknown."

### Masked / down-weighted BCE

$$\mathcal L=-\frac1C\sum_{i=1}^C\Big[w^+y_i\log\sigma(z_i)+w^-(1-y_i)\log(1-\sigma(z_i))\Big],\quad w^-\ll w^+$$

(or hard-masks a random subsample of negatives per row instead of a soft weight). Cheap
fix for (a) only — reweights the ~909:2 imbalance without changing what "negative" means.
Does not address (b): every unlabeled emoji is still pushed toward 0, just less hard.

### Multi-positive LSE-InfoNCE (current)

$$\text{row\_loss} = \log\sum_{i=1}^C \exp(z_i/\tau)\ -\ \log\!\!\sum_{i:\,y_i>0}\exp(z_i/\tau)$$

$$\mathcal L=\frac{1}{|\{n:\text{has\_pos}\}|}\sum_{n:\,\text{has\_pos}}\text{row\_loss}^{(n)}$$

with temperature $\tau=$ `INFONCE_TEMP_EMOJI` = `INFONCE_TEMP_STYLE` $=0.7$; rows with zero
positive labels drop out entirely (no gradient). Equivalent to
$-\log\frac{\sum_{i\in P}\exp(z_i/\tau)}{\sum_{i=1}^C\exp(z_i/\tau)}$ — softmax
cross-entropy over all $C$ classes where the positive class mass is the row's full
positive set lumped together, generalizing single-label softmax to multi-positive
retrieval.

Fixes (a) implicitly through the softmax denominator: unlike BCE's independent per-class
terms, the LSE normalizer over all 911 classes means the loss is governed by how
well-separated positives are from the *hardest* negatives, not by raw negative count — a
row with 909 easy, confidently-negative classes contributes little to `all_lse` once
they're well below the positives. Does not fix (b): `all_lse` still sums over every
unlabeled class as if it were a genuine competitor for softmax mass, so an
unlabeled-but-plausibly-true emoji is still penalized whenever it scores comparably to a
labeled positive — same blind spot as BCE, just expressed through a shared denominator
instead of independent per-class terms. This was the repo's first classifier loss,
replaced by ASL and then by `margin_loss`, and has since been reverted back to as the
current loss (see git history) — `docs/model.md:35-39` documents it directly.

### Asymmetric loss (ASL) — tried, replaced

Full formula in `docs/asymmetric_loss.md`. Focal-style $\gamma^+,\gamma^-$ down-weighting
of easy examples plus a margin that creates a gradient dead zone once a negative's
probability clears threshold. Fixes (a) via focal weighting and the sum-over-$C$
(not average) formulation that keeps `EmojiHead`/`StyleHead` gradient magnitudes
comparable despite the ~43x vocab-size mismatch. Does not fix (b): the margin dead zone
silences "confidently negative" predictions regardless of whether the label is a genuine
negative or an unlabeled true positive the model is correctly starting to favor. This repo
tried ASL and replaced it (see git history) with `margin_loss` below — included here as a
rejected-but-instructive comparison, not a live candidate.

### Margin loss — tried, replaced (most recent)

$$\mathcal L=\frac1N\sum_{n=1}^N\sum_{i=1}^C\left[\frac{\mathbb 1[y_i\le0]\,\mathrm{relu}(m+z_i)^2}{|\{i:y_i\le0\}|}+\frac{\mathbb 1[y_i>0]\,\mathrm{relu}(m-z_i)^2}{|\{i:y_i>0\}|}\right]$$

A per-label squared-hinge loss, margin $m=2.0$, applied identically to both heads,
weighted 1:1 (`LOSS_WEIGHT_EMOJI = LOSS_WEIGHT_STYLE = 1`, `config.py:60-61`). Per-row
normalization by `pos_count`/`neg_count` fixes (a) exactly: positive and negative terms
contribute equal total mass per row regardless of the 909:2 skew, without needing a tuned
weight or focal exponent.

**Gap: this loss does not encode (b).** Every one of the ~909 unlabeled classes per row
has `neg_mask = True` and receives full squared-hinge pressure once its logit exceeds
$-m$. `margin_loss` treats "missing" as "confirmed negative," exactly like BCE and ASL
above — just with better-behaved weighting. This was the repo's most recent classifier
loss (following ASL) before being reverted back to `lse_infonce` — see git history.

### Ranking / pairwise (WARP-style, or LSEP)

Pairwise hinge, per positive $p$ and negative $n$ in a row:

$$\mathcal L=\sum_{p:\,y_p>0}\ \sum_{n:\,y_n\le0}\mathrm{relu}(m-z_p+z_n)$$

or the smooth LSEP surrogate $\log\!\big(1+\sum_{p,n}\exp(z_n-z_p)\big)$.

Optimizes the *relative* order of positive vs. negative logits rather than their absolute
values. This is directly aligned with the metric this repo already uses for checkpoint
selection, `emoji/mrr/val` (`mrr` at `model/train.py:120-123`, logged at `train.py:209`,
monitored for early-stop/checkpointing per project memory). It is also a softer,
incidental mitigation of (b): an unlabeled-but-truly-relevant emoji only generates
gradient when it is ranked *below* a labeled positive it should plausibly beat, not merely
for scoring above zero. It does not correct for (b) explicitly, only relaxes its cost.
Failure mode at $C=911$: naive all-pairs cost is $O(|P|\times|N|)\approx 2\times909$ per
row — tractable un-sampled here, though WARP's importance-sampling trick exists precisely
to avoid this at larger vocabularies. Also produces no calibrated probability, which may
matter if downstream consumers (e.g. the web app) want a confidence score rather than a
ranking.

### PU-learning-corrected loss (non-negative PU risk estimator)

Non-negative PU (nnPU, Kiryo et al.) risk, per class $i$, with class prior $\pi_i$ (the
assumed true positive rate for that class) and base loss $\ell$:

$$\hat R_i=\pi_i\,\mathbb E_{p}[\ell(z_i,+1)]+\max\!\Big(0,\ \mathbb E_{u}[\ell(z_i,-1)]-\pi_i\,\mathbb E_{p}[\ell(z_i,-1)]\Big)$$

Treats unlabeled rows as a mixture of hidden positives (at rate $\pi_i$) and true
negatives, and corrects the negative-class risk estimate by subtracting the hidden-positive
contribution, clamped at 0 to avoid the unclamped (biased) PU estimator's known
negative-risk blow-up on flexible models. This is the only entry that models (b) directly,
as a modeling assumption, rather than as a side effect of reweighting or margin softening.

Cost: $\pi_i$ needs estimating per class (e.g. from source-keyword coverage per emoji, or
a single global $\pi$ as a first pass) — a new hyperparameter surface. On a 911-way vocab
with ~1.93 positives/row, most classes see very few positive examples in any batch, so the
prior estimate itself is noisiest exactly where it matters most (rare emojis).

## Comparison

| Loss | Treats missing label as | Pointwise / ranking | Known failure mode here | Status |
|---|---|---|---|---|
| BCE | confirmed negative | pointwise | negative-dominated at 0.2% positive rate | not used |
| Masked/weighted BCE | confirmed negative (less hard) | pointwise | still pushes true-unlabeled toward 0 | not used |
| LSE-InfoNCE | confirmed negative (via shared softmax denominator) | softmax/contrastive | zero-positive rows drop entirely; unlabeled classes still compete for softmax mass | **current** (reverted) |
| ASL | confirmed negative (dead zone) | pointwise | dead zone silences unlabeled true positives too | tried, replaced |
| Margin loss | confirmed negative | pointwise | per-row balanced, but (b) unaddressed | tried, replaced (most recent) |
| Ranking (WARP/LSEP) | confirmed negative, low-cost if unranked | ranking | no calibrated probability; $O(|P||N|)$ pairs/row | not used |
| nnPU-corrected | genuinely unknown (modeled) | pointwise | per-class $\pi_i$ noisy on rare emojis | not used |

## Recommendation

This repo's own history — `lse_infonce` → `asymmetric_loss` → `margin_loss` → back to
`lse_infonce` — has cycled through three different ways of fixing assumption (a): InfoNCE
via denominator normalization over all classes, ASL via focal+margin down-weighting,
`margin_loss` via explicit per-row pos/neg count normalization. None of the three ever
addressed assumption (b): all three still score every unlabeled emoji as a hard negative
target — including the current `lse_infonce`, whose softmax denominator softens but does
not remove that pressure.

Of the two candidates above that most directly close that gap, **ranking/pairwise** is the
lower-risk next step: it needs no new estimated hyperparameter, reuses the existing
multi-hot labels and batch structure, and optimizes the same quantity `emoji/mrr/val`
already measures — the checkpoint metric would still be the right one, arguably more
directly justified than it is under any of the three tried-and-replaced losses above.
**nnPU-corrected** is the more
principled fix for (b) specifically, but it changes what's being optimized (a calibrated
per-class risk under an estimated prior) in a way that's no longer obviously aligned with
a ranking metric — adopting it would need a paired or replacement metric sensitive to the
corrected risk, and per-class $\pi_i$ estimation on this vocab/density is itself an open
sub-problem, not a drop-in change.
