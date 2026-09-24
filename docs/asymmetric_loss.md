# Asymmetric loss (ASL)

`asymmetric_loss` (`model/train.py:112`) is the loss for both classifier
heads — `StyleHead` and `EmojiHead` (`model/model.py`) — replacing the
earlier multi-positive LSE-InfoNCE loss. Config knobs live in
`model/config.py` under `# CLASSIFIER LOSS (ASL)`: `ASL_GAMMA_POS`,
`ASL_GAMMA_NEG_EMOJI` / `ASL_GAMMA_NEG_STYLE`, `ASL_MARGIN_EMOJI` /
`ASL_MARGIN_STYLE`, `LOSS_WEIGHT_EMOJI` / `LOSS_WEIGHT_STYLE`.

## Formula

For logit $z_i$, label $y_i\in\{0,1\}$, and class $i$ of $C$
(`EMOJI_COUNT` = 911, `STYLE_COUNT` = 21), with $\gamma^+,\gamma^-$ the
pos/neg focusing exponents (`ASL_GAMMA_POS`, `ASL_GAMMA_NEG_*`) and $m$ the
negative margin (`ASL_MARGIN_*`):

**Probabilities**

$$p_i=\sigma(z_i),\qquad p_i^-=\operatorname{clip}(1-p_i+m,\;\max=1)$$

$p_i^-$ is the margin-shifted *negative*-class probability — the paper
shifts the positive probability down ($p_m=\operatorname{clip}(p_i-m,\min=0)$)
and uses $1-p_m$ in the negative loss; $p_i^-=1-p_m$ is that same quantity
computed directly as $\operatorname{clip}((1-p_i)+m,\max=1)$, which is why
the code *adds* $m$ to $1-p_i$ rather than subtracting it from $p_i$. The
lower clamp is redundant here since $1-p_i+m>0$ always holds for
$p_i<1,\ m>0$, so the code only applies `.clamp(max=1.0)`.

**Base per-class log-loss** — positives use $p_i$, negatives use the
margin-shifted $p_i^-$:

$$\ell_i=y_i\log p_i+(1-y_i)\log p_i^-$$

**Focal down-weighting** — applied whenever $\gamma^+>0$ or $\gamma^->0$;
$p_{t,i}$ and $\gamma_i$ are treated as constants w.r.t. the gradient (the
code computes them under `torch.no_grad()`):

$$p_{t,i}=y_ip_i+(1-y_i)p_i^-,\qquad \gamma_i=y_i\gamma^++(1-y_i)\gamma^-$$

$$\hat\ell_i=\ell_i\cdot(1-p_{t,i})^{\gamma_i}$$

**Per-sample loss** — sum over classes:

$$L=-\sum_{i=1}^{C}\hat\ell_i$$

**Batch loss** — mean over the $N$ samples in the batch:

$$\mathcal L=\frac1N\sum_{n=1}^{N}L^{(n)}$$

`_step` (`model/train.py:183`) then scales each head's $\mathcal L$ by its
`LOSS_WEIGHT_*` before summing into the total loss.

## Behavior notes

- $\gamma^+=0$: the positive term is plain BCE, $-\log p_i$, unweighted.
- The margin creates a dead zone: once $p_i<m$, $p_i^-$ saturates at 1 and
  $\log p_i^-\to0$ with zero gradient — confidently-correct negatives stop
  contributing once they clear the margin.
- Summing over $C$ (rather than averaging) keeps gradient magnitude from
  shrinking with vocab size; `EmojiHead` (911 classes) and `StyleHead` (21
  classes) would otherwise get ~43x-mismatched gradient contributions into
  the shared `TextEncoder`.
