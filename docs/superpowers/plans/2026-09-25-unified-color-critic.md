# Unified color Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ColorEmbedding`/`ColorCritic`/`CondColorCritic` with one text-conditioned `Critic` module (concat + deep net, no dual-tower dot product) trained on real / shuffled-pairing / random-color negatives, used identically by `LitEncoder`, `LitCondCriticProbe`, and `LitColorGAN`.

**Architecture:** Additive-then-migrate-then-delete. New primitives (`rnd_color_tensor` batched form, config constants, metric names, `PtFile.CRITIC`, the `Critic` class) are added first without touching call sites, so the repo keeps importing cleanly at every commit. Each `Lit*` class then migrates one task at a time (old classes stay in `model.py` until nothing references them). A final cleanup task deletes the now-dead old classes/constants/metrics.

**Tech Stack:** PyTorch, PyTorch Lightning, torchmetrics (`binary_auroc`). No pytest — this repo's `model/test_train_cli.py` is a plain-assert script driven by a Typer CLI (`uv run python model/test_train_cli.py`), not a pytest suite.

**Spec:** `docs/superpowers/specs/2026-09-25-unified-color-critic-design.md`

## Global Constraints

- No pytest suite in this repo. Verify each task with `uv run ruff check .` plus targeted `uv run python -c "..."` snippets or additions to `model/test_train_cli.py`. Do **not** run a real `train --local` / `train gan --local` / `train cond --local` as a per-task check — those are slow, GPU/data-dependent, and reserved for the human's own final verification pass (see end of this plan).
- `model/train.py` aborts on a dirty git tree (`require_clean_tree`) — commit at the end of every task before starting the next.
- `bun run regen` must have been run at least once before any of these Python snippets (it builds the label vocab `model/config.py`/`model/data.py` load at import time). Run it once now if `data/labels.json` / the derived vocab look stale; it's idempotent and cheap.
- No comments or docstrings in new/edited code (repo convention) — match the terse style already in `model/model.py` / `model/train.py`.
- The checkpoint rename (`cond_critic.pt` + `cond_color_embed.pt` -> `critic.pt`) is breaking. No migration path — old files are simply orphaned; a fresh `train --local` regenerates `pt/critic.pt`.
- Every task's file edits are additive relative to the previous task unless the task explicitly says "migrate" or "delete" — don't jump ahead and delete old symbols early, later tasks still need them.

---

### Task 1: Batch `rnd_color_tensor`

**Files:**
- Modify: `model/data.py:65-66`

**Interfaces:**
- Produces: `rnd_color_tensor(n: int, device=None) -> torch.Tensor` — shape `(n, COLOR_DIM)`, values in `{-127.5, ..., 127.5}` (byte-quantized), replacing the old no-arg single-vector form (confirmed unused anywhere in the repo, safe to change signature).

- [ ] **Step 1: Write the failing check**

```bash
uv run python -c "
from model.data import rnd_color_tensor
t = rnd_color_tensor(5)
assert t.shape == (5, 9)
"
```

- [ ] **Step 2: Run it to verify it fails**

Run the command above.
Expected: `TypeError: rnd_color_tensor() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Implement**

In `model/data.py`, replace:

```python
def rnd_color_tensor() -> torch.Tensor:
    return torch.randint(0, 256, (COLOR_DIM,), dtype=torch.float32) - 127.5
```

with:

```python
def rnd_color_tensor(n: int, device=None) -> torch.Tensor:
    return torch.randint(
        0, 256, (n, COLOR_DIM), device=device, dtype=torch.float32) - 127.5
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
import torch
from model.data import rnd_color_tensor
t = rnd_color_tensor(5)
assert t.shape == (5, 9)
assert torch.all((t >= -127.5) & (t <= 127.5))
t2 = rnd_color_tensor(3, device=torch.device('cpu'))
assert t2.shape == (3, 9)
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/data.py
git add model/data.py
git commit -m "$(cat <<'EOF'
Batch rnd_color_tensor for the unified color Critic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `CRITIC_HIDDEN_SIZE` / `COND_CRITIC_RANDOM_WEIGHT` config constants

**Files:**
- Modify: `model/config.py:75-88`

**Interfaces:**
- Produces: `CRITIC_HIDDEN_SIZE = 128`, `COND_CRITIC_RANDOM_WEIGHT = 0.1` (module-level constants in `model/config.py`).

- [ ] **Step 1: Write the failing check**

```bash
uv run python -c "from model.config import CRITIC_HIDDEN_SIZE, COND_CRITIC_RANDOM_WEIGHT"
```

- [ ] **Step 2: Run it to verify it fails**

Expected: `ImportError: cannot import name 'CRITIC_HIDDEN_SIZE'`

- [ ] **Step 3: Implement**

In `model/config.py`, change:

```python
# GAN
# Z_WEIGHT = 0.3
Z_SIZE = 32
GEN_HIDDEN_SIZE = 32
```

to:

```python
# GAN
# Z_WEIGHT = 0.3
Z_SIZE = 32
GEN_HIDDEN_SIZE = 32
CRITIC_HIDDEN_SIZE = 128
```

and change:

```python
GAN_LOSS_ENERGY = 0
GAN_LOSS_COLOR = 0
GAN_LOSS_COND = 1
COND_CRITIC_MISMATCH_WEIGHT = 0.9
```

to:

```python
GAN_LOSS_ENERGY = 0
GAN_LOSS_COLOR = 0
GAN_LOSS_COND = 1
COND_CRITIC_MISMATCH_WEIGHT = 0.9
COND_CRITIC_RANDOM_WEIGHT = 0.1
```

(`GAN_LOSS_COLOR` stays for now — still imported by the not-yet-migrated `model/train.py`; it's deleted in Task 9.)

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
from model.config import CRITIC_HIDDEN_SIZE, COND_CRITIC_RANDOM_WEIGHT
assert CRITIC_HIDDEN_SIZE == 128
assert COND_CRITIC_RANDOM_WEIGHT == 0.1
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/config.py
git add model/config.py
git commit -m "$(cat <<'EOF'
Add CRITIC_HIDDEN_SIZE / COND_CRITIC_RANDOM_WEIGHT config constants

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `AUROC_SHUF`/`AUROC_RAND` and GAN metric names

**Files:**
- Modify: `model/metric.py`

**Interfaces:**
- Produces: `Metric.AUROC_SHUF`, `Metric.AUROC_RAND`; `GanMetric.COND_AUROC_RAND`, `GanMetric.COND_MEAN_SCORE_RANDOM`.

- [ ] **Step 1: Write the failing check**

```bash
uv run python -c "from model.metric import Metric, GanMetric; Metric.AUROC_SHUF; Metric.AUROC_RAND; GanMetric.COND_AUROC_RAND; GanMetric.COND_MEAN_SCORE_RANDOM"
```

- [ ] **Step 2: Run it to verify it fails**

Expected: `AttributeError: AUROC_SHUF`

- [ ] **Step 3: Implement**

In `model/metric.py`, change:

```python
class Metric(StrEnum):
    ACC_1 = "acc@1"
    RATE = "rate"
    MAE = "mae"
    MRR = "mrr"
    R2 = "r2"
    AUROC = "auroc"
    LOSS = "loss"
```

to:

```python
class Metric(StrEnum):
    ACC_1 = "acc@1"
    RATE = "rate"
    MAE = "mae"
    MRR = "mrr"
    R2 = "r2"
    AUROC = "auroc"
    AUROC_SHUF = "auroc_shuf"
    AUROC_RAND = "auroc_rand"
    LOSS = "loss"
```

(`AUROC` stays for now — deleted in Task 9 once nothing logs it.)

And change:

```python
class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS_COND = "gan/gen_loss_cond"
    GEN_LOSS_COLOR = "gan/gen_loss_color"
    COND_LOSS = "gan/cond_loss"
    COND_AUROC_GEN = "gan/cond_auroc_gen"
    COND_AUROC_SHUF = "gan/cond_auroc_shuf"
    COND_MEAN_SCORE_REAL = "gan/cond_mean_score_real"
    COND_MEAN_SCORE_FAKE = "gan/cond_mean_score_fake"
    COND_MEAN_SCORE_WRONG = "gan/cond_mean_score_wrong"
    COLOR_LOSS = "gan/color_loss"
    COLOR_AUROC = "gan/color_auroc"
```

to:

```python
class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS_COND = "gan/gen_loss_cond"
    GEN_LOSS_COLOR = "gan/gen_loss_color"
    COND_LOSS = "gan/cond_loss"
    COND_AUROC_GEN = "gan/cond_auroc_gen"
    COND_AUROC_SHUF = "gan/cond_auroc_shuf"
    COND_AUROC_RAND = "gan/cond_auroc_rand"
    COND_MEAN_SCORE_REAL = "gan/cond_mean_score_real"
    COND_MEAN_SCORE_FAKE = "gan/cond_mean_score_fake"
    COND_MEAN_SCORE_WRONG = "gan/cond_mean_score_wrong"
    COND_MEAN_SCORE_RANDOM = "gan/cond_mean_score_random"
    COLOR_LOSS = "gan/color_loss"
    COLOR_AUROC = "gan/color_auroc"
```

(`GEN_LOSS_COLOR`, `COLOR_LOSS`, `COLOR_AUROC` stay for now — deleted in Task 9.)

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
from model.metric import Metric, GanMetric
assert Metric.AUROC_SHUF == 'auroc_shuf'
assert Metric.AUROC_RAND == 'auroc_rand'
assert GanMetric.COND_AUROC_RAND == 'gan/cond_auroc_rand'
assert GanMetric.COND_MEAN_SCORE_RANDOM == 'gan/cond_mean_score_random'
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/metric.py
git add model/metric.py
git commit -m "$(cat <<'EOF'
Add AUROC_SHUF/AUROC_RAND and GAN random-negative metric names

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add `PtFile.CRITIC`

**Files:**
- Modify: `files.py:32-49`

**Interfaces:**
- Produces: `PtFile.CRITIC` (value `"critic.pt"`), `CRITIC_PT` constant.

- [ ] **Step 1: Write the failing check**

```bash
uv run python -c "from files import PtFile, CRITIC_PT"
```

- [ ] **Step 2: Run it to verify it fails**

Expected: `ImportError: cannot import name 'CRITIC_PT'`

- [ ] **Step 3: Implement**

In `files.py`, change:

```python
class PtFile(StrEnum):
    ENC = "enc.pt"
    STYLE = "style.pt"
    EMOJI = "emoji.pt"
    GEN = "gen.pt"
    COND_CRITIC = "cond_critic.pt"
    COND_COLOR_EMBED = "cond_color_embed.pt"

    def in_dir(self, pt_dir: Path) -> Path:
        return pt_dir / self.value


ENC_PT = PtFile.ENC.in_dir(PT_DIR)
STYLE_PT = PtFile.STYLE.in_dir(PT_DIR)
EMOJI_PT = PtFile.EMOJI.in_dir(PT_DIR)
GEN_PT = PtFile.GEN.in_dir(PT_DIR)
COND_CRITIC_PT = PtFile.COND_CRITIC.in_dir(PT_DIR)
COND_COLOR_EMBED_PT = PtFile.COND_COLOR_EMBED.in_dir(PT_DIR)
```

to:

```python
class PtFile(StrEnum):
    ENC = "enc.pt"
    STYLE = "style.pt"
    EMOJI = "emoji.pt"
    GEN = "gen.pt"
    CRITIC = "critic.pt"
    COND_CRITIC = "cond_critic.pt"
    COND_COLOR_EMBED = "cond_color_embed.pt"

    def in_dir(self, pt_dir: Path) -> Path:
        return pt_dir / self.value


ENC_PT = PtFile.ENC.in_dir(PT_DIR)
STYLE_PT = PtFile.STYLE.in_dir(PT_DIR)
EMOJI_PT = PtFile.EMOJI.in_dir(PT_DIR)
GEN_PT = PtFile.GEN.in_dir(PT_DIR)
CRITIC_PT = PtFile.CRITIC.in_dir(PT_DIR)
COND_CRITIC_PT = PtFile.COND_CRITIC.in_dir(PT_DIR)
COND_COLOR_EMBED_PT = PtFile.COND_COLOR_EMBED.in_dir(PT_DIR)
```

(`COND_CRITIC`/`COND_COLOR_EMBED` stay for now — deleted in Task 9.)

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
from files import PtFile, CRITIC_PT, PT_DIR
assert PtFile.CRITIC.value == 'critic.pt'
assert CRITIC_PT == PT_DIR / 'critic.pt'
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check files.py
git add files.py
git commit -m "$(cat <<'EOF'
Add PtFile.CRITIC checkpoint path

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add the unified `Critic` module

**Files:**
- Modify: `model/model.py` (import block + new class appended at end of file, after `CondColorCritic`)
- Modify: `model/test_train_cli.py:25-30`

**Interfaces:**
- Consumes: `CRITIC_HIDDEN_SIZE` (Task 2), `COLOR_DIM` (`model/data.py`, already imported in `model.py`), `COLOR_SHIFT` (`model/color.py`, already imported), `cblk`/`sn` (already defined/imported in `model.py`).
- Produces: `Critic` (`nn.Module`) — `Critic()(cond: Tensor[N, EMBED_SIZE_TEXT], colors: Tensor[N, COLOR_DIM]) -> Tensor[N, 1]`.

- [ ] **Step 1: Write the failing test**

In `model/test_train_cli.py`, replace:

```python
def test_colorcritic_forward_shape():
    from model.model import ColorEmbedding, CondColorCritic

    color_embedding = ColorEmbedding()(torch.zeros(5, 9))
    score = CondColorCritic()(torch.zeros(5, EMBED_SIZE_TEXT), color_embedding)
    assert score.shape == (5, 1)
```

with:

```python
def test_colorcritic_forward_shape():
    from model.model import Critic

    score = Critic()(torch.zeros(5, EMBED_SIZE_TEXT), torch.zeros(5, 9))
    assert score.shape == (5, 1)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run python -c "import model.test_train_cli as t; t.test_colorcritic_forward_shape()"
```

Expected: `ImportError: cannot import name 'Critic'`

- [ ] **Step 3: Implement**

In `model/model.py`, add `CRITIC_HIDDEN_SIZE` to the config import block — change:

```python
from model.config import (
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_COLOR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_HIDDEN_SIZE,
    RELU_SLOPE,
    Z_SIZE,
)
```

to:

```python
from model.config import (
    CRITIC_HIDDEN_SIZE,
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_COLOR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_HIDDEN_SIZE,
    RELU_SLOPE,
    Z_SIZE,
)
```

(`EMBED_SIZE_COLOR` stays for now — `ColorEmbedding`/`CondColorCritic` still use it; deleted in Task 9.)

Then append at the end of `model/model.py`, after `CondColorCritic`:

```python


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

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "import model.test_train_cli as t; t.test_colorcritic_forward_shape(); print('ok')"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/model.py model/test_train_cli.py
git add model/model.py model/test_train_cli.py
git commit -m "$(cat <<'EOF'
Add unified Critic: concat(text, color) through a deep net

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `_critic_probe_step` helper + migrate `LitEncoder`

**Files:**
- Modify: `model/train.py` (imports; new helper function after `lse_infonce`; `LitEncoder.__init__`, `_step`, `configure_optimizers`)
- Modify: `model/test_train_cli.py` (new test)

**Interfaces:**
- Consumes: `Critic` (Task 5), `rnd_color_tensor` (Task 1), `Metric.AUROC_SHUF`/`Metric.AUROC_RAND` (Task 3), `binary_auroc`, `relu` (already imported in `train.py`).
- Produces: `_critic_probe_step(critic: Critic, cond: Tensor[N, EMBED_SIZE_TEXT], colors: Tensor[N, COLOR_DIM]) -> tuple[Tensor, Tensor, Tensor]` (loss, auroc_shuf, auroc_rand — all 0-dim), used by Task 7 too. `LitEncoder.critic: Critic` (was `cond_critic`/`color_embed`).

- [ ] **Step 1: Write the failing test**

In `model/test_train_cli.py`, add (after `test_colorcritic_forward_shape`):

```python
def test_critic_probe_step_shapes():
    from model.data import rnd_color_tensor
    from model.model import Critic
    from model.train import _critic_probe_step

    critic = Critic()
    cond = torch.randn(6, EMBED_SIZE_TEXT)
    colors = rnd_color_tensor(6)
    loss, auroc_shuf, auroc_rand = _critic_probe_step(critic, cond, colors)
    assert loss.shape == ()
    assert auroc_shuf.shape == ()
    assert auroc_rand.shape == ()
```

And add the call in `main()`, right after `test_colorcritic_forward_shape()`:

```python
    test_critic_probe_step_shapes()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run python -c "import model.test_train_cli as t; t.test_critic_probe_step_shapes()"
```

Expected: `ImportError: cannot import name '_critic_probe_step' from 'model.train'`

- [ ] **Step 3: Implement**

In `model/train.py`, add `rnd_color_tensor` to the `model.data` import — change:

```python
from model.data import (
    eval_data_loader,
    eval_ds,
    sample_colors_tensor,
    train_data_loader,
    train_ds,
)
```

to:

```python
from model.data import (
    eval_data_loader,
    eval_ds,
    rnd_color_tensor,
    sample_colors_tensor,
    train_data_loader,
    train_ds,
)
```

Add `Critic` to the `model.model` import (keep the old three for now, still used by `LitColorGAN`/`LitCondCriticProbe`) — change:

```python
from model.model import (
    ColorCritic,
    ColorEmbedding,
    ColorGen,
    CondColorCritic,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
```

to:

```python
from model.model import (
    ColorCritic,
    ColorEmbedding,
    ColorGen,
    CondColorCritic,
    Critic,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
```

Add the helper right after `lse_infonce` (which ends with `return row_loss[has_pos].mean()`), before `class LitEncoder`:

```python
def _critic_probe_step(
    critic: Critic, cond: torch.Tensor, colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = colors.shape[0]
    shuffled = colors[torch.randperm(n, device=colors.device)]
    random_colors = rnd_color_tensor(n, device=colors.device)

    pair = torch.cat([colors, shuffled, random_colors], dim=0)
    cond_pair = torch.cat([cond, cond, cond], dim=0)
    score = critic(cond_pair, pair)
    real, shuf_score, rand_score = score.chunk(3, dim=0)

    loss = relu(1 - real).mean() \
        + relu(1 + shuf_score).mean() \
        + relu(1 + rand_score).mean()

    target = torch.cat([score.new_ones(n), score.new_zeros(n)])
    auroc_shuf = binary_auroc(
        torch.cat([real, shuf_score], dim=0).detach().squeeze(-1), target.long())
    auroc_rand = binary_auroc(
        torch.cat([real, rand_score], dim=0).detach().squeeze(-1), target.long())

    return loss, auroc_shuf, auroc_rand
```

In `LitEncoder.__init__`, change:

```python
        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()
        self.cond_critic = CondColorCritic()
        self.color_embed = ColorEmbedding()

        self.train_dataset = None
```

to:

```python
        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()
        self.critic = Critic()

        self.train_dataset = None
```

In `LitEncoder._step`, change:

```python
        loss_cond_critic = enc.new_zeros(())
        n_c = int(has_color.sum())
        if n_c > 1:
            cond_c = enc[has_color]
            colors_c = colors[has_color]
            perm = torch.randperm(n_c, device=colors_c.device)
            fake = colors_c[perm]

            pair = torch.cat([colors_c, fake], dim=0)
            cond_pair = torch.cat([cond_c, cond_c], dim=0)
            score = self.cond_critic(cond_pair, self.color_embed(pair))
            real, fake_score = score.chunk(2, dim=0)

            loss_cond_critic = relu(1 - real).mean() + relu(1 + fake_score).mean()

            target = torch.cat([score.new_ones(n_c), score.new_zeros(n_c)])
            auroc = binary_auroc(score.detach().squeeze(-1), target.long())
            self._log(
                named_metric(LogStage.ENC, Source.COLOR, Metric.AUROC, split),
                auroc, n_c)

        return (
            loss_style
            + loss_emoji
            + LOSS_WEIGHT_COLOR * loss_cond_critic
        )
```

to:

```python
        loss_critic = enc.new_zeros(())
        n_c = int(has_color.sum())
        if n_c > 1:
            cond_c = enc[has_color]
            colors_c = colors[has_color]
            loss_critic, auroc_shuf, auroc_rand = _critic_probe_step(
                self.critic, cond_c, colors_c)
            self._log(
                named_metric(LogStage.ENC, Source.COLOR, Metric.AUROC_SHUF, split),
                auroc_shuf, n_c)
            self._log(
                named_metric(LogStage.ENC, Source.COLOR, Metric.AUROC_RAND, split),
                auroc_rand, n_c)

        return (
            loss_style
            + loss_emoji
            + LOSS_WEIGHT_COLOR * loss_critic
        )
```

In `LitEncoder.configure_optimizers`, change:

```python
    def configure_optimizers(self):
        params = (
            list(self.enc.parameters())
            + list(self.style.parameters())
            + list(self.emoji.parameters())
            + list(self.cond_critic.parameters())
            + list(self.color_embed.parameters())
        )
        return optim.Adam(params, lr=LR_ENCODER)
```

to:

```python
    def configure_optimizers(self):
        params = (
            list(self.enc.parameters())
            + list(self.style.parameters())
            + list(self.emoji.parameters())
            + list(self.critic.parameters())
        )
        return optim.Adam(params, lr=LR_ENCODER)
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
import model.test_train_cli as t
t.test_colorcritic_forward_shape()
t.test_critic_probe_step_shapes()
t.test_litencoder_builds_all_heads()
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/train.py model/test_train_cli.py
git add model/train.py model/test_train_cli.py
git commit -m "$(cat <<'EOF'
Migrate LitEncoder to the unified Critic (real/shuffled/random split)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Migrate `LitCondCriticProbe`

**Files:**
- Modify: `model/train.py` (`LitCondCriticProbe` class)
- Modify: `model/test_train_cli.py` (new test)

**Interfaces:**
- Consumes: `Critic` (Task 5), `_critic_probe_step` (Task 6), `Metric.AUROC_SHUF`/`Metric.AUROC_RAND` (Task 3).
- Produces: `LitCondCriticProbe.critic: Critic` (was `critic: CondColorCritic` + `color_embed: ColorEmbedding`).

- [ ] **Step 1: Write the failing test**

In `model/test_train_cli.py`, add (after `test_critic_probe_step_shapes`):

```python
def test_litcondcriticprobe_builds():
    m = T.LitCondCriticProbe()
    assert hasattr(m, "critic")
    assert not hasattr(m, "color_embed")
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.SGD)
```

And add the call in `main()`, right after `test_critic_probe_step_shapes()`:

```python
    test_litcondcriticprobe_builds()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run python -c "import model.test_train_cli as t; t.test_litcondcriticprobe_builds()"
```

Expected: `AssertionError` (`m.color_embed` still exists on the unmigrated class).

- [ ] **Step 3: Implement**

In `model/train.py`, replace the whole `LitCondCriticProbe` class:

```python
class LitCondCriticProbe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.critic = CondColorCritic()
        self.color_embed = ColorEmbedding()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        cond, colors = batch
        perm = torch.randperm(colors.shape[0], device=colors.device)
        fake = colors[perm]

        pair = torch.cat([colors, fake], dim=0)
        cond_pair = torch.cat([cond, cond], dim=0)
        score = self.critic(cond_pair, self.color_embed(pair))
        real, fake_score = score.chunk(2, dim=0)

        loss = relu(1 - real).mean() + relu(1 + fake_score).mean()

        n = colors.shape[0]
        target = torch.cat([score.new_ones(n), score.new_zeros(n)])
        auroc = binary_auroc(score.detach().squeeze(-1), target.long())
        return loss, auroc

    def training_step(self, batch, batch_idx):
        loss, auroc = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC, Split.TRAIN),
            auroc, on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, auroc = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC, Split.VAL),
            auroc, on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        params = list(self.critic.parameters()) + \
            list(self.color_embed.parameters())
        return optim.SGD(params, lr=LR_GAN_COND_CRITIC)
```

with:

```python
class LitCondCriticProbe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.critic = Critic()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        cond, colors = batch
        return _critic_probe_step(self.critic, cond, colors)

    def training_step(self, batch, batch_idx):
        loss, auroc_shuf, auroc_rand = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_SHUF, Split.TRAIN),
            auroc_shuf, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_RAND, Split.TRAIN),
            auroc_rand, on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, auroc_shuf, auroc_rand = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_SHUF, Split.VAL),
            auroc_shuf, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_RAND, Split.VAL),
            auroc_rand, on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        return optim.SGD(self.critic.parameters(), lr=LR_GAN_COND_CRITIC)
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
import model.test_train_cli as t
t.test_litcondcriticprobe_builds()
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/train.py model/test_train_cli.py
git add model/train.py model/test_train_cli.py
git commit -m "$(cat <<'EOF'
Migrate LitCondCriticProbe to the unified Critic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Migrate `LitColorGAN` (merge into one critic, four groups)

**Files:**
- Modify: `model/train.py` (imports; `LitColorGAN`; `_load_cond_critic` -> `_load_critic`; `_train_encoder`; `_train_gan`; `_run_local` call sites)
- Modify: `model/test_train_cli.py` (new test)

**Interfaces:**
- Consumes: `Critic` (Task 5), `rnd_color_tensor` (Task 1), `COND_CRITIC_RANDOM_WEIGHT` (Task 2), `GanMetric.COND_AUROC_RAND`/`COND_MEAN_SCORE_RANDOM` (Task 3), `PtFile.CRITIC` (Task 4).
- Produces: `LitColorGAN.__init__(self, critic: Critic)` (was `(critic: CondColorCritic, color_embed: ColorEmbedding)`); `_load_critic(pt_dir: Path) -> Critic` (was `_load_cond_critic(pt_dir) -> tuple[CondColorCritic, ColorEmbedding]`); `_train_gan(enc, enc_path, critic: Critic, ds, val_ds, out_dir) -> LitColorGAN` (drops the `color_embed` parameter).

- [ ] **Step 1: Write the failing test**

In `model/test_train_cli.py`, add (after `test_litcondcriticprobe_builds`):

```python
def test_litcolorgan_builds():
    from model.model import Critic

    m = T.LitColorGAN(Critic())
    assert hasattr(m, "gen") and hasattr(m, "critic")
    assert not hasattr(m, "color_critic")
    opts = m.configure_optimizers()
    assert len(opts) == 2
```

And add the call in `main()`, right after `test_litcondcriticprobe_builds()`:

```python
    test_litcolorgan_builds()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run python -c "import model.test_train_cli as t; t.test_litcolorgan_builds()"
```

Expected: `TypeError: LitColorGAN.__init__() missing 1 required positional argument: 'color_embed'`

- [ ] **Step 3: Implement**

In `model/train.py`, finalize the `model.model` import (drop the three old classes, they're unused in this file after this task) — change:

```python
from model.model import (
    ColorCritic,
    ColorEmbedding,
    ColorGen,
    CondColorCritic,
    Critic,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
```

to:

```python
from model.model import (
    ColorGen,
    Critic,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
```

In the `model.config` import block, drop `GAN_LOSS_COLOR` and `LR_GAN_CRITIC` (both unused after this task), add `COND_CRITIC_RANDOM_WEIGHT` — change:

```python
from model.config import (
    COND_CRITIC_MISMATCH_WEIGHT,
    CONFIG_NAME,
    EARLY_STOP_MIN_DELTA_GAN,
    EARLY_STOP_PATIENCE_ENCODER,
    EARLY_STOP_PATIENCE_GAN,
    ENERGY_TRAIN_SAMPLE_SIZE,
    ENERGY_VAL_SAMPLE_SIZE,
    EPOCHS_COND_PROBE,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_LOSS_COLOR,
    GAN_LOSS_COND,
    GAN_LOSS_ENERGY,
    GRAD_CLIP_CRITIC,
    GRAD_CLIP_GEN,
    INFONCE_TEMP_EMOJI,
    INFONCE_TEMP_STYLE,
    LOSS_WEIGHT_COLOR,
    LOSS_WEIGHT_EMOJI,
    LOSS_WEIGHT_STYLE,
    LR_ENCODER,
    LR_GAN_COND_CRITIC,
    LR_GAN_CRITIC,
    LR_GAN_GEN,
    SAMPLING_RATE_MAX,
    SAMPLING_RATE_MIN,
    SAMPLING_SOURCES,
    SEED,
    TASK_BATCH_SIZE,
    VAL_CHECK_INTERVAL,
)
```

to:

```python
from model.config import (
    COND_CRITIC_MISMATCH_WEIGHT,
    COND_CRITIC_RANDOM_WEIGHT,
    CONFIG_NAME,
    EARLY_STOP_MIN_DELTA_GAN,
    EARLY_STOP_PATIENCE_ENCODER,
    EARLY_STOP_PATIENCE_GAN,
    ENERGY_TRAIN_SAMPLE_SIZE,
    ENERGY_VAL_SAMPLE_SIZE,
    EPOCHS_COND_PROBE,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_LOSS_COND,
    GAN_LOSS_ENERGY,
    GRAD_CLIP_CRITIC,
    GRAD_CLIP_GEN,
    INFONCE_TEMP_EMOJI,
    INFONCE_TEMP_STYLE,
    LOSS_WEIGHT_COLOR,
    LOSS_WEIGHT_EMOJI,
    LOSS_WEIGHT_STYLE,
    LR_ENCODER,
    LR_GAN_COND_CRITIC,
    LR_GAN_GEN,
    SAMPLING_RATE_MAX,
    SAMPLING_RATE_MIN,
    SAMPLING_SOURCES,
    SEED,
    TASK_BATCH_SIZE,
    VAL_CHECK_INTERVAL,
)
```

Replace `LitColorGAN.__init__`:

```python
class LitColorGAN(pl.LightningModule):
    def __init__(self, critic: CondColorCritic, color_embed: ColorEmbedding):
        super().__init__()

        self.gen = ColorGen()
        self.critic = critic
        self.color_embed = color_embed
        self.color_critic = ColorCritic()
        self.color_critic_embed = ColorEmbedding()

        self.automatic_optimization = False
        self._val_cond: list[torch.Tensor] = []
        self._val_real: list[torch.Tensor] = []
```

with:

```python
class LitColorGAN(pl.LightningModule):
    def __init__(self, critic: Critic):
        super().__init__()

        self.gen = ColorGen()
        self.critic = critic

        self.automatic_optimization = False
        self._val_cond: list[torch.Tensor] = []
        self._val_real: list[torch.Tensor] = []
```

Replace the body of `training_step` (everything from `opt_gen, opt_critic, opt_color_critic = self.optimizers()` through the final `self.log(GanMetric.ENERGY_TRAIN, ...)` call):

```python
        opt_gen, opt_critic, opt_color_critic = self.optimizers()  # type: ignore

        fake = self.gen(cond)

        # CRITIC
        color_embed_real = self.color_embed(colors)
        color_embed_fake = self.color_embed(fake.detach())
        cond_wrong = cond[torch.randperm(cond.shape[0], device=cond.device)]

        real = self.critic(cond, color_embed_real)
        fake_score = self.critic(cond, color_embed_fake)
        wrong_score = self.critic(cond_wrong, color_embed_real)

        loss_critic = relu(1 - real).mean() \
            + COND_CRITIC_MISMATCH_WEIGHT * relu(1 + fake_score).mean() \
            + (1 - COND_CRITIC_MISMATCH_WEIGHT) * relu(1 + wrong_score).mean()

        n = colors.shape[0]
        auroc_target = torch.cat([real.new_ones(n), real.new_zeros(n)])
        auroc_gen = binary_auroc(
            torch.cat([real, fake_score], dim=0).detach().squeeze(-1),
            auroc_target.long())
        auroc_shuf = binary_auroc(
            torch.cat([real, wrong_score], dim=0).detach().squeeze(-1),
            auroc_target.long())

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(
            opt_critic,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm")

        opt_critic.step()

        # COLOR CRITIC (unconditional)
        pair = torch.cat([colors, fake.detach()], dim=0)
        color_score = self.color_critic(self.color_critic_embed(pair))
        color_real, color_fake_score = color_score.chunk(2, dim=0)

        loss_color_critic = \
            relu(1 - color_real).mean() + relu(1 + color_fake_score).mean()

        color_auroc = binary_auroc(
            color_score.detach().squeeze(-1), auroc_target.long())

        opt_color_critic.zero_grad()
        self.manual_backward(loss_color_critic)
        self.clip_gradients(
            opt_color_critic,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm")

        opt_color_critic.step()

        # GENERATOR
        gen_score = self.critic(cond, self.color_embed(fake))
        gen_color_score = self.color_critic(self.color_critic_embed(fake))
        loss_energy = energy_distance(
            rgb_to_oklab(_energy_subsample(fake, ENERGY_TRAIN_SAMPLE_SIZE)),
            rgb_to_oklab(_energy_subsample(colors, ENERGY_TRAIN_SAMPLE_SIZE)))

        loss_gen_critic = -gen_score.mean()
        loss_gen_color_critic = -gen_color_score.mean()

        loss_gen = \
            GAN_LOSS_COND * loss_gen_critic \
            + GAN_LOSS_ENERGY * loss_energy \
            + GAN_LOSS_COLOR * loss_gen_color_critic

        opt_gen.zero_grad()

        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm")

        opt_gen.step()

        self.log(GanMetric.COND_LOSS, loss_critic, prog_bar=True)
        self.log(GanMetric.COND_AUROC_GEN, auroc_gen, prog_bar=True)
        self.log(GanMetric.COND_AUROC_SHUF, auroc_shuf, prog_bar=True)
        self.log(
            GanMetric.COND_MEAN_SCORE_REAL,
            real.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_FAKE,
            fake_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_WRONG,
            wrong_score.detach().mean(), prog_bar=False)
        self.log(GanMetric.COLOR_LOSS, loss_color_critic, prog_bar=True)
        self.log(GanMetric.COLOR_AUROC, color_auroc, prog_bar=True)
        self.log(
            GanMetric.GEN_LOSS_COND,
            loss_gen_critic, prog_bar=True)
        self.log(
            GanMetric.GEN_LOSS_COLOR,
            loss_gen_color_critic, prog_bar=True)
        self.log(GanMetric.ENERGY_TRAIN, loss_energy, prog_bar=True)
```

with:

```python
        opt_gen, opt_critic = self.optimizers()  # type: ignore

        fake = self.gen(cond)

        # CRITIC
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

        n = colors.shape[0]
        auroc_target = torch.cat([real.new_ones(n), real.new_zeros(n)])
        auroc_gen = binary_auroc(
            torch.cat([real, fake_score], dim=0).detach().squeeze(-1),
            auroc_target.long())
        auroc_shuf = binary_auroc(
            torch.cat([real, wrong_score], dim=0).detach().squeeze(-1),
            auroc_target.long())
        auroc_rand = binary_auroc(
            torch.cat([real, random_score], dim=0).detach().squeeze(-1),
            auroc_target.long())

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(
            opt_critic,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm")

        opt_critic.step()

        # GENERATOR
        gen_score = self.critic(cond, fake)
        loss_energy = energy_distance(
            rgb_to_oklab(_energy_subsample(fake, ENERGY_TRAIN_SAMPLE_SIZE)),
            rgb_to_oklab(_energy_subsample(colors, ENERGY_TRAIN_SAMPLE_SIZE)))

        loss_gen_critic = -gen_score.mean()

        loss_gen = \
            GAN_LOSS_COND * loss_gen_critic \
            + GAN_LOSS_ENERGY * loss_energy

        opt_gen.zero_grad()

        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm")

        opt_gen.step()

        self.log(GanMetric.COND_LOSS, loss_critic, prog_bar=True)
        self.log(GanMetric.COND_AUROC_GEN, auroc_gen, prog_bar=True)
        self.log(GanMetric.COND_AUROC_SHUF, auroc_shuf, prog_bar=True)
        self.log(GanMetric.COND_AUROC_RAND, auroc_rand, prog_bar=True)
        self.log(
            GanMetric.COND_MEAN_SCORE_REAL,
            real.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_FAKE,
            fake_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_WRONG,
            wrong_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_RANDOM,
            random_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.GEN_LOSS_COND,
            loss_gen_critic, prog_bar=True)
        self.log(GanMetric.ENERGY_TRAIN, loss_energy, prog_bar=True)
```

Replace `configure_optimizers`:

```python
    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=LR_GAN_GEN)
        opt_critic = optim.SGD(
            list(self.critic.parameters()) + list(self.color_embed.parameters()),
            lr=LR_GAN_COND_CRITIC)
        opt_color_critic = optim.SGD(
            list(self.color_critic.parameters())
            + list(self.color_critic_embed.parameters()),
            lr=LR_GAN_CRITIC)

        # opt_gen = optim.Adam(
        #     self.gen.parameters(),
        #     lr=LR_GAN_GEN,
        #     betas=(0.5, 0.999))

        # opt_critic = optim.Adam(
        #     self.critic.parameters(),
        #     lr=LR_GAN_CRITIC,
        #     betas=(0.5, 0.999))

        return [opt_gen, opt_critic, opt_color_critic]
```

with:

```python
    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=LR_GAN_GEN)
        opt_critic = optim.SGD(self.critic.parameters(), lr=LR_GAN_COND_CRITIC)

        # opt_gen = optim.Adam(
        #     self.gen.parameters(),
        #     lr=LR_GAN_GEN,
        #     betas=(0.5, 0.999))

        # opt_critic = optim.Adam(
        #     self.critic.parameters(),
        #     lr=LR_GAN_COND_CRITIC,
        #     betas=(0.5, 0.999))

        return [opt_gen, opt_critic]
```

Replace `_load_cond_critic`:

```python
def _load_cond_critic(pt_dir: Path) -> tuple[CondColorCritic, ColorEmbedding]:
    critic_path = PtFile.COND_CRITIC.in_dir(pt_dir)
    embed_path = PtFile.COND_COLOR_EMBED.in_dir(pt_dir)
    critic = CondColorCritic()
    color_embed = ColorEmbedding()
    if not critic_path.exists() or not embed_path.exists():
        return critic, color_embed
    try:
        _load(critic, critic_path)  # type: ignore
        _load(color_embed, embed_path)  # type: ignore
        return critic, color_embed
    except Exception:
        print(
            f"{critic_path}: does not match CondColorCritic/ColorEmbedding, "
            "falling back to a fresh critic",
            flush=True,
        )
        return CondColorCritic(), ColorEmbedding()
```

with:

```python
def _load_critic(pt_dir: Path) -> Critic:
    critic_path = PtFile.CRITIC.in_dir(pt_dir)
    critic = Critic()
    if not critic_path.exists():
        return critic
    try:
        return _load(critic, critic_path)  # type: ignore
    except Exception:
        print(
            f"{critic_path}: does not match Critic, "
            "falling back to a fresh critic",
            flush=True,
        )
        return Critic()
```

In `_train_encoder`, change:

```python
    save_pt(mod.enc.state_dict(), PtFile.ENC.in_dir(out_dir), stage="enc")
    save_pt(mod.style.state_dict(), PtFile.STYLE.in_dir(out_dir), stage="enc")
    save_pt(mod.emoji.state_dict(), PtFile.EMOJI.in_dir(out_dir), stage="enc")
    save_pt(
        mod.cond_critic.state_dict(),
        PtFile.COND_CRITIC.in_dir(out_dir), stage="enc")
    save_pt(
        mod.color_embed.state_dict(),
        PtFile.COND_COLOR_EMBED.in_dir(out_dir), stage="enc")

    return mod
```

to:

```python
    save_pt(mod.enc.state_dict(), PtFile.ENC.in_dir(out_dir), stage="enc")
    save_pt(mod.style.state_dict(), PtFile.STYLE.in_dir(out_dir), stage="enc")
    save_pt(mod.emoji.state_dict(), PtFile.EMOJI.in_dir(out_dir), stage="enc")
    save_pt(
        mod.critic.state_dict(),
        PtFile.CRITIC.in_dir(out_dir), stage="enc")

    return mod
```

In `_train_gan`, change the signature and the two `LitColorGAN(...)` constructions:

```python
def _train_gan(
    enc: TextEncoder, enc_path: Path, critic: CondColorCritic,
    color_embed: ColorEmbedding,
    ds, val_ds, out_dir: Path,
) -> LitColorGAN:
```

to:

```python
def _train_gan(
    enc: TextEncoder, enc_path: Path, critic: Critic,
    ds, val_ds, out_dir: Path,
) -> LitColorGAN:
```

and change:

```python
    gan = LitColorGAN(critic, color_embed)
    trainer.fit(gan, gan_dl, val_dl)

    if ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(
            ckpt.best_model_path, critic=CondColorCritic(),
            color_embed=ColorEmbedding(),
        )
```

to:

```python
    gan = LitColorGAN(critic)
    trainer.fit(gan, gan_dl, val_dl)

    if ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(
            ckpt.best_model_path, critic=Critic(),
        )
```

In `_run_local`, change:

```python
    if stage == Stage.gan:
        if _pt_files_ok(_DEFAULT_PT):
            enc_path = PtFile.ENC.in_dir(_DEFAULT_PT)
            enc = _load(TextEncoder(), enc_path)
            critic, color_embed = _load_cond_critic(_DEFAULT_PT)
            _train_gan(
                enc, enc_path, critic, color_embed,  # type: ignore
                train_ds(mix_sources=False),
                eval_ds(),
                _DEFAULT_PT,
            )
```

to:

```python
    if stage == Stage.gan:
        if _pt_files_ok(_DEFAULT_PT):
            enc_path = PtFile.ENC.in_dir(_DEFAULT_PT)
            enc = _load(TextEncoder(), enc_path)
            critic = _load_critic(_DEFAULT_PT)
            _train_gan(
                enc, enc_path, critic,  # type: ignore
                train_ds(mix_sources=False),
                eval_ds(),
                _DEFAULT_PT,
            )
```

and change:

```python
    critic, color_embed = _load_cond_critic(_DEFAULT_PT)
    _train_gan(
        mod.enc, PtFile.ENC.in_dir(_DEFAULT_PT), critic, color_embed,
        train_ds(mix_sources=False),
        eval_ds(),
        _DEFAULT_PT,
    )
```

to:

```python
    critic = _load_critic(_DEFAULT_PT)
    _train_gan(
        mod.enc, PtFile.ENC.in_dir(_DEFAULT_PT), critic,
        train_ds(mix_sources=False),
        eval_ds(),
        _DEFAULT_PT,
    )
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run python -c "
import model.test_train_cli as t
t.test_litcolorgan_builds()
print('ok')
"
```

- [ ] **Step 5: Ruff and commit**

```bash
uv run ruff check model/train.py model/test_train_cli.py
git add model/train.py model/test_train_cli.py
git commit -m "$(cat <<'EOF'
Migrate LitColorGAN to one Critic trained on four groups

Drops the ephemeral unconditional ColorCritic (its generator-loss
weight was already 0); adds a random-color negative alongside the
existing real/generator-fake/shuffled groups.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Delete dead classes/constants/metrics, final verification

**Files:**
- Modify: `model/model.py` (delete `ColorEmbedding`, `ColorCritic`, `CondColorCritic`; drop `EMBED_SIZE_COLOR` import)
- Modify: `model/metric.py` (delete `Metric.AUROC`; delete `GanMetric.GEN_LOSS_COLOR`, `COLOR_LOSS`, `COLOR_AUROC`)
- Modify: `files.py` (delete `PtFile.COND_CRITIC`, `PtFile.COND_COLOR_EMBED`, `COND_CRITIC_PT`, `COND_COLOR_EMBED_PT`)
- Modify: `model/config.py` (delete `EMBED_SIZE_COLOR`, `LR_GAN_CRITIC`, `GAN_LOSS_COLOR`; rewrite `gan_str`)
- Modify: `tools/print_model_params.py` (dev utility script, not covered by the design spec's file list — discovered via a repo-wide grep at execution time: it imports `CondColorCritic` directly and would break at runtime once that class is deleted, since ruff's import-order check doesn't verify the imported name still exists)

**Interfaces:**
- Consumes: nothing new — this task only removes symbols nothing references anymore after Tasks 6-8.
- Produces: nothing new.

- [ ] **Step 1: Confirm the old symbols are actually dead**

```bash
grep -rn "ColorEmbedding\|ColorCritic\b\|CondColorCritic" --include=*.py . | grep -v "class Critic\b"
```

Expected: two hits outside `model/model.py`'s own definitions —
`tools/print_model_params.py:1` (import line) and
`tools/print_model_params.py:15` (`("ColorCritic", CondColorCritic)` table
row). Everything else should be only the `class ColorEmbedding` /
`class ColorCritic` / `class CondColorCritic` definition lines in
`model/model.py` (no other call sites left — Tasks 6-8 removed them
all). If anything else shows up, stop and re-check the earlier task's
edit before proceeding.

- [ ] **Step 1b: Fix `tools/print_model_params.py`**

Change:

```python
from model.model import ColorGen, CondColorCritic, EmojiHead, StyleHead, TextEncoder
from torch import nn
import typer
import sys
from pathlib import Path
```

to:

```python
import sys
from pathlib import Path

import typer
from torch import nn

from model.model import ColorGen, Critic, EmojiHead, StyleHead, TextEncoder
```

(fixes the pre-existing `ruff` import-order warning on this file while
touching it) and change:

```python
    ("ColorCritic", CondColorCritic),
```

to:

```python
    ("Critic", Critic),
```

- [ ] **Step 2: Implement — `model/model.py`**

Change the config import (drop `EMBED_SIZE_COLOR`):

```python
from model.config import (
    CRITIC_HIDDEN_SIZE,
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_COLOR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_HIDDEN_SIZE,
    RELU_SLOPE,
    Z_SIZE,
)
```

to:

```python
from model.config import (
    CRITIC_HIDDEN_SIZE,
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_HIDDEN_SIZE,
    RELU_SLOPE,
    Z_SIZE,
)
```

Delete `ColorEmbedding`, `ColorCritic`, `CondColorCritic` entirely (everything from `class ColorEmbedding(nn.Module):` through the end of `CondColorCritic.forward`, i.e. the three classes sitting between `ColorGen` and `Critic`), leaving `cblk`, `ColorGen`, and `Critic` as the only GAN-section classes.

- [ ] **Step 3: Implement — `model/metric.py`**

Change:

```python
class Metric(StrEnum):
    ACC_1 = "acc@1"
    RATE = "rate"
    MAE = "mae"
    MRR = "mrr"
    R2 = "r2"
    AUROC = "auroc"
    AUROC_SHUF = "auroc_shuf"
    AUROC_RAND = "auroc_rand"
    LOSS = "loss"
```

to:

```python
class Metric(StrEnum):
    ACC_1 = "acc@1"
    RATE = "rate"
    MAE = "mae"
    MRR = "mrr"
    R2 = "r2"
    AUROC_SHUF = "auroc_shuf"
    AUROC_RAND = "auroc_rand"
    LOSS = "loss"
```

Change:

```python
class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS_COND = "gan/gen_loss_cond"
    GEN_LOSS_COLOR = "gan/gen_loss_color"
    COND_LOSS = "gan/cond_loss"
    COND_AUROC_GEN = "gan/cond_auroc_gen"
    COND_AUROC_SHUF = "gan/cond_auroc_shuf"
    COND_AUROC_RAND = "gan/cond_auroc_rand"
    COND_MEAN_SCORE_REAL = "gan/cond_mean_score_real"
    COND_MEAN_SCORE_FAKE = "gan/cond_mean_score_fake"
    COND_MEAN_SCORE_WRONG = "gan/cond_mean_score_wrong"
    COND_MEAN_SCORE_RANDOM = "gan/cond_mean_score_random"
    COLOR_LOSS = "gan/color_loss"
    COLOR_AUROC = "gan/color_auroc"
```

to:

```python
class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS_COND = "gan/gen_loss_cond"
    COND_LOSS = "gan/cond_loss"
    COND_AUROC_GEN = "gan/cond_auroc_gen"
    COND_AUROC_SHUF = "gan/cond_auroc_shuf"
    COND_AUROC_RAND = "gan/cond_auroc_rand"
    COND_MEAN_SCORE_REAL = "gan/cond_mean_score_real"
    COND_MEAN_SCORE_FAKE = "gan/cond_mean_score_fake"
    COND_MEAN_SCORE_WRONG = "gan/cond_mean_score_wrong"
    COND_MEAN_SCORE_RANDOM = "gan/cond_mean_score_random"
```

- [ ] **Step 4: Implement — `files.py`**

Change:

```python
class PtFile(StrEnum):
    ENC = "enc.pt"
    STYLE = "style.pt"
    EMOJI = "emoji.pt"
    GEN = "gen.pt"
    CRITIC = "critic.pt"
    COND_CRITIC = "cond_critic.pt"
    COND_COLOR_EMBED = "cond_color_embed.pt"

    def in_dir(self, pt_dir: Path) -> Path:
        return pt_dir / self.value


ENC_PT = PtFile.ENC.in_dir(PT_DIR)
STYLE_PT = PtFile.STYLE.in_dir(PT_DIR)
EMOJI_PT = PtFile.EMOJI.in_dir(PT_DIR)
GEN_PT = PtFile.GEN.in_dir(PT_DIR)
CRITIC_PT = PtFile.CRITIC.in_dir(PT_DIR)
COND_CRITIC_PT = PtFile.COND_CRITIC.in_dir(PT_DIR)
COND_COLOR_EMBED_PT = PtFile.COND_COLOR_EMBED.in_dir(PT_DIR)
```

to:

```python
class PtFile(StrEnum):
    ENC = "enc.pt"
    STYLE = "style.pt"
    EMOJI = "emoji.pt"
    GEN = "gen.pt"
    CRITIC = "critic.pt"

    def in_dir(self, pt_dir: Path) -> Path:
        return pt_dir / self.value


ENC_PT = PtFile.ENC.in_dir(PT_DIR)
STYLE_PT = PtFile.STYLE.in_dir(PT_DIR)
EMOJI_PT = PtFile.EMOJI.in_dir(PT_DIR)
GEN_PT = PtFile.GEN.in_dir(PT_DIR)
CRITIC_PT = PtFile.CRITIC.in_dir(PT_DIR)
```

- [ ] **Step 5: Implement — `model/config.py`**

Delete the `EMBED_SIZE_COLOR = 32` line from the `# EMBEDDING` block.

Delete the `LR_GAN_CRITIC = 0.005` line from the `# LR` block.

Change:

```python
GAN_LOSS_ENERGY = 0
GAN_LOSS_COLOR = 0
GAN_LOSS_COND = 1
COND_CRITIC_MISMATCH_WEIGHT = 0.9
COND_CRITIC_RANDOM_WEIGHT = 0.1
```

to:

```python
GAN_LOSS_ENERGY = 0
GAN_LOSS_COND = 1
COND_CRITIC_MISMATCH_WEIGHT = 0.9
COND_CRITIC_RANDOM_WEIGHT = 0.1
```

Change `gan_str`:

```python
gan_str = " ".join([
    str(p)
    for p in (
        Z_SIZE,
        GEN_HIDDEN_SIZE,
        EMBED_SIZE_COLOR,
        LR_GAN_GEN,
        LR_GAN_CRITIC,
        LR_GAN_COND_CRITIC,
        GAN_LOSS_ENERGY,
        GAN_LOSS_COND,
        GAN_LOSS_COLOR)])
```

to:

```python
gan_str = " ".join([
    str(p)
    for p in (
        Z_SIZE,
        GEN_HIDDEN_SIZE,
        CRITIC_HIDDEN_SIZE,
        LR_GAN_GEN,
        LR_GAN_COND_CRITIC,
        GAN_LOSS_ENERGY,
        GAN_LOSS_COND,
        COND_CRITIC_RANDOM_WEIGHT)])
```

- [ ] **Step 6: Run the full smoke suite and ruff**

```bash
uv run python model/test_train_cli.py
uv run ruff check .
uv run ruff format --check .
```

Expected: `test_train_cli.py` prints `ok`; ruff reports no errors (in particular no unused-import warnings for `EMBED_SIZE_COLOR`/`LR_GAN_CRITIC`/`GAN_LOSS_COLOR`/`ColorEmbedding`/`ColorCritic`/`CondColorCritic`/`COND_CRITIC`/`COND_COLOR_EMBED`/`Metric.AUROC`/`GanMetric.COLOR_LOSS`/`GanMetric.COLOR_AUROC`/`GanMetric.GEN_LOSS_COLOR`, since Step 1 already confirmed nothing references them).

- [ ] **Step 7: Full repo grep for stragglers**

```bash
grep -rn "cond_critic\.pt\|cond_color_embed\.pt\|COND_CRITIC\|COND_COLOR_EMBED\|color_auroc\b\|GEN_LOSS_COLOR\|EMBED_SIZE_COLOR\|LR_GAN_CRITIC\b" --include=*.py --include=*.md .
```

Expected: no hits in any `.py` file. `.md` hits are expected and fine —
this plan file, the design spec, and pre-existing historical docs
(`docs/superpowers/plans/2026-09-07-*`, `docs/model.md`, `docs/logs.md`,
`docs/overview.md`, `data/data.md`, `plans/26-09-18-*`) reference the
*old* names/architecture; none of those are touched by this plan
(`docs/model.md`/`docs/logs.md` were already stale relative to the
current code before this change — e.g. `docs/model.md` references
metrics like `F1/val` that don't exist in `model/metric.py` today — so
bringing them in sync is a separate, larger cleanup, out of scope here).

- [ ] **Step 8: Commit**

```bash
git add model/model.py model/metric.py files.py model/config.py tools/print_model_params.py
git commit -m "$(cat <<'EOF'
Delete dead ColorEmbedding/ColorCritic/CondColorCritic and their
constants/metrics/checkpoints, now fully replaced by Critic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## After this plan (manual, not a task)

Per repo convention (no CI test gate — real verification is a full
training run, not quick checks), once all 9 tasks are committed, run
the actual pipeline by hand and eyeball TensorBoard / the report:

```bash
train enc --local     # watch enc/color_auroc_shuf, enc/color_auroc_rand
train cond --local    # watch cond/color_auroc_shuf, cond/color_auroc_rand
train gan --local      # watch gan/cond_auroc_gen, gan/cond_auroc_shuf, gan/cond_auroc_rand
train --local          # full pipeline; tools/report.py auto-runs after
```

Confirm `pt/critic.pt` is written (and `pt/cond_critic.pt` /
`pt/cond_color_embed.pt` are no longer produced — old copies from
before this change are simply stale, not read by anything anymore).
