"""Train the emojic CNN classifier (PyTorch Lightning).

A ``LitEmojic`` LightningModule wraps ``model.Model``. Only the feeling head is
optimized right now (``training_step`` returns ``loss_feeling`` alone -- the
emoji triplet term is built but not added); feeling is the current priority.
Validation runs every ``EVAL_EPOCHS`` epochs against the fixed gold holdout
``eval.jsonl`` (see gen_eval.ts / data.py). Training and validation each log
exactly four scalars and nothing else -- ``loss/f/{train,val}`` (feeling
cross-entropy), ``loss/e/{train,val}`` (emoji triplet), ``acc/f/{train,val}``
(feeling top-1) and ``acc5/e/{train,val}`` (emoji top-5 retrieval). The
``ExportBest`` callback keeps the best checkpoint (by ``acc/f/val``) and, every
time the best improves, rewrites both ``model.pt`` and the static web app's
artifacts in ``docs/`` (``model.onnx`` + ``meta.json`` + ``config.json``), so
the page can be watched live during a run.
"""

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import lightning as pl
import torch
import torch.utils.data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim

from config import (
    CONFIG_NAME,
    EMOJI_NEGATIVES,
    EPOCHS,
    EVAL_EPOCHS,
    GRAD_CLIP,
    LR,
    MAX_TEXT_LEN,
    TRIPLET_MARGIN,
    WEIGHT_DECAY,
)
from data import (
    CHARS,
    EMOJIS,
    FEELING,
    PAD_IDX,
    data_sets,
    eval_data_loader,
    train_data_loader,
)
from model import Model

MODEL_PT = Path("model.pt")
# Full training state (optimizer / epoch / global step / RNG / callback state),
# written under the gitignored runs/ dir at a fixed path so `--resume` finds it
# even though CONFIG_NAME (and thus the TensorBoard log dir) is timestamped.
LAST_CKPT = Path("runs") / "last.ckpt"
DOCS = Path("docs")
ONNX_OPSET = 18


class ExportWrapper(nn.Module):
    """Collapse the emoji embedding head into a single ``emoji_logits`` tensor.

    ``Model.forward`` returns ``(feeling_logits, q, emoji_embed)`` -- the raw
    pieces the triplet loss needs, with ``q`` and every ``emoji_embed`` row
    L2-normalized. The browser only wants a class score per emoji, so this
    wrapper scores ``q`` against every emoji embedding by cosine similarity
    (a plain matmul of unit vectors). Since both sides are unit-norm this is
    a monotonic function of the L2 distance -- ``-||q - e||^2 = 2(cos - 1)`` --
    so ``argmax`` picks the same nearest embedding the triplet loss trains and
    ``training_step`` / ``validation_step`` score with. Keeps the ONNX contract
    at ``(feeling_logits, emoji_logits)`` so ``app.js`` stays a plain argmax
    path; a single matmul traces cleanly on every ONNX opset.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feeling_logits, q, emoji_embed = self.model(x)
        return feeling_logits, q @ emoji_embed.t()


def export_onnx(model: nn.Module, dst: Path) -> None:
    """Trace ``model`` to an ONNX file with a dynamic batch axis."""
    wrapper = ExportWrapper(model).eval()
    dummy = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(dst),
            input_names=["input"],
            output_names=["feeling_logits", "emoji_logits"],
            opset_version=ONNX_OPSET,
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch"},
                "feeling_logits": {0: "batch"},
                "emoji_logits": {0: "batch"},
            },
        )


def export_web(model: nn.Module) -> None:
    """Refresh docs/model.onnx + docs/meta.json + docs/config.json for the app.

    meta.json carries everything docs/app.js must not hardcode from the Python
    side: the char vocab, MAX_TEXT_LEN, the label sets for both heads, and the
    export date (footer). (The feeling color palette is not here -- it lives in
    docs/palette.json, read directly by app.js.)

    config.json holds the plain app-tuning knobs (currently just max_text_len,
    used to cap the input field) kept apart from the model metadata.
    """
    DOCS.mkdir(exist_ok=True)
    export_onnx(model, DOCS / "model.onnx")
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": FEELING,
        # ISO 8601 UTC instant of this export (minute precision, with the
        # +00:00 offset kept) so docs/app.js can parse it and render it in the
        # viewer's own local time zone for the footer.
        "exported_at": datetime.now(UTC).isoformat(timespec="minutes"),
    }
    (DOCS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS / "config.json").write_text(
        json.dumps({"max_text_len": MAX_TEXT_LEN}, indent=2), encoding="utf-8"
    )


class LitEmojic(pl.LightningModule):
    """Feeling-head training wrapper around ``model.Model``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Model()
        self.feeling_ce = nn.CrossEntropyLoss()
        # Emoji head is trained by metric learning: pull the projected hidden
        # state toward its true emoji vector and push it off EMOJI_NEGATIVES
        # sampled wrong emoji vectors, by TRIPLET_MARGIN in L2. reduction="mean"
        # averages over all anchor x negative triplets in the batch.
        self.emoji_triplet = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x)

    def _feeling_terms(self, logits_feeling, target_feeling):
        """Feeling cross-entropy loss and top-1 accuracy for a batch."""
        loss = self.feeling_ce(logits_feeling, target_feeling)
        acc = (logits_feeling.argmax(dim=-1) == target_feeling).float().mean()
        return loss, acc

    def _emoji_terms(self, q, emoji_embd, target_emoji):
        """Emoji triplet loss and top-5 retrieval accuracy for a batch.

        EMOJI_NEGATIVES wrong classes per row: shift the true index by random
        1..N-1 offsets (mod N) -- uniform over the wrong classes, never the
        target. anchor/positive are tiled to (B*K, D) so one
        nn.TripletMarginLoss call scores every (row, negative) pair and means
        over all B*K triplets. q and emoji_embd are unit-norm, so the largest
        cosine similarities are the smallest L2 distances the triplet trains --
        the exact score ExportWrapper ships -- and top-1 retrieval over the full
        emoji set is hopeless, so track the top-5 hit rate.
        """
        n = emoji_embd.size(0)
        offset = torch.randint(
            1, n,
            (target_emoji.size(0), EMOJI_NEGATIVES),
            device=self.device)
        neg_emoji_idx = (target_emoji.unsqueeze(1) + offset) % n  # (B, K)

        pos = emoji_embd[target_emoji].repeat_interleave(EMOJI_NEGATIVES, dim=0)
        neg = emoji_embd[neg_emoji_idx.reshape(-1)]  # (B*K, D)
        loss = self.emoji_triplet(
            q.repeat_interleave(EMOJI_NEGATIVES, dim=0),  # anchor
            pos,  # positive
            neg,  # negatives
        )

        top5 = (q @ emoji_embd.t()).topk(5, dim=-1).indices
        acc5 = (top5 == target_emoji.unsqueeze(1)).any(dim=-1).float().mean()
        return loss, acc5

    def _log_split(self, split, batch_size, loss_f, loss_e, acc_f, acc5_e):
        """Log exactly the four scalars for one split (train/val), nothing else."""
        kw = dict(
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"loss/f/{split}", loss_f, **kw)
        self.log(f"loss/e/{split}", loss_e, **kw)
        self.log(f"acc/f/{split}", acc_f, **kw)
        self.log(f"acc5/e/{split}", acc5_e, **kw)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, target_emoji, target_feeling = batch
        logits_feeling, q, emoji_embd = self.model(x)

        loss_feeling, acc_feeling = self._feeling_terms(
            logits_feeling, target_feeling)
        loss_emoji, acc_emoji5 = self._emoji_terms(q, emoji_embd, target_emoji)

        self._log_split(
            "train", x.size(0),
            loss_feeling, loss_emoji, acc_feeling, acc_emoji5)

        # TEMP, Debugging the feeling prediction path.
        return loss_feeling
        return loss_feeling + loss_emoji

    def validation_step(self, batch, batch_idx) -> None:
        x, target_emoji, target_feeling = batch
        logits_feeling, q, emoji_embd = self.model(x)

        loss_feeling, acc_feeling = self._feeling_terms(
            logits_feeling, target_feeling)
        loss_emoji, acc_emoji5 = self._emoji_terms(q, emoji_embd, target_emoji)

        self._log_split(
            "val", x.size(0),
            loss_feeling, loss_emoji, acc_feeling, acc_emoji5)

    def configure_optimizers(self):
        # return optim.SGD(
        #     self.parameters(),
        #     lr=LR,
        #     weight_decay=WEIGHT_DECAY)

        return optim.Adam(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY)


class ExportBest(pl.Callback):
    """Save model.pt + refresh docs/ whenever acc/f/val improves (max)."""

    def __init__(self) -> None:
        self.best_acc = 0.0

    def state_dict(self) -> dict:
        # Persisted into the checkpoint so best_acc survives --resume; without
        # it the first post-resume validation re-saves model.pt + re-exports on
        # a non-improvement.
        return {"best_acc": self.best_acc}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_acc = state_dict["best_acc"]

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LitEmojic) -> None:
        metric = trainer.callback_metrics.get("acc/f/val")
        if metric is None:
            return
        acc = float(metric)
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(pl_module.model.state_dict(), MODEL_PT)
            export_web(pl_module.model)


def param_table(model: nn.Module) -> str:
    """Render a per-module / per-parameter breakdown of ``model``'s params.

    One indented row per leaf parameter (with its shape), grouped under each
    top-level child module with that child's subtotal and share of the total.
    """
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    name_w = max((len(n) for n, _ in named), default=18) + 4
    head = f"{'module / parameter':<{name_w}}{'shape':>16}{'params':>12}{'%':>8}"
    rule = "-" * len(head)

    out = [head, rule]
    for child_name, child in model.named_children():
        sub = sum(p.numel() for p in child.parameters())
        pct = 100 * sub / total if total else 0.0
        out.append(f"{child_name:<{name_w}}{'':>16}{sub:>12,}{pct:>7.1f}%")
        for pname, p in child.named_parameters():
            shape = "x".join(map(str, tuple(p.shape)))
            out.append(f"  {pname:<{name_w - 2}}{shape:>16}{p.numel():>12,}")
    out.append(rule)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    out.append(f"{'total':<{name_w}}{'':>16}{total:>12,}{100.0:>7.1f}%")
    if trainable != total:
        out.append(f"{'trainable':<{name_w}}{'':>16}{trainable:>12,}")
    return "\n".join(out)


def train(resume: bool = False) -> None:
    pl.seed_everything(0, workers=True)

    train_ds, eval_ds = data_sets()
    train_loader = train_data_loader(train_ds)
    eval_loader = eval_data_loader(eval_ds)

    lit = LitEmojic()
    export_best = ExportBest()
    # Full-state checkpoint for --resume: save_top_k=1 monitored on acc/f/val
    # keeps the best-by-feeling-accuracy runs/ckpt.ckpt, plus the runs/last.ckpt
    # copy --resume reads. model.pt / model.onnx (via ExportBest) are the
    # shipped artifacts, gated on the same acc/f/val.
    checkpoint = ModelCheckpoint(
        dirpath=str(LAST_CKPT.parent),
        filename="ckpt",
        save_last=True,
        save_top_k=1,
        monitor="acc/f/val",
        mode="max",
        every_n_epochs=EVAL_EPOCHS,
    )

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(param_table(lit.model), "\n")

    logger = TensorBoardLogger(
        "runs",
        name=CONFIG_NAME,
        version="",
        # Don't emit the placeholder hp_metric scalar -- only the eight
        # loss/acc scalars from _log_split should appear in TensorBoard.
        default_hp_metric=False)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        # config guarantees EPOCHS % EVAL_EPOCHS == 0, so the last epoch validates.
        check_val_every_n_epoch=EVAL_EPOCHS,
        gradient_clip_val=GRAD_CLIP,
        accelerator="cpu",
        devices='auto',
        logger=logger,
        callbacks=[export_best, checkpoint],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    ckpt_path = str(LAST_CKPT) if resume and LAST_CKPT.exists() else None
    if resume and ckpt_path is None:
        print(f"--resume: no checkpoint at {LAST_CKPT}, starting fresh")

    trainer.fit(
        lit,
        train_loader,
        # train_loader,
        eval_loader,
        ckpt_path=ckpt_path,
    )

    print(
        f"\nBest acc/f/val: {export_best.best_acc:.4f}  ->  "
        f"{MODEL_PT} and docs/ refreshed"
    )

    # Behavioral test suite + Markdown report (report/<MM-DD-HH:MM>.md).
    # Runs against the saved best checkpoint (model.pt), i.e. what ships.
    from test_model import run as run_tests

    run_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the emojic feeling classifier (PyTorch Lightning)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"resume training from {LAST_CKPT} (optimizer / epoch / RNG state)",
    )
    train(resume=parser.parse_args().resume)
