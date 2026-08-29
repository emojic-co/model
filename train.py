"""Train the emojic CNN classifier (PyTorch Lightning).

A ``LitEmojic`` LightningModule wraps ``model.Model`` and trains both heads
(feeling + emoji, summed cross-entropy). Validation runs every ``EVAL_EPOCHS``
epochs; the ``ExportBest`` callback keeps the best checkpoint (by eval feeling
loss) and, every time the best improves, rewrites both ``model.pt`` and the
static web app's artifacts in ``docs/`` (``model.onnx`` + ``meta.json`` +
``config.json``), so the page can be watched live during a run.
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
    pieces the triplet loss needs. The browser only wants a class score per
    emoji, so this wrapper scores ``q`` against every emoji embedding as the
    negative squared L2 distance: ``argmax`` then picks the nearest embedding,
    matching the metric the triplet loss trains and ``validation_step``'s
    ``torch.cdist`` accuracy. Keeps the ONNX contract at
    ``(feeling_logits, emoji_logits)`` so ``app.js`` stays a plain argmax path.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feeling_logits, q, emoji_embed = self.model(x)
        # ||q - e||^2 = ||q||^2 - 2 q.e + ||e||^2. Expanded rather than
        # torch.cdist so it traces to matmul/reduce ops that every ONNX opset
        # supports. The ||q||^2 term is constant per row (doesn't move argmax)
        # but is kept so the values are true negative distances for the panel.
        d2 = (
            q.pow(2).sum(-1, keepdim=True)
            - 2.0 * q @ emoji_embed.t()
            + emoji_embed.pow(2).sum(-1)
        )
        return feeling_logits, -d2


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
        # state toward its true emoji vector and push it off one sampled wrong
        # emoji vector, by TRIPLET_MARGIN in L2.
        self.emoji_triplet = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, target_emoji, target_feeling = batch

        logits_feeling, q, emoji_embed = self.model(x)
        loss_feeling = self.feeling_ce(logits_feeling, target_feeling)

        # One negative emoji per row: shift the true index by a random 1..N-1
        # offset (mod N) -- uniform over the wrong classes, never the target.
        n = emoji_embed.size(0)
        offset = torch.randint(1, n, target_emoji.shape, device=self.device)
        neg_emoji = (target_emoji + offset) % n

        loss_emoji = self.emoji_triplet(
            q,                          # anchor: projected hidden state
            emoji_embed[target_emoji],  # positive: true emoji vector
            emoji_embed[neg_emoji],     # negative: sampled wrong emoji vector
        )

        acc_feeling = (
            logits_feeling.argmax(dim=-1) == target_feeling).float().mean()

        # Nearest emoji embedding under the same L2 metric the triplet loss trains.
        pred_emoji = torch.cdist(q, emoji_embed).argmin(dim=-1)
        acc_emoji = (pred_emoji == target_emoji).float().mean()

        def log(k, v):
            self.log(
                k, v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.size(0))

        log("train/f_acc", acc_feeling)
        log("train/e_acc", acc_emoji)
        log("train/f_loss", loss_feeling)
        log("train/e_loss", loss_emoji)
        return loss_feeling + loss_emoji

    def validation_step(self, batch, batch_idx) -> None:
        x, target_emoji, target_feeling = batch
        logits_feeling, q, emoji_embed = self.model(x)

        acc_feeling = (
            logits_feeling.argmax(dim=-1) == target_feeling).float().mean()

        # Nearest emoji embedding under the same L2 metric the triplet loss trains.
        pred_emoji = torch.cdist(q, emoji_embed).argmin(dim=-1)
        acc_emoji = (pred_emoji == target_emoji).float().mean()

        # batch_size weights the epoch mean, matching the old size-weighted eval.

        def log(k, v):
            self.log(
                k, v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.size(0))

        # log("eval/f_loss", loss_feeling)
        # log("eval/e_loss", loss_emoji)
        log("eval/f_acc", acc_feeling)
        log("eval/e_acc", acc_emoji)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY)

        # return optim.Adam(
        #     self.parameters(),
        #     lr=LR,
        #     weight_decay=WEIGHT_DECAY)


class ExportBest(pl.Callback):
    """Save model.pt + refresh docs/ whenever eval f_acc improves."""

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
        metric = trainer.callback_metrics.get("eval/f_acc")
        if metric is None:
            return
        acc = float(metric)
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(pl_module.model.state_dict(), MODEL_PT)
            export_web(pl_module.model)


def train(resume: bool = False) -> None:
    pl.seed_everything(0, workers=True)

    train_ds, eval_ds = data_sets()
    train_loader = train_data_loader(train_ds)
    eval_loader = eval_data_loader(eval_ds)

    lit = LitEmojic()
    export_best = ExportBest()
    # Rotating full-state checkpoint: one runs/<name>.ckpt overwritten every
    # EVAL_EPOCHS (at validation end), plus the runs/last.ckpt copy --resume
    # reads. model.pt (via ExportBest) stays the "best" artifact, so no top-k.
    checkpoint = ModelCheckpoint(
        dirpath=str(LAST_CKPT.parent),
        filename="ckpt",
        save_last=True,
        save_top_k=1,
        every_n_epochs=EVAL_EPOCHS,
    )

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(f"Params: {sum(p.numel() for p in lit.model.parameters()):,}\n")

    logger = TensorBoardLogger(
        "runs",
        name=CONFIG_NAME,
        version="")

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
        f"\nBest eval f_acc: {export_best.best_acc:.4f}  ->  "
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
