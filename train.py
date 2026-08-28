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


def export_onnx(model: nn.Module, dst: Path) -> None:
    """Trace ``model`` to an ONNX file with a dynamic batch axis."""
    model.eval()
    dummy = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
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
    side: the char vocab, MAX_TEXT_LEN, and the label sets for both heads. (The
    feeling color palette is not here -- it lives in docs/palette.json, read
    directly by app.js.)

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
        self.emoji_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _, target_feeling, target_emoji = batch

        logits_feeling, logits_emoji = self.model(x)
        loss_feeling = self.feeling_ce(logits_feeling, target_feeling)
        loss_emoji = self.emoji_ce(logits_emoji, target_emoji)

        def log(k, v):
            self.log(
                k, v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.size(0))

        log("train/f_loss", loss_feeling)
        log("train/e_loss", loss_emoji)

        return loss_feeling + loss_emoji

    def validation_step(self, batch, batch_idx) -> None:
        x, _, target_feeling, target_emoji = batch
        (logits_feeling, logits_emoji) = self.model(x)
        # loss_feeling = self.feeling_ce(logits_feeling, target_feeling)
        # loss_emoji = self.emoji_ce(logits_emoji, target_emoji)

        acc_feeling = (
            logits_feeling.argmax(dim=-1) == target_feeling).float().mean()

        acc_emoji = (
            logits_emoji.argmax(dim=-1) == target_emoji).float().mean()

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
        return optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        return optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


class ExportBest(pl.Callback):
    """Save model.pt + refresh docs/ whenever eval f_loss improves."""

    def __init__(self) -> None:
        self.best_loss = float("inf")

    def state_dict(self) -> dict:
        # Persisted into the checkpoint so best_loss survives --resume; without
        # it the first post-resume validation re-saves model.pt + re-exports on
        # a non-improvement.
        return {"best_loss": self.best_loss}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_loss = state_dict["best_loss"]

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LitEmojic) -> None:
        metric = trainer.callback_metrics.get("eval/f_loss")
        if metric is None:
            return
        loss = float(metric)
        if loss < self.best_loss:
            self.best_loss = loss
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

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        # config guarantees EPOCHS % EVAL_EPOCHS == 0, so the last epoch validates.
        check_val_every_n_epoch=EVAL_EPOCHS,
        gradient_clip_val=GRAD_CLIP,
        accelerator="cpu",
        devices='auto',
        logger=TensorBoardLogger("runs", name=CONFIG_NAME, version=""),
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
        f"\nBest eval loss: {export_best.best_loss:.4f}  ->  "
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
