"""Train the emojic CNN feeling classifier (PyTorch Lightning).

A ``LitEmojic`` LightningModule wraps ``model.Model`` and trains the feeling
head only. Validation runs every ``EVAL_EPOCHS`` epochs; the ``ExportBest``
callback keeps the best checkpoint (by eval feeling loss) and, every time the
best improves, rewrites both ``model.pt`` and the static web app's artifacts in
``docs/`` (``model.onnx`` + ``meta.json`` + ``config.json``), so the page can be
watched live during a run.

The data pipeline (``data.py``) still carries the emoji label per row, so an
emoji head can be added back later; the model and this script currently train
the feeling head only.
"""

import json
import warnings
from pathlib import Path

import lightning as pl
import torch
import torch.utils.data
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
            output_names=["feeling_logits"],
            opset_version=ONNX_OPSET,
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch"},
                "feeling_logits": {0: "batch"},
            },
        )


def export_web(model: nn.Module) -> None:
    """Refresh docs/model.onnx + docs/meta.json + docs/config.json for the app.

    meta.json carries everything docs/app.js must not hardcode from the Python
    side: the char vocab, MAX_TEXT_LEN, and the label sets. The emoji list is
    still emitted so the front-end scaffolding can stay in place, even though
    the current model only has a feeling head. (The feeling color palette is
    not here -- it lives in docs/palette.json, read directly by app.js.)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _, target_feeling = batch
        logits = self.model(x)
        loss = self.feeling_ce(logits, target_feeling)
        acc = (logits.argmax(dim=-1) == target_feeling).float().mean()
        self.log(
            "train/f_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/f_acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, _, target_feeling = batch
        logits = self.model(x)
        loss = self.feeling_ce(logits, target_feeling)
        acc = (logits.argmax(dim=-1) == target_feeling).float().mean()
        # batch_size weights the epoch mean, matching the old size-weighted eval.
        self.log(
            "eval/f_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log(
            "eval/f_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        return optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


class ExportBest(pl.Callback):
    """Save model.pt + refresh docs/ whenever eval f_loss improves."""

    def __init__(self) -> None:
        self.best_loss = float("inf")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LitEmojic) -> None:
        metric = trainer.callback_metrics.get("eval/f_loss")
        if metric is None:
            return
        loss = float(metric)
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(pl_module.model.state_dict(), MODEL_PT)
            export_web(pl_module.model)


def train() -> None:
    pl.seed_everything(0, workers=True)

    train_ds, eval_ds = data_sets()
    train_loader = train_data_loader(train_ds)
    eval_loader = eval_data_loader(eval_ds)

    lit = LitEmojic()
    export_best = ExportBest()

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
        callbacks=[export_best],
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    trainer.fit(
        lit,
        train_loader,
        # train_loader,
        eval_loader
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
    train()
