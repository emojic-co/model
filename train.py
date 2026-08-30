import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn import functional as F

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_EPOCHS,
    GRAD_CLIP,
    INFONCE_TEMP,
    LR,
    MAX_TEXT_LEN,
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
LAST_CKPT = Path("runs") / "last.ckpt"
WEB_PUBLIC = Path("web/public")
ONNX_OPSET = 18


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feeling_logits, q, emoji_embed = self.model(x)
        return feeling_logits, q @ emoji_embed.t()


def export_onnx(model: nn.Module, dst: Path) -> None:
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
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    export_onnx(model, WEB_PUBLIC / "model.onnx")
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": FEELING,
        "exported_at": datetime.now(UTC).isoformat(timespec="minutes"),
    }
    (WEB_PUBLIC / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (WEB_PUBLIC / "config.json").write_text(
        json.dumps({"max_text_len": MAX_TEXT_LEN}, indent=2), encoding="utf-8"
    )


class LitEmojic(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = Model()
        self.feeling_ce = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x)

    def _feeling_terms(self, logits_feeling, target_feeling):
        loss = self.feeling_ce(logits_feeling, target_feeling)
        acc = (logits_feeling.argmax(dim=-1) == target_feeling).float().mean()
        return loss, acc

    def _emoji_terms(self, q, emoji_embd, target_emoji):
        logits = q @ emoji_embd.t()
        loss = F.cross_entropy(logits / INFONCE_TEMP, target_emoji)

        top10 = logits.topk(10, dim=-1).indices
        hit10 = top10 == target_emoji.unsqueeze(1)
        acc5 = hit10[:, :5].any(dim=-1).float().mean()
        acc10 = hit10.any(dim=-1).float().mean()
        return loss, acc5, acc10

    def _log_split(
            self,
            split,
            batch_size,
            loss_f,
            loss_e,
            acc_f,
            acc5_e,
            acc10_e
    ):
        kw = dict(
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"loss/f/{split}", loss_f, **kw)  # type: ignore
        self.log(f"loss/e/{split}", loss_e, **kw)  # type: ignore
        self.log(f"acc/f/{split}", acc_f, **kw)  # type: ignore
        self.log(f"acc5/e/{split}", acc5_e, **kw)  # type: ignore
        self.log(f"acc10/e/{split}", acc10_e, **kw)  # type: ignore

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, target_emoji, target_feeling = batch
        logits_feeling, q, emoji_embd = self.model(x)

        loss_feeling, acc_feeling = self._feeling_terms(
            logits_feeling, target_feeling)
        loss_emoji, acc_emoji5, acc_emoji10 = self._emoji_terms(
            q, emoji_embd, target_emoji)

        self._log_split(
            "train", x.size(0),
            loss_feeling,
            loss_emoji,
            acc_feeling,
            acc_emoji5,
            acc_emoji10
        )

        return loss_feeling + loss_emoji

    def validation_step(self, batch, batch_idx) -> None:
        x, target_emoji, target_feeling = batch
        logits_feeling, q, emoji_embd = self.model(x)

        loss_feeling, acc_feeling = self._feeling_terms(
            logits_feeling, target_feeling)
        loss_emoji, acc_emoji5, acc_emoji10 = self._emoji_terms(
            q, emoji_embd, target_emoji)

        self._log_split(
            "val", x.size(0),
            loss_feeling,
            loss_emoji,
            acc_feeling,
            acc_emoji5,
            acc_emoji10
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LR)


class ExportBest(pl.Callback):

    def __init__(self) -> None:
        self.best_acc = 0.0

    def state_dict(self) -> dict:
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


class SaveLast(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: LitEmojic) -> None:
        trainer.save_checkpoint(LAST_CKPT, weights_only=False)


def param_table(model: nn.Module) -> str:
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
    early_stop = EarlyStopping(
        monitor="acc/f/val",
        mode="max",
        patience=EARLY_STOP_PATIENCE,
        check_on_train_epoch_end=False,
    )
    LAST_CKPT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(param_table(lit.model), "\n")

    logger = TensorBoardLogger(
        "runs",
        name=CONFIG_NAME,
        version="",
        default_hp_metric=False)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        check_val_every_n_epoch=EVAL_EPOCHS,
        gradient_clip_val=GRAD_CLIP,
        accelerator="cpu",
        devices='auto',
        logger=logger,
        enable_checkpointing=False,
        callbacks=[export_best, early_stop, SaveLast()],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    ckpt_path = str(LAST_CKPT) if resume and LAST_CKPT.exists() else None
    if resume and ckpt_path is None:
        print(f"--resume: no checkpoint at {LAST_CKPT}, starting fresh")

    trainer.fit(
        lit,
        train_loader,
        eval_loader,
        ckpt_path=ckpt_path,
    )

    print(
        f"\nBest acc/f/val: {export_best.best_acc:.4f}  ->  "
        f"{MODEL_PT} and web/public/ refreshed"
    )

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
