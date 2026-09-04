import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EPOCHS_GAN,
    GAN_BATCH_SIZE,
    SEED,
)
from data import (
    eval_data_loader,
    train_data_loader,
    train_ds,
)
from export_onnx import export
from model import TextEncoder
from runmeta import require_clean_tree, save_pt
from train import LitColorGAN


def load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    return mod


if __name__ == "__main__":
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    enc = load(TextEncoder(), "enc.pt")

    gan_ckpt = ModelCheckpoint(
        monitor="energy/gan/val", mode="min", save_top_k=1, filename="best-gan-{step}"
    )

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        callbacks=[
            gan_ckpt,
            EarlyStopping(
                monitor="energy/gan/val", mode="min", patience=EARLY_STOP_PATIENCE
            ),
            TQDMProgressBar(),
            ModelSummary(),
        ],
    )

    gan = LitColorGAN(enc)  # type: ignore
    gan_trainer.fit(
        gan,
        train_data_loader(data_set=train_ds(), batch_size=GAN_BATCH_SIZE),
        eval_data_loader(),
    )

    if gan_ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(gan_ckpt.best_model_path, enc=enc)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="gan")

    export()
