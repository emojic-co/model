
import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    ENERGY_KEYWORD_MAX_TEXTS,
    ENERGY_KEYWORD_MIN_TEXTS,
    ENERGY_KEYWORDS_PATH,
    EPOCHS_GAN,
    GAN_BATCH_SIZE,
    SEED,
)
from data import (
    eval_data_loader,
    keyword_index,
    load_energy_keywords,
    train_data_loader,
    train_ds,
)
from export_onnx import export
from model import TextEncoder
from train import LitColorGAN


def load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    return mod


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    enc = load(TextEncoder(), "enc.pt")

    kw_index = keyword_index(
        load_energy_keywords(ENERGY_KEYWORDS_PATH),
        max_texts=ENERGY_KEYWORD_MAX_TEXTS,
        min_texts=ENERGY_KEYWORD_MIN_TEXTS,
        seed=SEED)

    gan_ckpt = ModelCheckpoint(
        monitor="energy/gan/val",
        mode="min",
        save_top_k=1,
        filename="best-gan-{step}")

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        callbacks=[
            gan_ckpt,
            EarlyStopping(
                monitor="energy/gan/val", mode="min",
                patience=EARLY_STOP_PATIENCE),
        ],
    )

    gan = LitColorGAN(enc, kw_index)  # type: ignore
    gan_trainer.fit(
        gan,
        train_data_loader(data_set=train_ds(), batch_size=GAN_BATCH_SIZE),
        eval_data_loader())

    if gan_ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(
            gan_ckpt.best_model_path, enc=enc, kw_index=kw_index)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")

    export()
