import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from config import CONFIG_NAME, EPOCHS_GAN, GAN_BATCH_SIZE, SEED
from data import train_data_loader, train_ds
from model import TextEncoder
from train import LitColorGAN


def load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    return mod


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    enc = load(TextEncoder(), "enc.pt")

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        enable_checkpointing=False,
    )

    gan = LitColorGAN(enc)  # type: ignore
    gan_trainer.fit(
        gan, train_data_loader(data_set=train_ds(), batch_size=GAN_BATCH_SIZE))

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
