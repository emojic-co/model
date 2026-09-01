import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from config import CONFIG_NAME, GAN_EPOCHS, GEN_Z_DIM, SEED
from data import train_data_loader
from model import EmojiHead, TextEncoder
from train import LitColorGAN

FIXED_Z = torch.randn(1, GEN_Z_DIM, generator=torch.Generator().manual_seed(SEED))


def load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    return mod


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    enc = load(TextEncoder(), "enc.pt")
    emoji = load(EmojiHead(), "emoji.pt")

    gan = LitColorGAN(enc, emoji)
    gan.gen.sample_z = lambda n, device=None: FIXED_Z.to(device).expand(n, -1)

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan-fixedz", default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=GAN_EPOCHS,
        enable_checkpointing=False,
    )

    gan_trainer.fit(gan, train_data_loader())

    for name, mod in (("gen", gan.gen), ("tst", gan.tst)):
        torch.save(mod.state_dict(), f"{name}.fixedz.pt")
