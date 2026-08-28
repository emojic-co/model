"""Train the emojic CNN baseline.

Runs the training loop, evaluates on the held-out split after every epoch, and
keeps the best checkpoint (by mean of emoji/feeling accuracy). Every time the
best improves it rewrites both ``model.pt`` and the static web app's artifacts
in ``docs/`` (``model.onnx`` + ``meta.json``), so the page can be watched live
during a run.
"""

import json
import warnings
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CONFIG_NAME,
    EPOCHS,
    EVAL_EPOCHS,
    GRAD_CLIP,
    LABEL_SMOOTHING,
    LR,
    MAX_TEXT_LEN,
    WEIGHT_DECAY,
)
from data import (
    CHARS,
    EMOJIS,
    FEELING,
    PAD_IDX,
    collate_fn,
    data_sets,
    train_data_loader,
)
from model import Model

MODEL_PT = Path("model.pt")
DOCS = Path("docs")
ONNX_OPSET = 18


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Return emoji/feeling accuracy over ``loader``."""
    model.eval()
    n = emoji_correct = feeling_correct = 0
    for x, target_emoji, target_feeling in loader:
        emoji_logits, feeling_logits = model(x)
        emoji_correct += (emoji_logits.argmax(-1) == target_emoji).sum().item()
        feeling_correct += (feeling_logits.argmax(-1) ==
                            target_feeling).sum().item()
        n += x.size(0)
    return {"emoji_acc": emoji_correct / n, "feeling_acc": feeling_correct / n}


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
            output_names=["emoji_logits", "feeling_logits"],
            opset_version=ONNX_OPSET,
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch"},
                "emoji_logits": {0: "batch"},
                "feeling_logits": {0: "batch"},
            },
        )


def export_web(model: nn.Module) -> None:
    """Refresh docs/model.onnx + docs/meta.json for the backend-free web app.

    meta.json carries everything docs/app.js must not hardcode from the Python
    side: the char vocab, MAX_TEXT_LEN, and the label sets. (The feeling color
    palette is not here -- it lives in docs/palette.json, read directly by app.js.)
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


def train() -> None:
    torch.manual_seed(0)

    train_ds, eval_ds = data_sets()
    train_loader = train_data_loader(train_ds)
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = Model()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=f"runs/{CONFIG_NAME}")

    emoji_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    feeling_ce = nn.CrossEntropyLoss()

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_acc = -1.0
    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for x, target_emoji, target_feeling in train_loader:
            optimizer.zero_grad()
            emoji_logits, feeling_logits = model(x)
            loss = emoji_ce(emoji_logits, target_emoji) + feeling_ce(
                feeling_logits, target_feeling
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        postfix = {"loss": f"{avg_loss:.4f}"}
        writer.add_scalar("loss/train", avg_loss, epoch)

        if epoch % EVAL_EPOCHS == 0 or epoch == EPOCHS:
            m = evaluate(model, eval_loader)
            mean_acc = (m["emoji_acc"] + m["feeling_acc"]) / 2
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), MODEL_PT)
                # keep docs/ (model.onnx + meta.json) in sync
                export_web(model)
            writer.add_scalar("acc/emoji", m["emoji_acc"], epoch)
            writer.add_scalar("acc/feeling", m["feeling_acc"], epoch)
            writer.add_scalar("acc/mean", mean_acc, epoch)
            writer.add_scalar("acc/best", best_acc, epoch)
            postfix |= {
                "emoji_acc": f"{m['emoji_acc']:.3f}",
                "feeling_acc": f"{m['feeling_acc']:.3f}",
                "best": f"{best_acc:.3f}",
            }

        pbar.set_postfix(postfix)

    writer.close()
    print(
        f"\nBest mean acc: {best_acc:.3f}  ->  {MODEL_PT} and docs/ refreshed")


if __name__ == "__main__":
    train()
