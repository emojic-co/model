"""Train the emojic CNN feeling classifier.

Runs the training loop, evaluates on the held-out split every ``EVAL_EPOCHS``
epochs, and keeps the best checkpoint (by eval feeling loss). Every time the
best improves it rewrites both ``model.pt`` and the static web app's artifacts
in ``docs/`` (``model.onnx`` + ``meta.json``), so the page can be watched live
during a run.

The data pipeline (``data.py``) still carries the emoji label per row, so an
emoji head can be added back later; the model and this script currently train
the feeling head only.
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    feeling_ce: nn.Module,
) -> dict:
    """Return feeling loss over ``loader``.

    The loss uses the same criterion as training and is reported as a
    per-sample mean, weighting each batch by its size.
    """
    model.eval()
    n = 0
    feeling_loss_sum = 0.0
    for x, _, target_feeling in loader:
        bs = x.size(0)
        feeling_logits = model(x)
        feeling_loss_sum += feeling_ce(feeling_logits, target_feeling).item() * bs
        n += bs
    return {"feeling_loss": feeling_loss_sum / n}


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

    feeling_ce = nn.CrossEntropyLoss()

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_loss = float("inf")
    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        total_feeling_loss = 0.0
        for x, _, target_feeling in train_loader:
            optimizer.zero_grad()
            feeling_logits = model(x)
            feeling_loss = feeling_ce(feeling_logits, target_feeling)
            feeling_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_feeling_loss += feeling_loss.item()

        avg_feeling_loss = total_feeling_loss / len(train_loader)
        pbar.set_postfix({"loss": f"{avg_feeling_loss:.4f}"})
        writer.add_scalar("train/feeling_loss", avg_feeling_loss, epoch)

        if epoch % EVAL_EPOCHS == 0 or epoch == EPOCHS:
            m = evaluate(model, eval_loader, feeling_ce)
            eval_loss = m["feeling_loss"]
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), MODEL_PT)
                # keep docs/ (model.onnx + meta.json) in sync
                export_web(model)
            writer.add_scalar("eval/feeling_loss", m["feeling_loss"], epoch)

    writer.close()
    print(
        f"\nBest eval loss: {best_loss:.4f}  ->  {MODEL_PT} and docs/ refreshed")

    # Behavioral test suite + Markdown report (report/<MM-DD-HH:MM>.md).
    # Runs against the saved best checkpoint (model.pt), i.e. what ships.
    from test_model import run as run_tests

    run_tests()


if __name__ == "__main__":
    train()
