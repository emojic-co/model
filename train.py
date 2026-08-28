"""Train the emojic CNN baseline.

Runs the training loop, evaluates on the held-out split after every epoch, and
keeps the best checkpoint (by total eval loss, emoji + feeling). Every time the
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
    LR,
    MAX_TEXT_LEN,
    NEGATIVE_SAMPLES,
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
    emoji_ce: nn.Module,
    feeling_ce: nn.Module,
) -> dict:
    """Return emoji/feeling loss and accuracy over ``loader``.

    Losses use the same criteria as training (label smoothing included) and are
    reported as per-sample means, weighting each batch by its size.
    """
    model.eval()
    n = emoji_correct = feeling_correct = 0
    emoji_loss_sum = feeling_loss_sum = 0.0
    for x, target_emoji, target_feeling in loader:
        bs = x.size(0)
        emoji_logits, feeling_logits = model(x)
        emoji_loss_sum += emoji_ce(emoji_logits, target_emoji).item() * bs
        feeling_loss_sum += feeling_ce(feeling_logits,
                                       target_feeling).item() * bs
        emoji_correct += (emoji_logits.argmax(-1) == target_emoji).sum().item()
        feeling_correct += (feeling_logits.argmax(-1) ==
                            target_feeling).sum().item()
        n += bs
    return {
        "emoji_loss": emoji_loss_sum / n,
        "feeling_loss": feeling_loss_sum / n,
        "emoji_acc": emoji_correct / n,
        "feeling_acc": feeling_correct / n,
    }


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


def sample_negatives(
    emoji_logits: torch.Tensor,
    target_emoji: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict the emoji cross-entropy to the true emoji + NEGATIVE_SAMPLES
    random negatives per row (sampled softmax instead of the full ~133-way
    InfoNCE). Column 0 of the returned logits is always the positive, so the
    matching targets are all zero.
    """
    bs, num_emojis = emoji_logits.shape
    weights = torch.ones(bs, num_emojis)
    weights.scatter_(
        1,
        target_emoji.unsqueeze(1), 0.0)  # never sample the positive
    neg = torch.multinomial(weights, NEGATIVE_SAMPLES, replacement=False)
    cand = torch.cat([target_emoji.unsqueeze(1), neg], dim=1)
    return emoji_logits.gather(1, cand), torch.zeros(bs, dtype=torch.long)


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

    emoji_ce = nn.CrossEntropyLoss()
    feeling_ce = nn.CrossEntropyLoss()

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_loss = float("inf")
    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        total_emoji_loss = total_feeling_loss = 0.0
        for x, target_emoji, target_feeling in train_loader:
            optimizer.zero_grad()
            emoji_logits, feeling_logits = model(x)
            cand_logits, cand_target = sample_negatives(
                emoji_logits, target_emoji)
            emoji_loss = emoji_ce(cand_logits, cand_target)
            feeling_loss = feeling_ce(feeling_logits, target_feeling)
            loss = emoji_loss + feeling_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_emoji_loss += emoji_loss.item()
            total_feeling_loss += feeling_loss.item()

        avg_emoji_loss = total_emoji_loss / len(train_loader)
        avg_feeling_loss = total_feeling_loss / len(train_loader)
        postfix = {"loss": f"{avg_emoji_loss + avg_feeling_loss:.4f}"}
        writer.add_scalar("train/emoji_loss", avg_emoji_loss, epoch)
        writer.add_scalar("train/feeling_loss", avg_feeling_loss, epoch)

        if epoch % EVAL_EPOCHS == 0 or epoch == EPOCHS:
            m = evaluate(model, eval_loader, emoji_ce, feeling_ce)
            eval_loss = m["emoji_loss"] + m["feeling_loss"]
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), MODEL_PT)
                # keep docs/ (model.onnx + meta.json) in sync
                export_web(model)
            writer.add_scalar("eval/emoji_loss", m["emoji_loss"], epoch)
            writer.add_scalar("eval/feeling_loss", m["feeling_loss"], epoch)
            postfix |= {
                "emoji_acc": f"{m['emoji_acc']:.3f}",
                "feeling_acc": f"{m['feeling_acc']:.3f}",
                "best_loss": f"{best_loss:.4f}",
            }

        pbar.set_postfix(postfix)

    writer.close()
    print(
        f"\nBest eval loss: {best_loss:.4f}  ->  {MODEL_PT} and docs/ refreshed")

    # Behavioral test suite + Markdown report (report/<MM-DD-HH:MM>.md).
    # Runs against the saved best checkpoint (model.pt), i.e. what ships.
    from test_model import run as run_tests

    run_tests()


if __name__ == "__main__":
    train()
