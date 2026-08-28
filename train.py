from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from config import BATCH_SIZE, EMBED_SIZE, EPOCHS, H_SIZE, LR, NUM_LAYERS, WEIGHT_DECAY
from data import train_data_loader


def run_name() -> str:
    """Build a TensorBoard run name that encodes the training configuration."""
    time = datetime.now().strftime("%H:%M")
    return (
        f"emb{EMBED_SIZE}-h{H_SIZE}-l{NUM_LAYERS}-lr{LR}"
        f"-wd{WEIGHT_DECAY}-bs{BATCH_SIZE}-ep{EPOCHS}-{time}"
    )


def train():
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    data = train_data_loader()
    feeling_ce = nn.CrossEntropyLoss()
    emoji_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Starting training loop...\n")
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        total_emoji_loss = 0.0
        total_feeling_loss = 0.0

        for x, target_emoji, target_feeling in data:
            optimizer.zero_grad()

            emoji_logits, feeling_logits = model(x)

            # Losses for the two discrete heads.
            loss_feeling = feeling_ce(feeling_logits, target_feeling)
            loss_emoji = emoji_ce(emoji_logits, target_emoji)

            loss = loss_emoji + loss_feeling

            loss.backward()
            # Clip gradient norm to tame the loss spikes that Adam + a noisy
            # batch produce late in training.
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_emoji_loss += loss_emoji.item()
            total_feeling_loss += loss_feeling.item()

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if writer is not None:
            writer.add_scalar("loss/total", avg_loss, epoch)
            writer.add_scalar(
                "loss/emoji", total_emoji_loss / n_batches, epoch)
            writer.add_scalar(
                "loss/feeling", total_feeling_loss / n_batches, epoch)

    print("\nTraining completed successfully.")


@torch.no_grad()
def evaluate(model: Model, data) -> dict:
    """Return emoji/feeling accuracy over `data`."""
    model.eval()
    dataloader = DataLoader(data, batch_size=32)

    n = 0
    emoji_correct = 0
    feeling_correct = 0
    for x, target_emoji, target_feeling in dataloader:
        emoji_logits, feeling_logits = model(x)
        emoji_correct += (emoji_logits.argmax(dim=-1)
                          == target_emoji).sum().item()
        feeling_correct += (
            (feeling_logits.argmax(dim=-1) == target_feeling).sum().item()
        )
        n += x.size(0)

    return {
        "n": n,
        "emoji_acc": emoji_correct / n,
        "feeling_acc": feeling_correct / n,
    }


DOCS_DIR = Path(__file__).parent / "docs"
ONNX_OPSET = 18


class ExportWrapper(nn.Module):
    """Inference-only forward that ONNX can trace.

    The trained model uses `pack_padded_sequence`, which does not export. Here we
    run the LSTM over the full padded sequence and gather the top-layer hidden
    state at the last real character. For right-padded input the LSTM output at
    step t depends only on steps 0..t, so `out[b, len_b - 1]` is exactly the
    packed `h_n[-1]` the classifier reads during training.
    """

    def __init__(self, m: Model):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor):
        mask = x != PAD_IDX
        lengths = mask.sum(dim=1).clamp(min=1)

        emb = self.m.embedding(x)
        conv_out = self.m.relu(self.m.conv(emb.permute(0, 2, 1)))
        conv_out = conv_out * mask.unsqueeze(1).to(conv_out.dtype)

        out, _ = self.m.rnn(conv_out.permute(0, 2, 1))
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        # dropout is identity in eval mode
        last = out.gather(1, idx).squeeze(1)

        return self.m.emoji(last), self.m.feeling(last)


def export_onnx(model: Model, dst: Path) -> None:
    """Trace `model` (via ExportWrapper) to an ONNX file the web page runs."""
    dummy = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long)
    torch.onnx.export(
        ExportWrapper(model).eval(),
        (dummy,),
        str(dst),
        input_names=["input"],
        output_names=["emoji_logits", "feeling_logits"],
        opset_version=ONNX_OPSET,
        external_data=False,  # keep weights inside the single .onnx file
        dynamic_axes={
            "input": {0: "batch"},
            "emoji_logits": {0: "batch"},
            "feeling_logits": {0: "batch"},
        },
    )


def write_meta(dst: Path) -> None:
    """Write everything docs/app.js must not hardcode from the Python side."""
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": feeling,
        "feeling_palette": {name: feeling_colors(name) for name in feeling},
    }
    dst.write_text(json.dumps(meta, ensure_ascii=False,
                   indent=2), encoding="utf-8")


def export_web(model: Model | None = None) -> None:
    """Refresh docs/model.onnx and docs/meta.json for the static Pages site.

    Called at the end of a training run with the freshly trained `model`; with
    no argument (``uv run main.py --export-only``) it loads the committed
    `model.pt` instead. The rest of docs/ is static and committed as-is.
    """
    if model is None:
        ckpt = Path("model.pt")
        if not ckpt.is_file():
            raise SystemExit(
                "model.pt not found -- train first with `uv run main.py`")
        model = Model()
        model.load_state_dict(torch.load(
            ckpt, map_location="cpu", weights_only=True))
    model.eval()

    DOCS_DIR.mkdir(exist_ok=True)
    export_onnx(model, DOCS_DIR / "model.onnx")
    write_meta(DOCS_DIR / "meta.json")
    print(f"Wrote {DOCS_DIR.name}/model.onnx and {DOCS_DIR.name}/meta.json")


def predict(model: Model, text: str) -> dict:
    """Run inference for a single string, returning a plain-dict result.

    Colors are not predicted; they are looked up from the predicted feeling.
    """
    model.eval()
    text = normalize(text)
    with torch.no_grad():
        emoji_logits, feeling_logits = model(encode(text, normalized=True))
    feeling_name = feeling[feeling_logits.argmax(dim=-1).item()]
    return {
        "text": text,
        "emoji": EMOJIS[emoji_logits.argmax(dim=-1).item()],
        "feeling": feeling_name,
        **feeling_colors(feeling_name),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the emojic model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Warm-start from model.pt instead of random init.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training; rewrite docs/model.onnx + docs/meta.json from model.pt.",
    )
    args = parser.parse_args()

    if args.export_only:
        export_web()
        raise SystemExit(0)

    torch.manual_seed(0)

    train_set, test_set = load_split()

    print(f"Train: {len(train_set)}  Test: {len(test_set)}\n")

    model = Model()
    if args.resume:
        ckpt = Path("model.pt")
        if not ckpt.is_file():
            raise SystemExit("--resume: model.pt not found")
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )
        print("Resumed from model.pt (optimizer state is not restored)\n")

    name = run_name()
    writer = SummaryWriter(log_dir=str(Path("runs") / name))
    print(f"TensorBoard run: runs/{name}\n")

    train(
        model=model,
        data=train_set,
        writer=writer,
    )

    metrics = evaluate(model, test_set)
    print(
        f"\nTest ({metrics['n']}): "
        f"emoji_acc={metrics['emoji_acc']:.2f}  "
        f"feeling_acc={metrics['feeling_acc']:.2f}"
    )
    writer.add_scalar("test/emoji_acc", metrics["emoji_acc"])
    writer.add_scalar("test/feeling_acc", metrics["feeling_acc"])
    writer.add_hparams(
        {
            "embed_size": EMBED_SIZE,
            "h_size": H_SIZE,
            "num_layers": NUM_LAYERS,
            "max_text_len": MAX_TEXT_LEN,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        {
            "test/emoji_acc": metrics["emoji_acc"],
            "test/feeling_acc": metrics["feeling_acc"],
        },
        run_name=".",
    )
    writer.close()

    torch.save(model.state_dict(), "model.pt")
    print("Saved model to model.pt")

    export_web(model)

    sample_text = "party time!"
    result = predict(model, sample_text)
    print(f"\nInference on '{sample_text}':")
    print(f"Predicted Emoji:    {result['emoji']}")
    print(f"Predicted Feeling:  {result['feeling']}")
    print(f"Background Gradient: {result['bg1']} -> {result['bg2']}")
    print(f"Text Color:         {result['text_color']}")
