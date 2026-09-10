import json
import os
import sys
from datetime import datetime
from pathlib import Path

from files import ENERGY_KEYWORDS_TXT, LABELS_JSON

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# DATA
with open(LABELS_JSON, encoding="utf-8") as f:
    LABELS = json.load(f)

STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

STYLE_COUNT = len(STYLES)
EMOJI_COUNT = len(EMOJIS)

# DATA
# Sized for the step-1 keyword-search model (docs/step1-keyword-search.md):
# small vocab (~170), short inputs, ~1/5 encoder width.
MAX_TEXT_LEN = 32

# ENCODER
CHAR_EMBED_SIZE = 16
ENCODER_KERNEL_SIZE = 3

assert ENCODER_KERNEL_SIZE % 2 == 1, \
    "encoder kernel size must be odd"

ENCODER_CHANNELS = [64, 96]
ENCODER_DILATION = [1, 2]
TEXT_EMBED_SIZE = sum(ENCODER_CHANNELS)

assert len(ENCODER_CHANNELS) == len(ENCODER_DILATION), \
    "encoder channels and dilation must have the same length"

enc_str = " ".join([
    str(p) for p in (
        CHAR_EMBED_SIZE,
        ENCODER_KERNEL_SIZE,
        ENCODER_CHANNELS,
        ENCODER_DILATION)])

# EMOJI
EMOJI_EMBED_SIZE = 32
DROPOUT_EMOJI = 0.1

emj_str = " ".join([str(p) for p in (EMOJI_EMBED_SIZE, DROPOUT_EMOJI)])

# STYLE
STYLE_EMBED_SIZE = 16
DROPOUT_STYLE = 0.5

style_str = " ".join([
    str(p)
    for p in (STYLE_EMBED_SIZE, TEXT_EMBED_SIZE, DROPOUT_STYLE)])

# GAN
Z_WEIGHT = 0.35
GEN_CHANNELS = [64, 32]
CRITIC_COLOR_CHANNELS = [96]
CRITIC_TEXT_CHANNELS = [64]
DROPOUT_CRITIC = 0.2
GAN_GEN_LR = 0.01
GAN_CRITIC_LR = 0.02
GAN_GEN_MARGIN = 1.0
ENERGY_WEIGHT = 5.0

gan_str = " ".join([
    str(p)
    for p in (
        Z_WEIGHT,
        GEN_CHANNELS,
        CRITIC_COLOR_CHANNELS,
        CRITIC_TEXT_CHANNELS,
        DROPOUT_CRITIC,
        GAN_GEN_LR,
        GAN_CRITIC_LR,
        GAN_GEN_MARGIN,
        ENERGY_WEIGHT)])


# TRAINING
SEED = 42
TASK_BATCH_SIZE = int(os.environ.get("EMOJIC_TASK_BATCH_SIZE", "128"))
GAN_BATCH_SIZE = int(os.environ.get("EMOJIC_GAN_BATCH_SIZE", "512"))
RELU_SLOPE = 0.1
LR = 0.01
GRAD_CLIP_GEN = 1.0
GRAD_CLIP_CRITIC = 10.0
INFONCE_TEMP = 0.7

train_str = " ".join(
    [
        str(p)
        for p in (
            SEED,
            TASK_BATCH_SIZE,
            GAN_BATCH_SIZE,
            RELU_SLOPE,
            LR,
            GRAD_CLIP_GEN,
            GRAD_CLIP_CRITIC,
            INFONCE_TEMP,
        )
    ]
)

EPOCHS_TASK = 1500
EPOCHS_GAN = 300
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE = 20

# METRICS
EMOJI_AP_K = 10
STYLE_AP_K = 5

# ENERGY EVAL
ENERGY_Z_SAMPLES = 8
ENERGY_KEYWORD_MAX_TEXTS = 512
ENERGY_KEYWORD_MIN_TEXTS = 32
ENERGY_KEYWORDS_PATH = ENERGY_KEYWORDS_TXT

# TENSORBOARD RUN NAME
RUN_TIME = os.environ.get(
    "EMOJIC_RUN_TIME") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CONFIG_PARTS = [
    f"ENCODER: {enc_str}",
    f"EMOJI: {emj_str}",
    f"STYLE: {style_str}",
    f"GAN: {gan_str}",
    f"TRAIN: {train_str}",
]
CONFIG_NAME = " | ".join(
    [
        f"TIME: {RUN_TIME}",
        *CONFIG_PARTS,
    ]
)


def _receptive_field() -> int:
    return 1 + (ENCODER_KERNEL_SIZE - 1) * sum(ENCODER_DILATION)


def _effective_kernels() -> list[int]:
    return [(ENCODER_KERNEL_SIZE - 1) * d + 1 for d in ENCODER_DILATION]


def _encoder_conv_params() -> int:
    total = 0
    in_ch = CHAR_EMBED_SIZE
    for out_ch in ENCODER_CHANNELS:
        total += in_ch * out_ch * ENCODER_KERNEL_SIZE + out_ch
        in_ch = out_ch
    return total


def _head_params(embed_size: int, n_labels: int) -> int:
    return TEXT_EMBED_SIZE * embed_size + n_labels * (embed_size + 1)


def _stats() -> list[tuple[str, object]]:
    rf = _receptive_field()
    cover = "covers full input" if rf >= MAX_TEXT_LEN else "partial coverage"
    enc = _encoder_conv_params()
    emoji_head = _head_params(EMOJI_EMBED_SIZE, len(EMOJIS))
    style_head = _head_params(STYLE_EMBED_SIZE, len(STYLES))
    chain = " -> ".join(str(c) for c in (CHAR_EMBED_SIZE, *ENCODER_CHANNELS))
    return [
        ("NUM_LAYERS", len(ENCODER_CHANNELS)),
        ("channel chain", chain),
        ("ENCODER_KERNEL_SIZE", ENCODER_KERNEL_SIZE),
        ("ENCODER_DILATION", ENCODER_DILATION),
        ("effective kernel / layer", _effective_kernels()),
        ("RECEPTIVE_FIELD", rf),
        ("RF vs MAX_TEXT_LEN", f"{rf} / {MAX_TEXT_LEN}  ({cover})"),
        ("TEXT_EMBED_SIZE", TEXT_EMBED_SIZE),
        ("MAX_TEXT_LEN", MAX_TEXT_LEN),
        ("# styles", len(STYLES)),
        ("# emojis", len(EMOJIS)),
        ("STYLE_EMBED_SIZE", STYLE_EMBED_SIZE),
        ("EMOJI_EMBED_SIZE", EMOJI_EMBED_SIZE),
        ("encoder conv params", f"{enc:,}"),
        ("style head params", f"{style_head:,}"),
        ("emoji head params", f"{emoji_head:,}"),
        ("PARAM_COUNT (enc + heads)", f"{enc + style_head + emoji_head:,}"),
        ("TASK_BATCH_SIZE / GAN_BATCH_SIZE",
         f"{TASK_BATCH_SIZE} / {GAN_BATCH_SIZE}"),
        ("EPOCHS_TASK / EPOCHS_GAN", f"{EPOCHS_TASK} / {EPOCHS_GAN}"),
        ("Z_WEIGHT", Z_WEIGHT),
        ("GEN_CHANNELS", GEN_CHANNELS),
        ("CRITIC_COLOR_CHANNELS", CRITIC_COLOR_CHANNELS),
        ("CRITIC_TEXT_CHANNELS", CRITIC_TEXT_CHANNELS),
        ("DROPOUT_CRITIC", DROPOUT_CRITIC),
    ]


def main() -> None:
    rows = _stats()
    width = max(len(name) for name, _ in rows)
    for name, val in rows:
        print(f"{name:<{width}} : {val}")
    print()
    print(
        "PARAM_COUNT excludes the char-embedding table (needs |CHARS| from "
        "model/data.py) and the GAN generator/critic."
    )
    print("Run tools/print_model_params.py for exact per-module counts.")


if __name__ == "__main__":
    main()
