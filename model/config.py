import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from files import KEYWORDS_JSONL, LABELS_JSON, TERMS_JSONL
from model.metric import Metric, Source, named_metric

# from files import FLAGS_JSONL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# DATA
with LABELS_JSON.open(encoding="utf-8") as f:
    LABELS = json.load(f)

LANGS = LABELS["langs"]
STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

LANG_COUNT = len(LANGS)
STYLE_COUNT = len(STYLES)
EMOJI_COUNT = len(EMOJIS)

# DATA
# Sized for a small vocab, short inputs, ~1/5 encoder width.
MAX_TEXT_LEN = 42

# ENCODER
ENCODER_KERNEL_SIZE = 3


assert ENCODER_KERNEL_SIZE % 2 == 1, \
    "encoder kernel size must be odd"

ENCODER_CHANNELS = [160, 220, 200, 60]
ENCODER_DILATION = [1, 2, 4, 8]

# ENCODER LOSS WEIGHTS
LOSS_WEIGHT_COLOR_REGRESSION = 0.05
LOSS_WEIGHT_ENC_COND_COLOR_CRITIC = 0.1

# EMBEDDING
EMBED_SIZE_CHAR = 30
EMBED_SIZE_TEXT = 300
EMBED_SIZE_EMOJI = 60
EMBED_SIZE_STYLE = 12

# DROPOUT
DROPOUT = 0.1

assert len(ENCODER_CHANNELS) == len(ENCODER_DILATION), \
    "encoder channels and dilation must have the same length"

enc_str = " ".join([
    str(p) for p in (
        EMBED_SIZE_CHAR,
        ENCODER_KERNEL_SIZE,
        ENCODER_CHANNELS,
        ENCODER_DILATION)])


emj_str = " ".join([str(p) for p in (EMBED_SIZE_EMOJI,)])
style_str = " ".join([
    str(p)
    for p in (EMBED_SIZE_STYLE, EMBED_SIZE_TEXT,)])

# GAN
Z_WEIGHT = 0.3
NOISE_DIM = round(Z_WEIGHT * EMBED_SIZE_TEXT)

GEN_HIDDEN_SIZE = 80
CRITIC_HIDDEN_SIZE = 32
COND_CRITIC_EMBEDDING_SIZE = 10

GRAD_CLIP_GEN = 1
GRAD_CLIP_CRITIC = 1

# GAN LOSS WEIGHTS
LOSS_WEIGHT_ENERGY = 0.01
LOSS_WEIGHT_COLOR_CRITIC = .5
LOSS_WEIGHT_COND_COLOR_CRITIC = 1

# ENERGY
ENERGY_TRAIN_SAMPLE_SIZE = 128
ENERGY_VAL_SAMPLE_SIZE = 1024

# LR
LR_ENCODER = 0.01
LR_GAN_GEN = 0.005
LR_GAN_CRITIC = 0.005
LR_GAN_COND_CRITIC = 0.005

# TRAINING
SEED = 42
TASK_BATCH_SIZE = 1024
GAN_BATCH_SIZE = 1024
RELU_SLOPE = 0.1
INFONCE_TEMP_EMOJI = 0.7
INFONCE_TEMP_STYLE = 0.7

gan_str = " ".join([
    str(p)
    for p in (
        Z_WEIGHT,
        NOISE_DIM,
        GEN_HIDDEN_SIZE,
        COND_CRITIC_EMBEDDING_SIZE,
        LR_GAN_GEN,
        LR_GAN_CRITIC,
        LR_GAN_COND_CRITIC,
        LOSS_WEIGHT_ENERGY,
        LOSS_WEIGHT_COND_COLOR_CRITIC,
        LOSS_WEIGHT_COLOR_CRITIC)])


@dataclass(frozen=True)
class SamplingSource:
    path: Path
    metric: str
    from_: float
    to: float


SAMPLING_RATE_MAX = 0.15
SAMPLING_RATE_MIN = 0.01

SAMPLING_SOURCES: dict[Source, SamplingSource] = {
    Source.KEYWORD: SamplingSource(
        KEYWORDS_JSONL,
        named_metric(Source.KEYWORD, Metric.ACC_1),
        0, 0.95
    ),
    Source.TERM: SamplingSource(
        TERMS_JSONL,
        named_metric(Source.TERM, Metric.ACC_1),
        0, 0.9
    ),
}


train_str = " ".join(
    [
        str(p)
        for p in (
            SEED,
            TASK_BATCH_SIZE,
            GAN_BATCH_SIZE,
            RELU_SLOPE,
            LR_ENCODER,
            GRAD_CLIP_GEN,
            GRAD_CLIP_CRITIC,
            INFONCE_TEMP_EMOJI,
            SAMPLING_RATE_MAX,
            SAMPLING_RATE_MIN,
        )
    ]
)

EPOCHS_TASK = 1500
EPOCHS_GAN = 300
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE_ENCODER = 30
EARLY_STOP_PATIENCE_GAN = 200
EARLY_STOP_PATIENCE_ENERGY = 30
EARLY_STOP_MIN_DELTA_GAN = 0.005

# METRICS
MACRO_MIN_SUPPORT = 5

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
    in_ch = EMBED_SIZE_CHAR
    for out_ch in ENCODER_CHANNELS:
        total += in_ch * out_ch * ENCODER_KERNEL_SIZE + out_ch
        in_ch = out_ch
    return total


def _encoder_proj_params() -> int:
    pooled = sum(ENCODER_CHANNELS)
    return pooled * EMBED_SIZE_TEXT + 2 * EMBED_SIZE_TEXT


def _head_params(embed_size: int, n_labels: int) -> int:
    return EMBED_SIZE_TEXT * embed_size + n_labels * (embed_size + 1)


def _lang_head_params(n_labels: int) -> int:
    return EMBED_SIZE_TEXT * n_labels + n_labels


def _stats() -> list[tuple[str, object]]:
    rf = _receptive_field()
    cover = "covers full input" if rf >= MAX_TEXT_LEN else "partial coverage"
    enc = _encoder_conv_params() + _encoder_proj_params()
    emoji_head = _head_params(EMBED_SIZE_EMOJI, len(EMOJIS))
    style_head = _head_params(EMBED_SIZE_STYLE, len(STYLES))
    lang_head = _lang_head_params(len(LANGS))
    chain = " -> ".join(
        str(c) for c in (EMBED_SIZE_CHAR, *ENCODER_CHANNELS, EMBED_SIZE_TEXT))
    return [
        ("NUM_LAYERS", len(ENCODER_CHANNELS)),
        ("channel chain", chain),
        ("ENCODER_KERNEL_SIZE", ENCODER_KERNEL_SIZE),
        ("ENCODER_DILATION", ENCODER_DILATION),
        ("effective kernel / layer", _effective_kernels()),
        ("RECEPTIVE_FIELD", rf),
        ("RF vs MAX_TEXT_LEN", f"{rf} / {MAX_TEXT_LEN}  ({cover})"),
        ("TEXT_EMBED_SIZE", EMBED_SIZE_TEXT),
        ("MAX_TEXT_LEN", MAX_TEXT_LEN),
        ("# langs", len(LANGS)),
        ("# styles", len(STYLES)),
        ("# emojis", len(EMOJIS)),
        ("STYLE_EMBED_SIZE", EMBED_SIZE_STYLE),
        ("EMOJI_EMBED_SIZE", EMBED_SIZE_EMOJI),
        ("encoder conv params", f"{enc:,}"),
        ("lang head params", f"{lang_head:,}"),
        ("style head params", f"{style_head:,}"),
        ("emoji head params", f"{emoji_head:,}"),
        ("PARAM_COUNT (enc + heads)",
         f"{enc + lang_head + style_head + emoji_head:,}"),
        ("TASK_BATCH_SIZE / GAN_BATCH_SIZE",
         f"{TASK_BATCH_SIZE} / {GAN_BATCH_SIZE}"),
        ("EPOCHS_TASK / EPOCHS_GAN", f"{EPOCHS_TASK} / {EPOCHS_GAN}"),
        ("Z_WEIGHT", Z_WEIGHT),
        ("NOISE_DIM", NOISE_DIM),
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
