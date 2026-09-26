import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from files import GOALS_YML, KEYWORDS_JSONL, LABELS_JSON, TERMS_JSONL
from model.metric import LogStage, Metric, Source, named_metric

# from files import FLAGS_JSONL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# DATA
with LABELS_JSON.open(encoding="utf-8") as f:
    LABELS = json.load(f)

STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

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

# EMBEDDING
EMBED_SIZE_CHAR = 32
EMBED_SIZE_EMOJI = 64
EMBED_SIZE_STYLE = 16
EMBED_SIZE_TEXT = 128

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

LOSS_WEIGHT_EMOJI = 1
LOSS_WEIGHT_STYLE = 1
LOSS_WEIGHT_COLOR = 0.1

# CLASSIFIER LOSS (INFONCE)
INFONCE_TEMP_EMOJI = 0.7
INFONCE_TEMP_STYLE = 0.7

emj_str = " ".join([str(p) for p in (EMBED_SIZE_EMOJI,)])

style_str = " ".join([
    str(p)
    for p in (EMBED_SIZE_STYLE, EMBED_SIZE_TEXT)])

# GAN
HIDDEN_SIZE_GEN = 64
HIDDEN_SIZE_CRITIC = 128

Z_WEIGHT = 0.3
GRAD_CLIP_GEN = 2
GRAD_CLIP_CRITIC = 1

# GAN LOSS WEIGHTS
GAN_LOSS_ENERGY = 0
GAN_LOSS_CRITIC = 1

# ENERGY
ENERGY_TRAIN_SAMPLE_SIZE = 128
ENERGY_VAL_SAMPLE_SIZE = 1024

_COLOR_GEN_GOALS = yaml.safe_load(
    GOALS_YML.read_text(encoding="utf-8"))["goals"]["color generator"]
COLOR_ENERGY_KEYWORDS = list(_COLOR_GEN_GOALS["energy distance"])

# LR
LR_ENCODER = 0.01
LR_GAN_GEN = 0.001
LR_GAN_CRITIC = 0.00005

# TRAINING
SEED = 42
BATCH_SIZE_TEXT_ENCODER = 1024
BATCH_SIZE_GAN = 1024
RELU_SLOPE = 0.1

gan_str = " ".join([
    str(p)
    for p in (
        HIDDEN_SIZE_GEN,
        HIDDEN_SIZE_CRITIC,
        LR_GAN_GEN,
        LR_GAN_CRITIC,
        GAN_LOSS_ENERGY,
        GAN_LOSS_CRITIC)])


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
        named_metric(LogStage.ENC, Source.KEYWORD, Metric.ACC_1),
        0, 0.95
    ),
    Source.TERM: SamplingSource(
        TERMS_JSONL,
        named_metric(LogStage.ENC, Source.TERM, Metric.ACC_1),
        0, 0.9
    ),
}


train_str = " ".join(
    [
        str(p)
        for p in (
            SEED,
            BATCH_SIZE_TEXT_ENCODER,
            BATCH_SIZE_GAN,
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
EPOCHS_GAN = 80
EPOCHS_COND_PROBE = 100
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE_ENCODER = 30
EARLY_STOP_PATIENCE_ENERGY = 30

# TENSORBOARD RUN NAME
RUN_TIME = os.environ.get(
    "EMOJIC_RUN_TIME") or datetime.now().strftime("%Y%m%d-%H%M%S")


def _git_sha() -> str:
    sha = os.environ.get("EMOJIC_GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


GIT_SHA = _git_sha()
CONFIG_PARTS = [
    f"ENCODER: {enc_str}",
    f"EMOJI: {emj_str}",
    f"STYLE: {style_str}",
    f"GAN: {gan_str}",
    f"TRAIN: {train_str}",
]
CONFIG_NAME = f"{RUN_TIME}-{GIT_SHA}"


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


def _stats() -> list[tuple[str, object]]:
    rf = _receptive_field()
    cover = "covers full input" if rf >= MAX_TEXT_LEN else "partial coverage"
    enc = _encoder_conv_params() + _encoder_proj_params()
    emoji_head = _head_params(EMBED_SIZE_EMOJI, len(EMOJIS))
    style_head = _head_params(EMBED_SIZE_STYLE, len(STYLES))
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
        ("# styles", len(STYLES)),
        ("# emojis", len(EMOJIS)),
        ("STYLE_EMBED_SIZE", EMBED_SIZE_STYLE),
        ("EMOJI_EMBED_SIZE", EMBED_SIZE_EMOJI),
        ("encoder conv params", f"{enc:,}"),
        ("style head params", f"{style_head:,}"),
        ("emoji head params", f"{emoji_head:,}"),
        ("PARAM_COUNT (enc + heads)",
         f"{enc + style_head + emoji_head:,}"),
        ("TASK_BATCH_SIZE / GAN_BATCH_SIZE",
         f"{BATCH_SIZE_TEXT_ENCODER} / {BATCH_SIZE_GAN}"),
        ("EPOCHS_TASK / EPOCHS_GAN", f"{EPOCHS_TASK} / {EPOCHS_GAN}"),
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
