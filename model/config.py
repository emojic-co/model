import json
from datetime import datetime

from files import ENERGY_KEYWORDS_TXT, LABELS_JSON

# DATA
with open(LABELS_JSON, encoding="utf-8") as f:
    LABELS = json.load(f)

STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

# DATA
MAX_TEXT_LEN = 42

# ENCODER
CHAR_EMBED_SIZE = 16
ENCODER_KERNEL_SIZE = 3
assert ENCODER_KERNEL_SIZE % 2 == 1, "encoder kernel size must be odd"
ENCODER_CHANNELS = [96, 112, 128]

enc_str = " ".join(
    [str(p) for p in (CHAR_EMBED_SIZE, ENCODER_KERNEL_SIZE, ENCODER_CHANNELS)]
)

# EMOJI
EMOJI_EMBED_SIZE = 32
DROPOUT_EMOJI = 0.2

emj_str = " ".join([str(p) for p in (EMOJI_EMBED_SIZE, DROPOUT_EMOJI)])

# STYLE
STYLE_EMBED_SIZE = 12
TEXT_EMBED_SIZE = sum(ENCODER_CHANNELS)
DROPOUT_STYLE = 0.1

style_str = " ".join([str(p)
                     for p in (STYLE_EMBED_SIZE, TEXT_EMBED_SIZE, DROPOUT_STYLE)])

# GAN
Z_WEIGHT = 0.2
GEN_CHANNELS = [32]
CRITIC_CHANNELS = [256]
GAN_LR = 0.01

gan_str = " ".join([str(p)
                   for p in (Z_WEIGHT, GEN_CHANNELS, CRITIC_CHANNELS, GAN_LR)])


# TRAINING
SEED = 42
TASK_BATCH_SIZE = 256
GAN_BATCH_SIZE = 512
RELU_SLOPE = 0.1
LR = 0.01
GRAD_CLIP = 1.0
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
            GRAD_CLIP,
            INFONCE_TEMP,
        )
    ]
)

EPOCHS_TASK = 400
EPOCHS_GAN = 20
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
CONFIG_PARTS = [
    f"ENCODER: {enc_str}",
    f"EMOJI: {emj_str}",
    f"STYLE: {style_str}",
    f"GAN: {gan_str}",
    f"TRAIN: {train_str}",
]
CONFIG_NAME = " | ".join(
    [
        f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        *CONFIG_PARTS,
    ]
)
