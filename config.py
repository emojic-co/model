import json
from datetime import datetime

# DATA
with open('labels.json', encoding='utf-8') as f:
    LABELS = json.load(f)

STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

# DATA
MAX_TEXT_LEN = 42

# ENCODER
CHAR_EMBED_SIZE = 16
ENCODER_CHANNELS = [80, 100, 120]
ENCODER_KERNEL = 3
assert ENCODER_KERNEL % 2 == 1, "Encoder kernel size must be odd"

enc_str = ' '.join([str(p) for p in (
    CHAR_EMBED_SIZE,
    ENCODER_CHANNELS,
    ENCODER_KERNEL)])

# EMOJI
EMOJI_EMBED_SIZE = 48
DROPOUT_EMOJI = 0.2

emj_str = ' '.join([str(p) for p in (
    EMOJI_EMBED_SIZE,
    DROPOUT_EMOJI)])

# STYLE
STYLE_EMBED_SIZE = 12
TEXT_EMBED_SIZE = ENCODER_CHANNELS[-1]
DROPOUT_STYLE = 0.2

style_str = ' '.join([str(p) for p in (
    STYLE_EMBED_SIZE,
    TEXT_EMBED_SIZE,
    DROPOUT_STYLE)])

# GAN
Z_WEIGHT = 0.2
GEN_CHANNELS = [96, 64]
CRITIC_CHANNELS = [32, 16, 8]
GAN_LR = 0.01

gan_str = ' '.join([str(p) for p in (
    Z_WEIGHT,
    GEN_CHANNELS,
    CRITIC_CHANNELS,
    GAN_LR)])


# TRAINING
SEED = 42
TASK_BATCH_SIZE = 256
GAN_BATCH_SIZE = 512
RELU_SLOPE = 0.1
LR = 0.01
GRAD_CLIP = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

train_str = ' '.join([str(p) for p in (
    SEED,
    TASK_BATCH_SIZE,
    GAN_BATCH_SIZE,
    RELU_SLOPE,
    LR,
    GRAD_CLIP,
    FOCAL_ALPHA,
    FOCAL_GAMMA)])

EPOCHS_TASK = 200
EPOCHS_GAN = 50
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE = 20

# METRICS
EMOJI_AP_K = 10
STYLE_AP_K = 5

# TENSORBOARD RUN NAME
CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'ENCODER: {enc_str}',
    f'EMOJI: {emj_str}',
    f'STYLE: {style_str}',
    f'GAN: {gan_str}',
    f'TRAIN: {train_str}',
])
