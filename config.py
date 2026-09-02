import json
from datetime import datetime

# DATA
with open('labels.json', encoding='utf-8') as f:
    LABELS = json.load(f)

STYLES = LABELS["styles"]
EMOJIS = LABELS["emojis"]

MAX_TEXT_LEN = 42
BATCH_SIZE = 512

# MODEL
CHAR_EMBED_SIZE = 16
TEXT_ENCODER_CHANNELS = [64, 96]
ENCODER_KERNEL = 3
ENCODER_RELU_SLOPE = 0.1

EMOJI_EMBED_SIZE = 48
STYLE_EMBED_SIZE = 16
TEXT_EMBED_SIZE = TEXT_ENCODER_CHANNELS[-1]
DROPOUT_STYLE = 0.2
DROPOUT_EMOJI = 0.2

# GAN
Z_WEIGHT = 0.1
GEN_CHANNELS = [64]
GEN_RELU_SLOPE = 0.2
CRITIC_CHANNELS = [96, 64, 32, 16]
CRITIC_RELU_SLOPE = 0.2


# TRAINING
SEED = 42
LR = 0.01
GAN_LR = 0.01
GRAD_CLIP = 1.0
INFONCE_TEMP = 0.1

TASK_EPOCHS = 100
GAN_EPOCHS = 40
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE = 20

# TENSORBOARD RUN NAME
model_str = ' '.join([
    str(p) for p in
    (CHAR_EMBED_SIZE, TEXT_ENCODER_CHANNELS, ENCODER_KERNEL, DROPOUT_STYLE)])

train_str = ' '.join([
    str(p) for p in
    (LR, GAN_LR, BATCH_SIZE, GRAD_CLIP, INFONCE_TEMP)])

CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {model_str}',
    f'TRAIN: {train_str}',
])
