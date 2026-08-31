from datetime import datetime

# DATA
MAX_TEXT_LEN = 42

# MODEL
CHAR_EMBED_SIZE = 16
CONV = [
    (3, 64),
    (3, 96),
    (3, 128),
]
POOL_1D_SIZE = 2

# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0

DROPOUT_FEELING = 0.3
DROPOUT_EMOJI = 0.2
DROPOUT_COLOR = 0.5

INFONCE_TEMP = 0.1

EPOCHS = 500
VAL_CHECK_INTERVAL = 100
EARLY_STOP_PATIENCE = 30

model_str = ' '.join([
    str(p) for p in
    (CHAR_EMBED_SIZE, CONV)])

train_str = ' '.join([
    str(p) for p in
    (LR, BATCH_SIZE, GRAD_CLIP, DROPOUT_FEELING, DROPOUT_COLOR, DROPOUT_EMOJI, INFONCE_TEMP)])


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {model_str}',
    f'TRAIN: {train_str}',
])
