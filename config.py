from datetime import datetime

# DATA
MAX_TEXT_LEN = 42

# MODEL
CHAR_EMBED_SIZE = 16
CONV = [
    (4, 128),
    (3, 256),
    # (2, 64)
]
POOL_1D_SIZE = 2

# DROPOUT
DROPOUT_FEELING = 0.5
DROPOUT_EMOJI = 0.5


# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0
INFONCE_TEMP = 0.1

EPOCHS = 500
EVAL_EPOCHS = 1
EARLY_STOP_PATIENCE = 12


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHAR_EMBED_SIZE} {CONV} d{DROPOUT_FEELING}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} tmp {INFONCE_TEMP}',
])
