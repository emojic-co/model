from datetime import datetime

MAX_TEXT_LEN = 42

CHAR_EMBED_SIZE = 16
CONV = [
    (4, 96),
    (3, 128),
    (2, 196)
]
POOL_1D_SIZE = 2
DROPOUT = 0.5

INFONCE_TEMP = 0.2

LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0


EPOCHS = 300
EVAL_EPOCHS = 1
EARLY_STOP_PATIENCE = 12


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHAR_EMBED_SIZE} {CONV} d{DROPOUT}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} tmp {INFONCE_TEMP}',
])
