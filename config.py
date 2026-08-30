import datetime

MAX_TEXT_LEN = 42

CHAR_EMBED_SIZE = 16
CONV = [
    (3, 96),
    (2, 128),
]
POOL_1D_SIZE = 2
DROPOUT = 0.3

EMOJI_EMBED_SIZE = 40
TRIPLET_MARGIN = .6
EMOJI_NEGATIVES = 6

LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0


EPOCHS = 100
EVAL_EPOCHS = 1
EARLY_STOP_PATIENCE = 12


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHAR_EMBED_SIZE} {CONV} d{DROPOUT}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP}',
])
