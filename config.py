# DATA
import datetime

MAX_TEXT_LEN = 42

# MODEL
CHAR_EMBED_SIZE = 16
# (kernel, channels)
CONV = [
    (3, 64),
    (2, 128),
]
POOL_1D_SIZE = 2
DROPOUT = 0.3

EMOJI_EMBED_SIZE = 32
TRIPLET_MARGIN = .5
EMOJI_NEGATIVES = 5

# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0


# EPOCHS
EPOCHS = 20
EVAL_EPOCHS = 1


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHAR_EMBED_SIZE} {CONV} d{DROPOUT}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP}',
])
