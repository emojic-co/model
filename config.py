# DATA
import datetime

MAX_TEXT_LEN = 42

# MODEL
KERNEL = 3
CHAR_EMBED_SIZE = 16
CHANNELS = [96, 96]

EMOJI_EMBED_SIZE = 16
TRIPLET_MARGIN = .5
EMOJI_NEGATIVES = 5

# TRAINING
LR = 0.02
BATCH_SIZE = 256
GRAD_CLIP = 2.0


# EPOCHS
EPOCHS = 30
EVAL_EPOCHS = 1


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {KERNEL} {CHAR_EMBED_SIZE} {CHANNELS}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP}',
])
