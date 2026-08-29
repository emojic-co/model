# DATA
import datetime

MAX_TEXT_LEN = 42
TEST_LEN = 900

# MODEL
CHAR_EMBED_SIZE = 16
CHANNELS = (20, 30, 40)

# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
EMOJI_LABEL_SMOOTHING = 0.2


# EPOCHS
EPOCHS = 500
EVAL_EPOCHS = 2


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHANNELS}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} wd {WEIGHT_DECAY}',
])
