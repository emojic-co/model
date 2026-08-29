# DATA
import datetime

MAX_TEXT_LEN = 42
TEST_LEN = 600

# MODEL
KERNEL_1 = 3
CHANNELS = 36
HIDDEN = 16
EMOJI_EMBED_SIZE = 64
TRIPLET_MARGIN = 1.0

# TRAINING
LR = 0.2
BATCH_SIZE = 128
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-5


# EPOCHS
EPOCHS = 300
EVAL_EPOCHS = 2


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'DATA: mtl {MAX_TEXT_LEN} tl {TEST_LEN}',
    f'MODEL: cs1 {CHANNELS} cs2 {HIDDEN} ee {EMOJI_EMBED_SIZE} tm {TRIPLET_MARGIN}',   # noqa: E501
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} wd {WEIGHT_DECAY}',
])
