# DATA
import datetime

MAX_TEXT_LEN = 42
TEST_LEN = 500

# MODEL
KERNEL_1 = 3
CHANNELS_1 = 16
KERNEL_2 = 7
CHANNELS_2 = 32
EMOJI_EMBED_SIZE = 8
NEGATIVE_SAMPLES = 16

# TRAINING
LR = 0.005
BATCH_SIZE = 128
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-5


# EPOCHS
EPOCHS = 400
EVAL_EPOCHS = 10


CONFIG_NAME = ' | '.join([
    f'DATA: mtl {MAX_TEXT_LEN} tl {TEST_LEN}',
    f'MODEL: cs1 {CHANNELS_1} cs2 {CHANNELS_2} ee {EMOJI_EMBED_SIZE} ns {NEGATIVE_SAMPLES}',   # noqa: E501
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} wd {WEIGHT_DECAY}',
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
])


assert KERNEL_1 % 2 == 1, "KERNEL_1 must be odd for 'same' padding"
assert KERNEL_2 % 2 == 1, "KERNEL_2 must be odd for 'same' padding"
