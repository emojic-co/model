# DATA
import datetime

MAX_TEXT_LEN = 64
TEST_LEN = 500

# MODEL
EMBED_SIZE = 16
H_SIZE = 32
NUM_LAYERS = 1
EMOJI_EMBED_SIZE = 16
NEGATIVE_SAMPLES = 16

# TRAINING
LR = 0.005
BATCH_SIZE = 64
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-6


# EPOCHS
EPOCHS = 200
EVAL_EPOCHS = 10


CONFIG_NAME = ' | '.join([
    f'DATA: mtl {MAX_TEXT_LEN} tl {TEST_LEN}',
    f'MODEL: es {EMBED_SIZE}  hs {H_SIZE} nl {NUM_LAYERS} ee {EMOJI_EMBED_SIZE} ns {NEGATIVE_SAMPLES}',   # noqa: E501
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} wd {WEIGHT_DECAY}',
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
])
