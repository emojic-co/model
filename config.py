# DATA
import datetime

MAX_TEXT_LEN = 42
TEST_LEN = 900

# MODEL
CHAR_EMBED_SIZE = 16
KERNELS = (3, 4)
CHANNELS = (120, 40)
EMOJI_EMBED_SIZE = 32

# Emoji vectors (q and the embedding rows) are L2-normalized in Model.forward,
# so the triplet L2 distance is bounded in [0, 2]; margin stays well inside it.
TRIPLET_MARGIN = 0.5
# Wrong emoji classes sampled per row for the triplet loss (the anchor/positive
# pair is scored against each, then averaged).
EMOJI_NEGATIVES = 5

# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4


# EPOCHS
EPOCHS = 20
EVAL_EPOCHS = 2


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'DATA: mtl {MAX_TEXT_LEN} tl {TEST_LEN}',
    f'MODEL: cs {CHANNELS}',   # noqa: E501
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP} wd {WEIGHT_DECAY}',
])
