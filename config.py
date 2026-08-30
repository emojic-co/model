# DATA
import datetime

MAX_TEXT_LEN = 42
# The eval holdout is the fixed, curated eval.jsonl (see gen_eval.ts / data.py),
# not a slice of data.jsonl -- there is no eval-size knob here any more.

# MODEL
CHAR_EMBED_SIZE = 16

# Model configuration: one entry per conv layer, giving that layer's output
# channel count. Every layer is a bigram block -- Conv1d(kernel_size=2,
# stride=1, padding=0) -> BatchNorm1d -> LeakyReLU -> MaxPool1d(2, 2) -- so the
# time axis loses one step to the conv and is then halved by the pool. Stacking
# kernel-2 layers grows the receptive field geometrically (~2, ~4, ~8, ~16
# chars). With MAX_TEXT_LEN = 42 the length runs 42 -> 20 -> 9 -> 4 -> 1 over
# four layers; more than ~5 layers would drive it to 0.
CHANNELS = [64, 128]

EMOJI_EMBED_SIZE = 20
TRIPLET_MARGIN = 0.5
EMOJI_NEGATIVES = 5

# TRAINING
LR = 0.01
BATCH_SIZE = 128
GRAD_CLIP = 1.0


# EPOCHS
EPOCHS = 30
EVAL_EPOCHS = 2


CONFIG_NAME = ' | '.join([
    f'TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'MODEL: {CHAR_EMBED_SIZE} {CHANNELS}',
    f'TRAIN: lr {LR} bs {BATCH_SIZE} gc {GRAD_CLIP}',
])
