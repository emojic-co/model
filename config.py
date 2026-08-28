import json

# DATA
MAX_TEXT_LEN = 64
TEST_LEN = 200

# MODEL
EMBED_SIZE = 16
H_SIZE = 32
NUM_LAYERS = 1

# TRAINING
LR = 0.005
BATCH_SIZE = 64
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-6
EPOCHS = 100
EVAL_EPOCHS = 10

# COLORS
# Fixed per-feeling Oklab palette for the web app, loaded from palette.json
# (data, not code): each feeling maps to {bg1, bg2, text_color}, where bg1/bg2
# are the gradient background and text_color the foreground. Not learned, just
# baked into docs/meta.json. Values are [L 0..1, a -0.4..0.4, b -0.4..0.4].
# Unknown feelings fall back to Neutral.
with open("palette.json", encoding="utf-8") as f:
    FEELING_PALETTE = json.load(f)


def feeling_colors(name: str) -> dict:
    """Map a feeling to its web colors: {bg1, bg2, text_color} Oklab triples."""
    colors = FEELING_PALETTE.get(name, FEELING_PALETTE["Neutral"])
    return {k: list(v) for k, v in colors.items()}
