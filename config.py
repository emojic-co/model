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
# Fixed per-feeling Oklab palette for the web app: bg1/bg2 are the gradient
# background, text_color the foreground. Not learned, just baked into
# docs/meta.json. Values are [L 0..1, a -0.4..0.4, b -0.4..0.4]. Unknown
# feelings fall back to Neutral.
FEELING_PALETTE = {
    "Happy": ([0.90, 0.02, 0.13], [0.82, 0.06, 0.16], [0.22, 0.03, 0.06]),
    "Excited": ([0.80, 0.12, 0.10], [0.70, 0.16, 0.14], [0.97, 0.0, 0.02]),
    "Calm": ([0.88, -0.11, 0.05], [0.80, -0.13, 0.04], [0.28, -0.04, 0.02]),
    "Sad": ([0.55, -0.02, -0.09], [0.45, -0.02, -0.11], [0.95, -0.01, -0.02]),
    "Angry": ([0.48, 0.18, 0.09], [0.38, 0.16, 0.07], [0.97, 0.02, 0.01]),
    "Anxious": ([0.60, -0.03, -0.06], [0.50, 0.02, -0.04], [0.95, 0.0, -0.01]),
    "Neutral": ([0.88, -0.05, -0.04], [0.80, -0.06, -0.08], [0.30, -0.004, -0.016]),
    "Love": ([0.86, 0.10, 0.02], [0.78, 0.13, 0.03], [0.25, 0.06, 0.01]),
}


def feeling_colors(name: str) -> dict:
    """Map a feeling to its web colors: {bg1, bg2, text_color} Oklab triples."""
    bg1, bg2, text_color = FEELING_PALETTE.get(
        name, FEELING_PALETTE["Neutral"])
    return {"bg1": list(bg1), "bg2": list(bg2), "text_color": list(text_color)}
