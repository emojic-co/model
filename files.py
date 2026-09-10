import os

MODEL_DIR = "model"
TOOLS_DIR = "tools"
DATA_DIR = "data"
PT_DIR = "pt"
GOAL_DIR = "goal"

STEP1 = bool(os.environ.get("EMOJIC_STEP1"))
STEP1_DIR = f"{DATA_DIR}/step1"
_DD = STEP1_DIR if STEP1 else DATA_DIR

DATA_JSONL = f"{DATA_DIR}/data.jsonl"
TRAIN_JSONL = f"{_DD}/train.jsonl"
EVAL_JSONL = f"{_DD}/eval.jsonl"
LABELS_JSON = f"{_DD}/labels.json"
II_JSON = f"{_DD}/ii.json"
CLDR_JSONL = f"{DATA_DIR}/cldr.jsonl"
CLDR_BASELINE_JSON = f"{DATA_DIR}/cldr-baseline.json"
EMOJI_POPULARITY_JSON = f"{DATA_DIR}/emoji_popularity.json"
COLORS_JSONL = f"{DATA_DIR}/colors.jsonl"
ENERGY_KEYWORDS_TXT = f"{DATA_DIR}/energy_keywords.txt"
PRED_JSONL = f"{DATA_DIR}/pred.jsonl"

STEP1_EXACT_EVAL_JSONL = f"{STEP1_DIR}/exact_eval.jsonl"
STEP1_FUZZY_EVAL_JSONL = f"{STEP1_DIR}/fuzzy_eval.jsonl"

ENC_PT = f"{PT_DIR}/enc.pt"
STYLE_PT = f"{PT_DIR}/style.pt"
EMOJI_PT = f"{PT_DIR}/emoji.pt"
EMOJI_EMBED_PT = f"{PT_DIR}/emoji_embed.pt"
GEN_PT = f"{PT_DIR}/gen.pt"
CRITIC_PT = f"{PT_DIR}/critic.pt"
FUSION_GATE_PT = f"{PT_DIR}/fusion_gate.pt"
FUSION_GAIN_PT = f"{PT_DIR}/fusion_gain.pt"
FUSION_MIX_PT = f"{PT_DIR}/fusion_mix.pt"

WEB_PUBLIC_DIR = "web/public"
KWPROJ_JSON = f"{WEB_PUBLIC_DIR}/kwproj.json"
REPORT_DIR = "report"
RUNS_DIR = "runs"
