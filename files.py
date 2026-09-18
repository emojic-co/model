from enum import StrEnum
from pathlib import Path

MODEL_DIR = Path("model")
TOOLS_DIR = Path("tools")
DATA_DIR = Path("data")
PT_DIR = Path("pt")

GOALS_YML = Path("goals.yml")

DATA_JSONL = DATA_DIR / "data.jsonl"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
EVAL_JSONL = DATA_DIR / "eval.jsonl"
LABELS_JSON = DATA_DIR / "labels.json"
II_JSON = DATA_DIR / "ii.json"
CLDR_JSONL = DATA_DIR / "cldr.jsonl"
EMOJILIB_JSONL = DATA_DIR / "emojilib.jsonl"
WA_KEYWORDS_JSON = DATA_DIR / "wa-keywords.json"
KEYWORDS_JSONL = DATA_DIR / "keywords.jsonl"
TERMS_JSONL = DATA_DIR / "terms.jsonl"
FLAGS_JSONL = DATA_DIR / "flags.jsonl"
CLDR_BASELINE_JSON = DATA_DIR / "cldr-baseline.json"
EMOJI_POPULARITY_JSON = DATA_DIR / "emoji_popularity.json"
GROUP_JSON = DATA_DIR / "group.json"
COLORS_JSONL = DATA_DIR / "colors.jsonl"
COLOR_TERMS_JSONL = DATA_DIR / "color_terms.jsonl"
COLOR_NAMES_SOURCE_JSON = DATA_DIR / "color-names-source.json"
PRED_JSONL = DATA_DIR / "pred.jsonl"


class PtFile(StrEnum):
    ENC = "enc.pt"
    LANG = "lang.pt"
    STYLE = "style.pt"
    EMOJI = "emoji.pt"
    EMOJI_EMBED = "emoji_embed.pt"
    GEN = "gen.pt"

    def in_dir(self, pt_dir: Path) -> Path:
        return pt_dir / self.value


ENC_PT = PtFile.ENC.in_dir(PT_DIR)
LANG_PT = PtFile.LANG.in_dir(PT_DIR)
STYLE_PT = PtFile.STYLE.in_dir(PT_DIR)
EMOJI_PT = PtFile.EMOJI.in_dir(PT_DIR)
EMOJI_EMBED_PT = PtFile.EMOJI_EMBED.in_dir(PT_DIR)
GEN_PT = PtFile.GEN.in_dir(PT_DIR)

WEB_PUBLIC_DIR = Path("web/public")
REPORT_DIR = Path("report")
RUNS_DIR = Path("runs")
