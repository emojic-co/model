from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"


class Source(StrEnum):
    KEYWORD = "keyword"
    TERM = "term"
    COLOR = "color"
    FULL_TEXT = "full_text"
    EMOJI = "emoji"
    STYLE = "style"


class Metric(StrEnum):
    ACC_1 = "acc@1"
    RATE = "rate"
    MAE = "mae"
    MRR = "mrr"
    R2 = "r2"
    AUROC = "auroc"
    LOSS = "loss"
    AVG_LOGITS_POS = "avg_logits/pos"
    AVG_LOGITS_NEG = "avg_logits/neg"


def named_metric(
    source: Source, metric: Metric, split: Split | None = None
) -> str:
    parts = [source.value, metric.value]
    if split is not None:
        parts.append(split.value)
    return "/".join(parts)


class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS_COND_COLOR_CRITIC = "gan/gen/loss/cond_color_critic"
    GEN_LOSS_COLOR_CRITIC = "gan/gen/loss/color_critic"
    COND_COLOR_CRITIC_LOSS = "gan/cond_color_critic/loss"
    COND_COLOR_CRITIC_AUROC = "gan/cond_color_critic/auroc"
    COND_COLOR_CRITIC_MEAN_SCORE_REAL = "gan/cond_color_critic/mean_score/real"
    COND_COLOR_CRITIC_MEAN_SCORE_FAKE = "gan/cond_color_critic/mean_score/fake"
    COND_COLOR_CRITIC_MEAN_SCORE_WRONG = "gan/cond_color_critic/mean_score/wrong"
    COLOR_CRITIC_LOSS = "gan/color_critic/loss"
    COLOR_CRITIC_AUROC = "gan/color_critic/auroc"
