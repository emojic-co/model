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
    GEN_LOSS_COND_COLOR_CRITIC_TRAIN = "gan/gen/loss/cond_color_critic/train"
    GEN_LOSS_COND_COLOR_CRITIC_VAL = "gan/gen/loss/cond_color_critic/val"
    GEN_LOSS_COLOR_CRITIC_TRAIN = "gan/gen/loss/color_critic/train"
    GEN_LOSS_COLOR_CRITIC_VAL = "gan/gen/loss/color_critic/val"
    CRITIC_LOSS = "gan/critic/loss"
    CRITIC_AUROC_TRAIN = "gan/critic/auroc/train"
    CRITIC_AUROC_VAL = "gan/critic/auroc/val"
    CRITIC_MEAN_SCORE_REAL = "gan/critic/mean_score/real"
    CRITIC_MEAN_SCORE_FAKE = "gan/critic/mean_score/fake"
    COLOR_CRITIC_LOSS = "gan/color_critic/loss"
    COLOR_CRITIC_AUROC_TRAIN = "gan/color_critic/auroc/train"
