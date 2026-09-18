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
    GEN_LOSS = "gan/gen/loss"
    CRITIC_LOSS = "gan/critic/loss"
