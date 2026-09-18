from enum import ReprEnum, StrEnum


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


class NamedMetric(str):
    __slots__ = ()


def named_metric(
    source: Source, metric: Metric, split: Split | None = None
) -> NamedMetric:
    parts = [source.value, metric.value]
    if split is not None:
        parts.append(split.value)
    return NamedMetric("/".join(parts))


class GanMetric(NamedMetric, ReprEnum):
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    GEN_LOSS = "gan/gen/loss"
    CRITIC_LOSS = "gan/critic/loss"
