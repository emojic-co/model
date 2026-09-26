from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"


class LogStage(StrEnum):
    ENC = "enc"
    GAN = "gan"
    COND = "cond"


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
    AUROC_SHUF = "auroc_shuf"
    AUROC_RAND = "auroc_rand"
    LOSS = "loss"


def named_metric(
    stage: LogStage, source: Source, metric: Metric, split: Split | None = None
) -> str:
    parts = [stage.value, f"{source.value}_{metric.value}"]
    if split is not None:
        parts.append(split.value)
    return "/".join(parts)


class GanMetric:
    ENERGY_TRAIN = "gan/energy/train"
    ENERGY_VAL = "gan/energy/val"
    ENERGY_KEYWORD_AVG = "gan/energy/keyword_avg"
    GEN_LOSS_COND = "gan/gen_loss"
    COND_LOSS = "gan/critic_loss"
    COND_AUROC_GEN = "gan/critic_gen_auroc"
    COND_MEAN_SCORE_REAL = "gan/cond_mean_score_real"
    COND_MEAN_SCORE_FAKE = "gan/cond_mean_score_fake"
