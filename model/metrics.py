import torch


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot.clamp(min=1e-12)).mean()


def macro_average(
    per_row_values: torch.Tensor,
    target: torch.Tensor,
    min_support: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    support = target.sum(dim=0)
    class_sum = target.t() @ per_row_values
    eligible = support >= min_support
    if not bool(eligible.any()):
        return per_row_values.new_zeros(()), support, 0
    class_mean = class_sum[eligible] / support[eligible]
    return class_mean.mean(), support, int(eligible.sum())
