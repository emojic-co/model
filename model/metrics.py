import torch


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
