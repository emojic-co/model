import torch


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot.clamp(min=1e-12)).mean()
