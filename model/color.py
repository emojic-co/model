import torch


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()
