import torch

COLOR_SHIFT = 127.5

_LIN_TO_LMS = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_LAB = torch.tensor([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(
        c <= 0.04045, c / 12.92, ((c.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    shape = rgb.shape
    c = ((rgb + COLOR_SHIFT) / 255.0).clamp(0.0, 1.0).reshape(*shape[:-1], -1, 3)
    lms = _srgb_to_linear(c) @ _LIN_TO_LMS.to(c).t()
    lms_ = lms.sign() * lms.abs().clamp(min=1e-12) ** (1 / 3)
    return (lms_ @ _LMS_TO_LAB.to(c).t()).reshape(shape)


_LMS_TO_LAB_INV = torch.linalg.inv(_LMS_TO_LAB)
_LIN_TO_LMS_INV = torch.linalg.inv(_LIN_TO_LMS)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(min=0.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def oklab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Inverse of `rgb_to_oklab`: oklab -> shifted RGB (same [-SHIFT, SHIFT] convention)."""
    shape = lab.shape
    lab = lab.reshape(*shape[:-1], -1, 3)
    lms_ = lab @ _LMS_TO_LAB_INV.to(lab).t()
    lms = lms_.sign() * lms_.abs() ** 3
    c = _linear_to_srgb(lms @ _LIN_TO_LMS_INV.to(lab).t())
    return (c.clamp(0.0, 1.0) * 255.0 - COLOR_SHIFT).reshape(shape)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()
