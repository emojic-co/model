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
