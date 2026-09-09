import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.data import EMOJIS
from model.model import FusionHead, fusion_features

V = len(EMOJIS)


def _inputs(b=4):
    logit = torch.randn(b, V)
    raw = torch.zeros(b, V, 10)
    raw[:, :5, 0] = torch.rand(b, 5) + 0.1
    raw[:, :5, 1] = torch.rand(b, 5)
    raw[:, :5, 2] = 1.0
    raw[:, :5, 5] = torch.tensor([1.0, 0.5, 1 / 3, 0.25, 0.2])
    raw[:, :5, 6:] = torch.randint(0, 40, (b, 5, 4)).float()
    q = torch.tensor([[3.0, 2.0, 12.0, 6.0, 8.0]]).repeat(b, 1)
    return logit, raw, q


def test_feature_shapes():
    logit, raw, q = _inputs()
    scal, idx = fusion_features(logit, raw, q)
    assert scal.shape == (4, V, 18)
    assert idx.shape == (4, V, 4)
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() <= 16


def test_zero_init_is_identity():
    torch.manual_seed(0)
    head = FusionHead()
    with torch.no_grad():
        head.net[-1].weight.zero_()
        head.net[-1].bias.zero_()
    logit, raw, q = _inputs()
    out = head(logit, raw, q)
    assert torch.allclose(out, logit, atol=1e-5)


def test_forward_shape_and_grad():
    head = FusionHead()
    logit, raw, q = _inputs()
    out = head(logit, raw, q)
    assert out.shape == (4, V)
    out.sum().backward()
    assert head.net[0].weight.grad is not None


if __name__ == "__main__":
    test_feature_shapes()
    test_zero_init_is_identity()
    test_forward_shape_and_grad()
    print("ok")
