import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.config import EMOJIS
from model.model import FusionHead

B = 4
N = len(EMOJIS)


def _inputs(requires_grad=False):
    torch.manual_seed(0)
    logit_m = torch.randn(B, N, requires_grad=requires_grad)
    kw = torch.zeros(B, N)
    kw[:, 3] = 0.7
    kw[:, 9] = 0.4
    return logit_m, kw


def test_shape():
    logit_m, kw = _inputs()
    assert FusionHead()(logit_m, kw).shape == (B, N)


def test_init_is_logits_plus_kw():
    logit_m, kw = _inputs()
    head = FusionHead().eval()
    assert torch.allclose(head(logit_m, kw), logit_m + kw, atol=1e-5)


def test_per_emoji_weights_and_bias():
    logit_m, kw = _inputs()
    head = FusionHead().eval()
    with torch.no_grad():
        head.w_dl.copy_(torch.linspace(0.1, 2.0, N))
        head.w_search.copy_(torch.linspace(2.0, 0.1, N))
        head.b.copy_(torch.linspace(-1.0, 1.0, N))
    expect = head.w_dl * logit_m + head.w_search * kw + head.b
    assert torch.allclose(head(logit_m, kw), expect, atol=1e-6)


def test_transparent_to_logit_grad_and_trains_params():
    logit_m, kw = _inputs(requires_grad=True)
    head = FusionHead()
    lm = logit_m.detach().clone().requires_grad_(True)
    out = head(lm, kw)
    out.sum().backward()
    assert lm.grad is not None and lm.grad.abs().sum() > 0
    pg = [p.grad for p in head.parameters()]
    assert pg and all(g is not None for g in pg)
    assert head.w_search.grad.abs().sum() > 0


if __name__ == "__main__":
    test_shape()
    test_init_is_logits_plus_kw()
    test_per_emoji_weights_and_bias()
    test_transparent_to_logit_grad_and_trains_params()
    print("ok")
