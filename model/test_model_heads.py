import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn.functional import softplus

from model.config import EMOJIS
from model.model import FusionHeadGain, FusionHeadGate, FusionHeadMix, _z

B = 4
N = len(EMOJIS)


def _inputs(requires_grad=False):
    torch.manual_seed(0)
    logit_m = torch.randn(B, N, requires_grad=requires_grad)
    kw = torch.zeros(B, N)
    kw[:, 3] = 0.7
    kw[:, 9] = 0.4
    return logit_m, kw


def test_shapes():
    logit_m, kw = _inputs()
    for head in (FusionHeadGate(), FusionHeadGain(), FusionHeadMix()):
        assert head(logit_m, kw).shape == (B, N)


def test_gain_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadGain().eval()
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), logit_m)
    assert torch.isclose(softplus(head.raw_beta), softplus(torch.zeros(())))


def test_mix_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadMix().eval()
    g = torch.sigmoid(torch.tensor(-2.0))
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), (1 - g) * _z(logit_m), atol=1e-5)


def test_gate_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadGate().eval()
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), 0.5 * _z(logit_m), atol=1e-4)
    assert torch.isclose(head.last_a, torch.tensor(0.5), atol=1e-4)


def test_kw_zero_is_noop_per_emoji():
    logit_m, kw = _inputs()
    for head in (FusionHeadGain().eval(), FusionHeadMix().eval(), FusionHeadGate().eval()):
        full = head(logit_m, kw)
        none = head(logit_m, torch.zeros(B, N))
        untouched = (kw == 0).all(dim=0)
        assert torch.allclose(full[:, untouched], none[:, untouched], atol=1e-4)


def test_heads_are_transparent_to_logit_grad_and_train_their_params():
    logit_m, kw = _inputs(requires_grad=True)
    for head in (FusionHeadGate(), FusionHeadGain(), FusionHeadMix()):
        lm = logit_m.detach().clone().requires_grad_(True)
        out = head(lm, kw)
        out.sum().backward()
        assert lm.grad is not None and lm.grad.abs().sum() > 0
        pg = [p.grad for p in head.parameters()]
        assert pg and all(g is not None for g in pg)


if __name__ == "__main__":
    test_shapes()
    test_gain_neutral_at_init()
    test_mix_neutral_at_init()
    test_gate_neutral_at_init()
    test_kw_zero_is_noop_per_emoji()
    test_heads_are_transparent_to_logit_grad_and_train_their_params()
    print("ok")
