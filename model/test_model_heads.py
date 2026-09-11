import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.config import EMOJIS, TEXT_EMBED_SIZE
from model.model import FusionHead

B = 4
N = len(EMOJIS)


def _inputs(requires_grad=False):
    torch.manual_seed(0)
    text_embed = torch.randn(B, TEXT_EMBED_SIZE, requires_grad=requires_grad)
    logit_m = torch.randn(B, N, requires_grad=requires_grad)
    kw = torch.zeros(B, N)
    kw[:, 3] = 0.7
    kw[:, 9] = 0.4
    return text_embed, logit_m, kw


def test_shape():
    text_embed, logit_m, kw = _inputs()
    assert FusionHead()(text_embed, logit_m, kw).shape == (B, N)


def test_gate_in_unit_interval():
    text_embed, _, _ = _inputs()
    a = FusionHead().eval().gate(text_embed)
    assert a.shape == (B,)
    assert torch.all(a > 0) and torch.all(a < 1)


def test_forward_is_gated_mix_of_sigmoid_logits_and_kw():
    text_embed, logit_m, kw = _inputs()
    head = FusionHead().eval()
    a = head.gate(text_embed).unsqueeze(-1)
    expect = a * torch.sigmoid(logit_m) + (1 - a) * kw
    assert torch.allclose(head(text_embed, logit_m, kw), expect, atol=1e-6)


def test_gate_near_one_favors_model():
    text_embed, logit_m, kw = _inputs()
    head = FusionHead().eval()
    with torch.no_grad():
        head.gate_proj.weight.zero_()
        head.gate_proj.bias.fill_(10.0)
    assert torch.allclose(head(text_embed, logit_m, kw), torch.sigmoid(logit_m), atol=1e-3)


def test_gate_near_zero_favors_search():
    text_embed, logit_m, kw = _inputs()
    head = FusionHead().eval()
    with torch.no_grad():
        head.gate_proj.weight.zero_()
        head.gate_proj.bias.fill_(-10.0)
    assert torch.allclose(head(text_embed, logit_m, kw), kw, atol=1e-3)


def test_gradients_flow_to_text_embed_and_logits_and_train_params():
    text_embed, logit_m, kw = _inputs(requires_grad=True)
    head = FusionHead()
    out = head(text_embed, logit_m, kw)
    out.sum().backward()
    assert text_embed.grad is not None and text_embed.grad.abs().sum() > 0
    assert logit_m.grad is not None and logit_m.grad.abs().sum() > 0
    pg = [p.grad for p in head.parameters()]
    assert pg and all(g is not None for g in pg)


if __name__ == "__main__":
    test_shape()
    test_gate_in_unit_interval()
    test_forward_is_gated_mix_of_sigmoid_logits_and_kw()
    test_gate_near_one_favors_model()
    test_gate_near_zero_favors_search()
    test_gradients_flow_to_text_embed_and_logits_and_train_params()
    print("ok")
