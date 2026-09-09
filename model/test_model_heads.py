import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.config import EMOJIS, TEXT_EMBED_SIZE
from model.data import FLEX_N
from model.model import EmojiEmbedding, EmojiHead, FusionHead, KWHead

B = 4


def test_shapes():
    emb = EmojiEmbedding()
    q_txt = EmojiHead()(torch.randn(B, TEXT_EMBED_SIZE))
    q_kw = KWHead()(torch.randn(B, FLEX_N))
    assert q_txt.shape == (B, 64)
    assert q_kw.shape == (B, 64)
    assert emb.score(q_txt).shape == (B, len(EMOJIS))


def test_kwhead_zero_init():
    q_kw = KWHead()(torch.randn(B, FLEX_N))
    assert torch.allclose(q_kw, torch.zeros_like(q_kw))


def test_gate_starts_near_one():
    a = FusionHead()(torch.randn(B, TEXT_EMBED_SIZE), torch.randn(B, FLEX_N))
    assert a.shape == (B,)
    assert (a > 0.97).all() and (a < 1.0).all()


def test_untrained_fusion_matches_emojihead():
    torch.manual_seed(0)
    emb = EmojiEmbedding()
    eh = EmojiHead().eval()
    kw = KWHead().eval()
    gate = FusionHead().eval()
    x = torch.randn(B, TEXT_EMBED_SIZE)
    tf = torch.randn(B, FLEX_N)
    q_txt = eh(x)
    q_kw = kw(tf)
    a = gate(x, tf).unsqueeze(-1)
    fused = emb.score(a * q_txt + (1 - a) * q_kw)
    base = emb.score(q_txt)
    assert (fused - base).abs().max() < 0.05 * base.abs().max()


if __name__ == "__main__":
    test_shapes()
    test_kwhead_zero_init()
    test_gate_starts_near_one()
    test_untrained_fusion_matches_emojihead()
    print("ok")
