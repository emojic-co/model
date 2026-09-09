import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.data import FLEX_N, KW_VOCAB, TRAIN_PATH, EmojiDataset, _row_tf, read


def test_flex_n_matches_vocab():
    assert FLEX_N == len(KW_VOCAB)
    assert FLEX_N > 500


def test_row_tf_scatters_sparse_pairs():
    t = _row_tf({"flex_tf": [[3, 1.0], [7, 0.667]]})
    assert t.shape == (FLEX_N,)
    assert t.dtype == torch.float32
    assert t[3].item() == 1.0
    assert abs(t[7].item() - 0.667) < 1e-6
    assert abs(t.sum().item() - 1.667) < 1e-6


def test_row_tf_empty():
    assert _row_tf({"flex_tf": []}).sum().item() == 0.0
    assert _row_tf({}).sum().item() == 0.0


def test_dataset_batch_shape():
    records = []
    for r in read(TRAIN_PATH):
        records.append(r)
        if len(records) == 64:
            break
    ds = EmojiDataset(records)
    text, emoji, style, colors, flex_tf = ds[0]
    assert flex_tf.shape == (FLEX_N,)
    assert ds.flex_tf.shape == (len(records), FLEX_N)


if __name__ == "__main__":
    test_flex_n_matches_vocab()
    test_row_tf_scatters_sparse_pairs()
    test_row_tf_empty()
    test_dataset_batch_shape()
    print("ok")
