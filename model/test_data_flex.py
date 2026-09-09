import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.data import (
    EMOJIS,
    FLEX_MAX_K,
    FLEX_RAW_DIM,
    FLEXQ_DIM,
    _row_flex,
    scatter_flex,
)

V = len(EMOJIS)


def test_row_flex_builds_model_order_vector():
    e0, e1 = EMOJIS[0], EMOJIS[1]
    row = {
        "flexsearch": [
            [e0, 5.0, 1.0, 1, 0, 7.0, 9, 5, 5, 5],
            [e1, 2.5, 0.5, 0, 1, 8.5, 17, 6, 5, 5],
        ],
        "flexq": {"tokens": 3, "matched": 2, "sum": 7.5, "max": 5.0, "cand": 2},
    }
    idx, raw, q = _row_flex(row)
    assert idx.shape == (FLEX_MAX_K,) and raw.shape == (FLEX_MAX_K, FLEX_RAW_DIM)
    assert q.tolist() == [3.0, 2.0, 7.5, 5.0, 2.0]
    assert idx[0].item() == 0 and idx[1].item() == 1 and idx[2].item() == -1
    assert raw[0].tolist() == [5.0, 1.0, 1.0, 0.0, 7.0, 1.0, 9.0, 5.0, 5.0, 5.0]
    assert abs(raw[1][5].item() - 0.5) < 1e-6


def test_row_flex_missing_fields():
    idx, raw, q = _row_flex({})
    assert idx.tolist() == [-1] * FLEX_MAX_K
    assert raw.abs().sum().item() == 0.0
    assert q.tolist() == [0.0] * FLEXQ_DIM


def test_scatter_flex_places_rows_by_index():
    idx = torch.full((2, FLEX_MAX_K), -1, dtype=torch.long)
    idx[0, 0] = 3
    idx[1, 0] = 5
    idx[1, 1] = 3
    raw = torch.zeros(2, FLEX_MAX_K, FLEX_RAW_DIM)
    raw[0, 0, 0] = 9.0
    raw[1, 0, 0] = 4.0
    raw[1, 1, 0] = 2.0
    dense = scatter_flex(idx, raw)
    assert dense.shape == (2, V, FLEX_RAW_DIM)
    assert dense[0, 3, 0].item() == 9.0
    assert dense[1, 5, 0].item() == 4.0
    assert dense[1, 3, 0].item() == 2.0
    assert dense[0, 0, 0].item() == 0.0


if __name__ == "__main__":
    test_row_flex_builds_model_order_vector()
    test_row_flex_missing_fields()
    test_scatter_flex_places_rows_by_index()
    print("ok")
