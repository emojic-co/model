import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.kwtokens import STOPWORDS, query_tokens


def test_splits_and_lowercases():
    assert query_tokens("The Dog is Running FAST") == ["dog", "running", "fast"]


def test_strips_punctuation_and_short_tokens():
    assert query_tokens("pizza-time, y'all! ok?") == ["pizza", "time", "all", "ok"]


def test_drops_stopwords():
    assert "the" in STOPWORDS and "with" in STOPWORDS
    assert query_tokens("a cat and the hat") == ["cat", "hat"]


def test_digits_kept():
    assert query_tokens("bus 42 now") == ["bus", "42", "now"]


if __name__ == "__main__":
    test_splits_and_lowercases()
    test_strips_punctuation_and_short_tokens()
    test_drops_stopwords()
    test_digits_kept()
    print("ok")
