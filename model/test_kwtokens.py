import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.kwtokens import STOPWORDS, keyword_category, query_tokens, word_count


def test_splits_and_lowercases():
    assert query_tokens("The Dog is Running FAST") == ["dog", "running", "fast"]


def test_strips_punctuation_and_short_tokens():
    assert query_tokens("pizza-time, y'all! ok?") == ["pizza", "time", "all", "ok"]


def test_drops_stopwords():
    assert "the" in STOPWORDS and "with" in STOPWORDS
    assert query_tokens("a cat and the hat") == ["cat", "hat"]


def test_digits_kept():
    assert query_tokens("bus 42 now") == ["bus", "42", "now"]


def test_word_count():
    assert word_count("cake") == 1
    assert word_count("thumbs up") == 2
    assert word_count("face palm emoji") == 3


def test_keyword_category():
    assert keyword_category(1) == "single"
    assert keyword_category(2) == "double"
    assert keyword_category(3) == "multi"
    assert keyword_category(0) == "single"


if __name__ == "__main__":
    test_splits_and_lowercases()
    test_strips_punctuation_and_short_tokens()
    test_drops_stopwords()
    test_digits_kept()
    test_word_count()
    test_keyword_category()
    print("ok")
