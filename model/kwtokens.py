import re

STOPWORDS = set(
    (
        "a an the to of in on at is it its i you we they he she this that for and or but"
        " not with my your me am are was were be been being do does did have has had"
        " will would can could just so if"
    ).split()
)
_NON = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def query_tokens(text: str) -> list[str]:
    t = _NON.sub(" ", text.lower())
    return [w for w in _WS.split(t) if len(w) >= 2 and w not in STOPWORDS]


KEYWORD_CATEGORIES = ("single", "double", "multi")


def word_count(text: str) -> int:
    return len(text.split())


def keyword_category(count: int) -> str:
    if count <= 1:
        return "single"
    if count == 2:
        return "double"
    return "multi"
