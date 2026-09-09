import json
import re
from pathlib import Path

FUZZY_MIN_LEN = 4
FUZZY_MAX_LEN_DELTA = 3
FUZZY_WEIGHT = 0.6
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


def fuzzy_match(a: str, b: str) -> bool:
    if len(a) < FUZZY_MIN_LEN or len(b) < FUZZY_MIN_LEN:
        return False
    if abs(len(a) - len(b)) > FUZZY_MAX_LEN_DELTA:
        return False
    return a.startswith(b) or b.startswith(a)


def _r3(x: float) -> float:
    return float(f"{x:.3f}")


class FlexRanker:
    def __init__(self, flex_json_path):
        j = json.loads(Path(flex_json_path).read_text(encoding="utf-8"))
        self.emojis = j["emojis"]
        self.keywords = j["keywords"]
        self._idf = j["idf"]
        self._idf_default = j["idf_default"]
        self.kw_count = [len(k) for k in self.keywords]
        self.kw_to_glyphs: dict[str, set[int]] = {}
        self.prefix4: dict[str, set[str]] = {}
        self._global_kw: set[str] = set()
        for gi, toks in enumerate(self.keywords):
            for kw in toks:
                for w in kw.split():
                    self._global_kw.add(w)
                self.kw_to_glyphs.setdefault(kw, set()).add(gi)
                if len(kw) >= FUZZY_MIN_LEN:
                    self.prefix4.setdefault(kw[:FUZZY_MIN_LEN], set()).add(kw)

    def idf(self, w: str) -> float:
        return self._idf.get(w, self._idf_default)

    def _score(self, text: str):
        q = query_tokens(text)
        acc: dict[int, dict] = {}

        def bump(gi, s, tok_idf, is_exact, word_len, kw_len, overlap):
            e = acc.get(gi)
            if e is None:
                e = {
                    "score": 0.0,
                    "exact": 0,
                    "fuzzy": 0,
                    "best_idf": -1.0,
                    "best_exact": False,
                    "kw_len": 0,
                    "word_len": 0,
                    "overlap": 0,
                }
                acc[gi] = e
            e["score"] += s
            if is_exact:
                e["exact"] += 1
            else:
                e["fuzzy"] += 1
            win = (
                tok_idf > e["best_idf"]
                or (tok_idf == e["best_idf"] and is_exact and not e["best_exact"])
                or (
                    tok_idf == e["best_idf"]
                    and is_exact == e["best_exact"]
                    and overlap > e["overlap"]
                )
            )
            if win:
                e["best_idf"] = tok_idf
                e["best_exact"] = is_exact
                e["kw_len"] = kw_len
                e["word_len"] = word_len
                e["overlap"] = overlap

        for w in q:
            w_idf = self.idf(w)
            exact = self.kw_to_glyphs.get(w)
            if exact:
                for gi in exact:
                    bump(gi, w_idf, w_idf, True, len(w), len(w), len(w))
            if len(w) >= FUZZY_MIN_LEN:
                bucket = self.prefix4.get(w[:FUZZY_MIN_LEN])
                if bucket:
                    fuzz: dict[int, tuple[str, int]] = {}
                    for kw in bucket:
                        if kw == w or not fuzzy_match(w, kw):
                            continue
                        ov = min(len(w), len(kw))
                        for gi in self.kw_to_glyphs[kw]:
                            cur = fuzz.get(gi)
                            if cur is None or ov > cur[1]:
                                fuzz[gi] = (kw, ov)
                    for gi, (kw, ov) in fuzz.items():
                        if exact and gi in exact:
                            continue
                        bump(gi, w_idf * FUZZY_WEIGHT, w_idf, False, len(w), len(kw), ov)

        total = sum(v["score"] for v in acc.values())
        mx = max((v["score"] for v in acc.values()), default=0.0)
        matched = sum(1 for w in q if w in self._global_kw)
        return q, acc, total, mx, matched

    def rank(self, text: str):
        _q, acc, _t, mx, _m = self._score(text)
        ranked = sorted(
            acc.items(),
            key=lambda kv: (-kv[1]["score"], -(kv[1]["exact"] + kv[1]["fuzzy"]), kv[0]),
        )
        out = []
        for gi, v in ranked:
            out.append(
                (
                    self.emojis[gi],
                    _r3(v["score"]),
                    _r3(v["score"] / mx) if mx else 0.0,
                    v["exact"],
                    v["fuzzy"],
                    _r3(v["best_idf"]),
                    self.kw_count[gi],
                    v["kw_len"],
                    v["word_len"],
                    v["overlap"],
                )
            )
            if len(out) >= 32:
                break
        return out

    def flexq(self, text: str) -> dict:
        q, acc, total, mx, matched = self._score(text)
        return {
            "tokens": len(q),
            "matched": matched,
            "sum": _r3(total),
            "max": _r3(mx),
            "cand": len(acc),
        }
