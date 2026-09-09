import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from files import FLEX_JSON
from model.flexrank import FlexRanker

FIX = Path("web/src/flexrank.fixture.json")


def test_matches_shared_fixture():
    fix = json.loads(FIX.read_text())
    r = FlexRanker(FLEX_JSON)
    for case in fix["cases"]:
        got = r.rank(case["text"])
        exp = case["flexsearch"]
        assert len(got) == len(exp), case["text"]
        for g, e in zip(got, exp, strict=True):
            assert g[0] == e[0], (case["text"], g, e)
            for gi, ei in zip(g[1:], e[1:], strict=True):
                assert abs(float(gi) - float(ei)) < 1e-6, (case["text"], g, e)
        q = r.flexq(case["text"])
        for k, v in case["flexq"].items():
            assert abs(q[k] - v) < 1e-6, (case["text"], k, q[k], v)


if __name__ == "__main__":
    test_matches_shared_fixture()
    print("ok")
