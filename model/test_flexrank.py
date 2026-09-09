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
    assert r.kw_vocab == fix["kw_vocab"], "kw_vocab drift vs fixture"
    for case in fix["cases"]:
        got = r.tf_vec(case["text"])
        exp = case["tf"]
        assert len(got) == len(exp), (case["text"], len(got), len(exp))
        for i, (g, e) in enumerate(zip(got, exp, strict=True)):
            assert abs(g - e) < 1e-6, (case["text"], i, g, e)


if __name__ == "__main__":
    test_matches_shared_fixture()
    print("ok")
