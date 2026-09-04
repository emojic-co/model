import os

from runmeta import run_meta


def test_run_meta_shape():
    m = run_meta()
    assert set(m) == {"sha", "dirty", "generated", "config", "train_sha"}
    assert isinstance(m["sha"], str) and m["sha"]
    assert isinstance(m["dirty"], bool)
    assert "T" in m["generated"]
    assert isinstance(m["config"], list) and m["config"][0].startswith("ENCODER: ")
    assert m["train_sha"] is None or (
        isinstance(m["train_sha"], str) and len(m["train_sha"]) == 12
    )


def test_run_meta_sha_env_override():
    os.environ["EMOJIC_GIT_SHA"] = "deadbee"
    os.environ["EMOJIC_GIT_DIRTY"] = "0"
    try:
        m = run_meta()
        assert m["sha"] == "deadbee"
        assert m["dirty"] is False
    finally:
        del os.environ["EMOJIC_GIT_SHA"]
        del os.environ["EMOJIC_GIT_DIRTY"]


if __name__ == "__main__":
    test_run_meta_shape()
    test_run_meta_sha_env_override()
    print("ok")
