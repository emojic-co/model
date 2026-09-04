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


def test_save_load_round_trip(tmp_path="/tmp/runmeta-rt.pt"):
    import torch

    from runmeta import load_pt, save_pt

    save_pt({"w": torch.zeros(1)}, tmp_path, stage="task")
    sd, meta = load_pt(tmp_path)
    assert list(sd) == ["w"]
    assert meta["stage"] == "task"
    assert meta["sha"]


def test_load_pt_legacy_bare(tmp_path="/tmp/runmeta-legacy.pt"):
    import torch

    from runmeta import load_pt

    torch.save({"w": torch.zeros(1)}, tmp_path)
    sd, meta = load_pt(tmp_path)
    assert list(sd) == ["w"]
    assert meta is None


def test_model_slug():
    from runmeta import model_slug

    assert model_slug({"sha": "abc1234"}) == "abc1234"
    assert model_slug(None) == "nometa"
    assert model_slug({}) == "nometa"
    assert model_slug({"sha": ""}) == "nometa"


if __name__ == "__main__":
    test_run_meta_shape()
    test_run_meta_sha_env_override()
    test_save_load_round_trip()
    test_load_pt_legacy_bare()
    test_model_slug()
    print("ok")
