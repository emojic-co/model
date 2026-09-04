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


def test_require_clean_tree_dispatch_skip():
    import os

    from runmeta import require_clean_tree

    os.environ["EMOJIC_DISPATCH_CHECKED"] = "1"
    try:
        require_clean_tree()
    finally:
        del os.environ["EMOJIC_DISPATCH_CHECKED"]


def test_require_clean_tree_dirty_exits(tmp_path="/tmp/runmeta-gitdirty"):
    import os
    import subprocess

    import runmeta

    subprocess.run(["rm", "-rf", tmp_path], check=True)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    open(f"{tmp_path}/x.txt", "w").write("hi")
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        raised = False
        try:
            runmeta.require_clean_tree()
        except SystemExit as e:
            raised = True
            assert "clean git tree" in str(e)
        assert raised
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    test_run_meta_shape()
    test_run_meta_sha_env_override()
    test_save_load_round_trip()
    test_load_pt_legacy_bare()
    test_model_slug()
    test_require_clean_tree_dispatch_skip()
    test_require_clean_tree_dirty_exits()
    print("ok")
