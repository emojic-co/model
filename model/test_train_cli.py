import os
import sys
from pathlib import Path

import torch
import typer
from typer.testing import CliRunner

import model.train as T
from model.config import EMBED_SIZE_TEXT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


runner = CliRunner()


def test_litencoder_builds_all_heads():
    m = T.LitEncoder()
    assert hasattr(m, "style") and hasattr(m, "emoji")
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)


def test_colorcritic_forward_shape():
    from model.model import ColorEmbedding, CondColorCritic

    color_embedding = ColorEmbedding()(torch.zeros(5, 9))
    score = CondColorCritic()(torch.zeros(5, EMBED_SIZE_TEXT), color_embedding)
    assert score.shape == (5, 1)


def _stub_runners():
    calls = {}
    T._run_local = lambda *a, **k: calls.setdefault("local", (a, k))
    T._dispatch = lambda *a, **k: calls.setdefault("dispatch", (a, k))
    return calls


def test_cli_help_ok():
    res = runner.invoke(T._app, ["--help"])
    assert res.exit_code == 0
    assert "gan" in res.output


def test_cli_bad_stage_aborts():
    res = runner.invoke(T._app, ["enc", "--local"])
    assert res.exit_code != 0


def test_cli_gan_dispatches_local():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["gan", "--local"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["local"]
        assert a[0] == T.Stage.gan
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_cli_no_stage_local():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["--local"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["local"]
        assert a[0] is None
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_cli_no_flags_dispatches_remote():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, [])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["dispatch"]
        assert a[0] is None
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_dispatch_pipeline_trains_gan_locally_after_encoder():
    import contextlib

    import modal

    orig_remote, orig_retrieve, orig_local, orig_run, orig_output = (
        T._run_remote, T._retrieve_and_cleanup, T._run_local,
        T.modal_app.run, modal.enable_output,
    )
    calls = {}
    T._run_remote = lambda stage, *a, **k: calls.setdefault("remote", stage)
    T._retrieve_and_cleanup = lambda: True
    T._run_local = lambda *a, **k: calls.setdefault("local", a)
    T.modal_app.run = lambda: contextlib.nullcontext(T.modal_app)
    modal.enable_output = lambda: contextlib.nullcontext()
    os.environ["EMOJIC_DISPATCH_CHECKED"] = "1"
    try:
        T._dispatch(None)
        assert calls["remote"] == T.Stage.encoder.value
        assert calls["local"] == (T.Stage.gan,)
    finally:
        del os.environ["EMOJIC_DISPATCH_CHECKED"]
        T._run_remote, T._retrieve_and_cleanup, T._run_local = (
            orig_remote, orig_retrieve, orig_local
        )
        T.modal_app.run = orig_run
        modal.enable_output = orig_output


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Run the train.py CLI assertion checks."""
    test_litencoder_builds_all_heads()
    test_colorcritic_forward_shape()
    test_cli_help_ok()
    test_cli_bad_stage_aborts()
    test_cli_gan_dispatches_local()
    test_cli_no_stage_local()
    test_cli_no_flags_dispatches_remote()
    test_dispatch_pipeline_trains_gan_locally_after_encoder()
    print("ok")


if __name__ == "__main__":
    _app()
