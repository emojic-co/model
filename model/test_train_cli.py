import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer
from typer.testing import CliRunner

import model.train as T
from model.config import TEXT_EMBED_SIZE

runner = CliRunner()


def test_parse_heads_default():
    assert T._parse_heads(None) == ("style", "emoji", "critic", "fusion")


def test_parse_heads_orders_canonically():
    assert T._parse_heads("emoji,style") == ("style", "emoji")
    assert T._parse_heads(" critic , emoji ") == ("emoji", "critic")


def test_all_heads_includes_fusion():
    assert T.ALL_HEADS == ("style", "emoji", "critic", "fusion")


def test_heads_fusion_requires_emoji():
    assert T._parse_heads("emoji,fusion") == ("emoji", "fusion")
    raised = False
    try:
        T._parse_heads("style,fusion")
    except typer.BadParameter:
        raised = True
    assert raised


def test_parse_heads_rejects_unknown():
    for bad in ("bogus", "emoji,bogus", "", ","):
        raised = False
        try:
            T._parse_heads(bad)
        except typer.BadParameter:
            raised = True
        assert raised, bad


def test_validate_heads_only_with_enc():
    for stage in (None, T.Stage.gan):
        raised = False
        try:
            T._validate(stage, True, "emoji", Path("pt"), Path("pt"), "", False)
        except typer.BadParameter:
            raised = True
        assert raised


def test_validate_nondefault_folder_needs_local():
    raised = False
    try:
        T._validate(None, False, None, Path("other"), Path("pt"), "", False)
    except typer.BadParameter:
        raised = True
    assert raised
    assert T._validate(None, True, None, Path("other"), Path("pt"), "", False) is None
    assert T._validate(T.Stage.enc, True, None, Path("pt"), Path("pt"), "", False) == (
        "style",
        "emoji",
        "critic",
        "fusion",
    )


def test_validate_gpu_rejects_local():
    raised = False
    try:
        T._validate(None, True, None, Path("pt"), Path("pt"), "T4", False)
    except typer.BadParameter:
        raised = True
    assert raised
    assert T._validate(None, False, None, Path("pt"), Path("pt"), "T4", False) is None


def test_validate_gpu_and_cpu_mutually_exclusive():
    raised = False
    try:
        T._validate(None, False, None, Path("pt"), Path("pt"), "T4", True)
    except typer.BadParameter:
        raised = True
    assert raised
    assert T._validate(None, False, None, Path("pt"), Path("pt"), "", True) is None


def test_roc_auc_perfect_and_reversed():
    pos = torch.tensor([3.0, 4.0, 5.0])
    neg = torch.tensor([0.0, 1.0, 2.0])
    assert torch.isclose(T.roc_auc(pos, neg), torch.tensor(1.0))
    assert torch.isclose(T.roc_auc(neg, pos), torch.tensor(0.0))


def test_roc_auc_chance_and_empty():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.isclose(T.roc_auc(x, x.clone()), torch.tensor(0.5))
    assert torch.isclose(T.roc_auc(torch.empty(0), x), torch.tensor(0.0))


def test_litencoder_builds_only_selected_heads():
    m = T.LitEncoder(heads=("style",))
    assert hasattr(m, "style") and not hasattr(m, "emoji") and not hasattr(m, "critic")
    m2 = T.LitEncoder(heads=("emoji", "critic"))
    assert hasattr(m2, "emoji") and hasattr(m2, "critic") and not hasattr(m2, "style")
    opt = m2.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)


def test_colorcritic_forward_shape():
    from model.model import ColorCritic

    out = ColorCritic()(torch.zeros(5, TEXT_EMBED_SIZE), torch.zeros(5, 9))
    assert out.shape == (5, 1)


def _stub_runners():
    calls = {}
    T._run_local = lambda *a, **k: calls.setdefault("local", (a, k))
    T._dispatch = lambda *a, **k: calls.setdefault("dispatch", (a, k))
    return calls


def test_cli_help_ok():
    res = runner.invoke(T._app, ["--help"])
    assert res.exit_code == 0
    assert "enc" in res.output and "gan" in res.output


def test_cli_bad_heads_aborts():
    res = runner.invoke(T._app, ["enc", "--heads", "bogus", "--local"])
    assert res.exit_code != 0


def test_cli_heads_with_gan_aborts():
    res = runner.invoke(T._app, ["gan", "--heads", "style", "--local"])
    assert res.exit_code != 0


def test_cli_nondefault_pt_on_modal_aborts():
    res = runner.invoke(T._app, ["--pt", "somewhere"])
    assert res.exit_code != 0


def test_cli_valid_enc_dispatches_local():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["enc", "--heads", "emoji,style", "--local"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["local"]
        assert a[0] == T.Stage.enc
        assert a[1] == ("style", "emoji")
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_cli_remote_defaults_to_gpu():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["enc"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["dispatch"]
        assert a[2] == T.DEFAULT_GPU
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_cli_cpu_flag_forces_cpu_box():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["enc", "--cpu"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["dispatch"]
        assert a[2] == ""
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


def test_cli_gpu_type_override():
    orig_local, orig_dispatch = T._run_local, T._dispatch
    calls = _stub_runners()
    try:
        res = runner.invoke(T._app, ["enc", "--gpu", "A10G"])
        assert res.exit_code == 0, res.output
        (a, _k) = calls["dispatch"]
        assert a[2] == "A10G"
    finally:
        T._run_local, T._dispatch = orig_local, orig_dispatch


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Run the train.py CLI assertion checks."""
    test_parse_heads_default()
    test_parse_heads_orders_canonically()
    test_all_heads_includes_fusion()
    test_heads_fusion_requires_emoji()
    test_parse_heads_rejects_unknown()
    test_validate_heads_only_with_enc()
    test_validate_nondefault_folder_needs_local()
    test_validate_gpu_rejects_local()
    test_validate_gpu_and_cpu_mutually_exclusive()
    test_roc_auc_perfect_and_reversed()
    test_roc_auc_chance_and_empty()
    test_litencoder_builds_only_selected_heads()
    test_colorcritic_forward_shape()
    test_cli_help_ok()
    test_cli_bad_heads_aborts()
    test_cli_heads_with_gan_aborts()
    test_cli_nondefault_pt_on_modal_aborts()
    test_cli_valid_enc_dispatches_local()
    test_cli_remote_defaults_to_gpu()
    test_cli_cpu_flag_forces_cpu_box()
    test_cli_gpu_type_override()
    print("ok")


if __name__ == "__main__":
    _app()
