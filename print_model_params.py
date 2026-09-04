import typer
from torch import nn

from model import ColorDsc, ColorGen, EmojiHead, StyleHead, TextEncoder

MODELS: list[tuple[str, type[nn.Module]]] = [
    ("TextEncoder", TextEncoder),
    ("StyleHead", StyleHead),
    ("EmojiHead", EmojiHead),
    ("ColorGen", ColorGen),
    ("ColorDsc", ColorDsc),
]


def groups(mod: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for pname, p in mod.named_parameters():
        key = pname.split(".")[0]
        out[key] = out.get(key, 0) + p.numel()
    return out


def main() -> None:
    total = 0
    for name, cls in MODELS:
        mod = cls()
        g = groups(mod)
        for key, n in g.items():
            print(f"  {name}.{key:<12} {n:>10,}")
        n = sum(g.values())
        total += n
        print(f"{name:<24} {n:>10,}")
        print()

    print(f"{'TOTAL':<24} {total:>10,}")


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli() -> None:
    """Print per-module parameter counts for each model class."""
    main()


if __name__ == "__main__":
    _app()
