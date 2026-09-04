import json
import subprocess
from pathlib import Path

import typer

DATA_PATH = Path("data.jsonl")
REPORT_DIR = Path("report")
GOAL_K = 5


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _commit(msg: str) -> None:
    subprocess.run(["git", "add", "data.jsonl", "report"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "commit", "-m", msg], check=True)


def _emoji_report() -> dict:
    _run(["uv", "run", "python", "tools/report.py", "--only", "emoji"])
    latest = max(REPORT_DIR.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))["emoji"]


def _failure_detail(kw: dict, target: float) -> None:
    print("\n===== FAILURE DETAIL =====", flush=True)
    print(
        f"final acc@{GOAL_K}={kw[f'acc@{GOAL_K}']:.1%}  target={target:.1%}  "
        f"short by {target - kw[f'acc@{GOAL_K}']:.1%}  (n={kw['n']})",
        flush=True,
    )
    print(
        f"acc@1={kw['acc@1']:.0%}  acc@3={kw['acc@3']:.0%}  "
        f"acc@5={kw['acc@5']:.0%}  acc@10={kw['acc@10']:.0%}  MRR={kw['MRR']:.3f}",
        flush=True,
    )
    missed = [r for r in kw["words"] if r["rank"] is None or r["rank"] > GOAL_K]
    print(f"\nmissed keywords ({len(missed)}/{kw['n']}, worst first):", flush=True)
    print(f"  {'word':<20} {'expected':<12} {'rank':>5}  model top 3", flush=True)
    print(f"  {'-' * 20} {'-' * 12} {'-' * 5}  {'-' * 20}", flush=True)
    for r in missed:
        rank_s = "none" if r["rank"] is None else str(r["rank"])
        print(
            f"  {r['keyword'][:20]:<20} {' '.join(r['expected'])[:12]:<12} "
            f"{rank_s:>5}  {' '.join(r['top3'])}",
            flush=True,
        )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    iterations: int = typer.Option(5, help="max train -> report -> upsample iterations"),
    target: float = typer.Option(0.95, help="stop once keyword acc@5 reaches this"),
    rank: int = typer.Option(5, help="upsample words ranked worse than this"),
    cpu: int | None = typer.Option(None, help="Modal --cpu for the task stage"),
    memory: int | None = typer.Option(None, help="Modal --memory in MiB for the task stage"),
) -> None:
    """Emoji train -> report -> upsample loop (Modal task stage, then tools/report.py)."""
    from runmeta import require_clean_tree

    require_clean_tree()

    modal_cmd = ["uv", "run", "modal", "run", "train-modal.py::main"]
    if cpu:
        modal_cmd += ["--cpu", str(cpu)]
    if memory:
        modal_cmd += ["--memory", str(memory)]

    history: list[float] = []
    last: dict | None = None
    for i in range(1, iterations + 1):
        print(f"\n===== iteration {i}/{iterations} =====", flush=True)
        _run(modal_cmd)

        last = _emoji_report()["keywords"]
        acc5 = last[f"acc@{GOAL_K}"]
        history.append(acc5)
        print(f"iteration {i}: acc@5={acc5:.3f} (target {target:.3f})", flush=True)
        if acc5 >= target:
            print(f"\ntarget reached at iteration {i}", flush=True)
            break
        if i == iterations:
            break

        before = _lines(DATA_PATH)
        _run(["bun", "run", "tools/data/upsample-emoji-test.ts", "--rank", str(rank)])
        added = _lines(DATA_PATH) - before
        print(f"upsample added {added} rows to {DATA_PATH}", flush=True)
        if added <= 0:
            print("\nnothing left to upsample, stopping", flush=True)
            break
        _run(["bun", "run", "tools/data/regen.ts"])
        _commit(f"loop_emoji iter {i}: +{added} rows")

    print("\n===== summary =====", flush=True)
    for n, acc5 in enumerate(history, 1):
        print(f"  iter {n}: acc@5={acc5:.3f}", flush=True)
    ok = bool(history) and history[-1] >= target
    final = history[-1] if history else 0.0
    print(
        f"\n{'PASS' if ok else 'STOP'}: final acc@5={final:.3f} target={target:.3f}",
        flush=True,
    )

    if ok:
        print("\ngoal reached -- training the color GAN locally", flush=True)
        _commit("loop_emoji: commit reports before color GAN")
        _run(["uv", "run", "python", "train_gan.py"])
    elif last is not None:
        _failure_detail(last, target)

    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    _app()
