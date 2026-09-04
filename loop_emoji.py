import subprocess
from pathlib import Path

import typer

from test_emoji import _acc, _freq_groups, _length_groups, test_emoji

DATA_PATH = Path("data.jsonl")
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


def _weak_buckets(title: str, groups: list[tuple[str, list[dict]]], target: float) -> None:
    print(f"\n{title}", flush=True)
    for label, rows in groups:
        n, acc, mrr = _acc(rows)
        flag = "  <-- weak" if acc[GOAL_K] < target else ""
        print(
            f"  {label:>12}  n={n:<4} acc@{GOAL_K}={acc[GOAL_K]:.0%}  mrr={mrr:.2f}{flag}",
            flush=True,
        )


def _failure_detail(results: list[dict], target: float) -> None:
    n, acc, mrr = _acc(results)
    missed = [r for r in results if r["rank"] is None or r["rank"] > GOAL_K]
    missed.sort(key=lambda r: (0 if r["rank"] is None else 1, -(r["rank"] or 0)))

    print("\n===== FAILURE DETAIL =====", flush=True)
    print(
        f"final acc@{GOAL_K}={acc[GOAL_K]:.1%}  target={target:.1%}  "
        f"short by {target - acc[GOAL_K]:.1%}  "
        f"({len(missed)}/{n} words with target emoji outside top {GOAL_K})",
        flush=True,
    )
    print(
        f"acc@1={acc[1]:.0%}  acc@3={acc[3]:.0%}  acc@5={acc[5]:.0%}  "
        f"acc@10={acc[10]:.0%}  mrr={mrr:.3f}",
        flush=True,
    )

    print("\nmissed words (worst first):", flush=True)
    head = f"  {'word':<20} {'expected':<12} {'freq':>5} {'rank':>5}  model top 5"
    print(head, flush=True)
    print(f"  {'-' * 20} {'-' * 12} {'-' * 5} {'-' * 5}  {'-' * 20}", flush=True)
    for r in missed:
        rank_s = "none" if r["rank"] is None else str(r["rank"])
        print(
            f"  {r['word'][:20]:<20} {' '.join(r['expected'])[:12]:<12} "
            f"{r['freq_sum']:>5} {rank_s:>5}  {' '.join(r['top'])}",
            flush=True,
        )

    _weak_buckets("acc@5 by word length (chars):", _length_groups(results), target)
    _weak_buckets(
        "acc@5 by expected-emoji freq in train.jsonl (avg):",
        _freq_groups(results),
        target,
    )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    iterations: int = typer.Option(5, help="max train -> test -> upsample iterations"),
    target: float = typer.Option(0.95, help="stop once emoji acc@5 reaches this"),
    rank: int = typer.Option(5, help="upsample words ranked worse than this"),
    cpu: int | None = typer.Option(None, help="Modal --cpu for the task stage"),
    memory: int | None = typer.Option(None, help="Modal --memory in MiB for the task stage"),
) -> None:
    """Emoji train -> test -> upsample loop (Modal task stage, then test_emoji)."""
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

        last = test_emoji()
        acc5 = last["acc"][GOAL_K]
        history.append(acc5)
        print(
            f"iteration {i}: acc@5={acc5:.3f} (target {target:.3f})",
            flush=True,
        )
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
        _failure_detail(last["results"], target)

    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    _app()
