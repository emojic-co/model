import argparse
import subprocess
import sys
from pathlib import Path

from test_emoji import test_emoji

TRAIN_PATH = Path("train.jsonl")


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--cpu", type=int)
    p.add_argument("--memory", type=int)
    args = p.parse_args()

    modal_cmd = ["uv", "run", "modal", "run", "train-modal.py"]
    if args.cpu:
        modal_cmd += ["--cpu", str(args.cpu)]
    if args.memory:
        modal_cmd += ["--memory", str(args.memory)]

    history: list[float] = []
    for i in range(1, args.iterations + 1):
        print(f"\n===== iteration {i}/{args.iterations} =====", flush=True)
        _run(modal_cmd)

        acc5 = test_emoji()["acc"][5]
        history.append(acc5)
        print(
            f"iteration {i}: acc@5={acc5:.3f} (target {args.target:.3f})",
            flush=True,
        )
        if acc5 >= args.target:
            print(f"\ntarget reached at iteration {i}", flush=True)
            break
        if i == args.iterations:
            break

        before = _lines(TRAIN_PATH)
        _run(["bun", "run", "tools/data/upsample-emoji-test.ts", "--rank", str(args.rank)])
        added = _lines(TRAIN_PATH) - before
        print(f"upsample added {added} rows to {TRAIN_PATH}", flush=True)
        if added <= 0:
            print("\nnothing left to upsample, stopping", flush=True)
            break

    print("\n===== summary =====", flush=True)
    for n, acc5 in enumerate(history, 1):
        print(f"  iter {n}: acc@5={acc5:.3f}", flush=True)
    ok = bool(history) and history[-1] >= args.target
    final = history[-1] if history else 0.0
    print(
        f"\n{'PASS' if ok else 'STOP'}: final acc@5={final:.3f} target={args.target:.3f}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
