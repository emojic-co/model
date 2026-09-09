# GPU remote training

Should the Modal remote run (`model/train.py`, no `--local`) use a GPU box
instead of the 16‑core CPU box? The `gpu` branch makes a **GPU box the default
for remote runs**, with `--cpu` to fall back. This note is the reasoning behind
it and what still has to be measured before we keep it.

## What the `gpu` branch changes

- **`train` (remote) now runs on a GPU by default** (`DEFAULT_GPU = "T4"`),
  dispatched via `Function.with_options(gpu=…, cpu=8, …)`. `--gpu L4` /
  `--gpu A10G` pick another type; **`--cpu`** forces the old CPU box; `--gpu`
  and `--cpu` are mutually exclusive. `--gpu` with `--local` is rejected (a
  local GPU is already used by Lightning's `accelerator="auto"`); `--local`
  itself is unaffected.
- **torch is now a `cpu` / `gpu` group split** (`pyproject.toml`). `uv sync` /
  `uv run` locally still resolve `torch==2.13.0+cpu` (default groups
  `dev`, `cpu`) — unchanged. The Modal image builds with
  `uv sync --frozen --no-default-groups --group dev --group gpu`, which pulls
  `torch==2.13.0+cu126` and the CUDA runtime libs. One image serves both CPU and
  GPU boxes.
- **Data loaders** (`model/data.py`): `pin_memory` is on automatically whenever
  CUDA is present; `num_workers` is `EMOJIC_DATA_WORKERS` (default `0`, set to
  `4` on the GPU path) with `persistent_workers` + `prefetch_factor=4` when
  workers > 0.
- **Batch size** is env‑overridable — `EMOJIC_TASK_BATCH_SIZE` /
  `EMOJIC_GAN_BATCH_SIZE` (defaults `256` / `512`, unchanged for `--local`). The
  GPU dispatch sets `512` / `1024` for better device utilisation.
- **GPU‑only training tweaks** (`_run_local`, gated on `torch.cuda.is_available()`):
  `cudnn.benchmark=True` (input shapes are static — fixed seq len, `drop_last`),
  `set_float32_matmul_precision("high")` (TF32), and `deterministic="warn"`
  instead of `True` (CUDA `nn.Embedding` backward has no deterministic kernel).
  CPU runs are byte‑for‑byte unchanged.

Host CPU on the GPU box drops from 16 to 8 cores (`GPU_CPU`), which is the main
cost lever.

## How to run

```
train                      # stage 1 + stage 2 + export on a T4 (default)
train enc                  # stage 1 only, on a T4
train --gpu L4             # override the GPU type
train gan --gpu A10G       # stage 2 only (needs enc.pt/critic.pt/style.pt/emoji.pt)
train --cpu                # fall back to the CPU Modal box
train --local              # unchanged: this machine
```

Tune without code edits via env vars on the dispatch, e.g.
`EMOJIC_TASK_BATCH_SIZE=1024 EMOJIC_DATA_WORKERS=8 train --gpu A10G`.

## Will a GPU actually help here?

The model is tiny (~377 K params in encoder + heads; GAN gen/critic are small
MLPs; char input, seq len 42), so per‑step math is trivial and the card is
latency‑bound, not compute‑bound. What can still improve:

- **GAN stage** — the longer of the two stages; frozen‑encoder forward over
  batch 512–1024 benefits most.
- **Validation `cdist`** — `energy_distance` / `roc_auc` are O(n²) over the
  2000‑row eval set every validation; GPU‑friendly.
- **Bigger batch** — fewer Python‑side step iterations, better occupancy.

The pre‑materialised in‑RAM tensor dataset means `num_workers` mostly hides a
~1–2 ms/step main‑thread collate; it is not a big lever and >4 workers risks
memory on the smaller box.

## Rough estimates (unmeasured — order‑of‑magnitude only)

Back‑of‑envelope from the model shape; no baseline run has been timed.

| | Speedup vs current CPU | Notes |
|---|---|---|
| Pessimistic | ~1.5× | short run dominated by container start / image pull |
| Best guess | **~3×** | GPU step ~10–25 ms vs CPU ~40–100 ms; bigger batch |
| Optimistic | ~5× | GAN stage and `cdist` validation dominate |

| | Cost reduction vs current | Assumption |
|---|---|---|
| Pessimistic | ~1× (break‑even) | A10G+ chosen, speedup only ~1.5× |
| Best guess | **~3×** | T4/L4 + 8 host cores, ~3× speedup |
| Optimistic | ~5× | ~5× speedup on a cheap GPU |

An entry GPU (T4 ≈ $0.6/hr, L4 ≈ $0.8/hr on top of host CPU/RAM) costs about the
same per hour as the current 16‑core box once host cores drop to 8, so the cost
factor roughly tracks the speedup. Rates move — confirm against
`modal.com/pricing` and the dashboard cost of a real run.

## Cons / risks still open

- **Uncertain payoff** — the model may be too small to keep any GPU busy.
- **Cost may rise** if the GPU is underutilised or a big tier is picked.
- **Determinism** — GPU runs are not bit‑identical to the CPU baselines
  (`deterministic="warn"`, TF32, cudnn autotune).
- **Bigger image** (~5–7 GiB vs ~2 GiB) → slower first build/pull; amortised by
  layer caching.
- **Batch 512/1024 changes training dynamics** — LR is unchanged, so watch
  `MRR/e/val` / `energy/gan/val` vs a CPU baseline; revert the GPU batch sizes
  (`GPU_TASK_BATCH_SIZE` / `GPU_GAN_BATCH_SIZE` in `model/train.py`) if quality
  regresses.
- GPU capacity/queueing on Modal can add scheduling latency.

## What still needs measuring before keeping this

1. Baseline: wall‑clock + Modal cost of a current CPU `enc` and `gan` run.
2. One `train enc --gpu T4` and one `train gan --gpu T4`; compare wall‑clock,
   cost, and `MRR/e/val` / `energy/gan/val` against the baseline.
3. Repeat on L4 / A10G if T4 is loader‑ or memory‑bound.
4. Keep the GPU default only if both the speedup **and** the cost factor land
   above ~2× with no metric regression; otherwise flip the default back so
   remote runs are CPU unless `--gpu` is passed, or revert the branch.
