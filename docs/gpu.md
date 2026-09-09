# GPU remote training

Should the Modal remote run (`model/train.py`, no `--local`) move from a 16‑core
CPU box to a GPU box? This note lays out the trade‑offs so we can decide before
spending effort on it.

## Where we are today

- Remote training runs on `modal.Image.debian_slim` with `cpu=16`,
  `memory=16 GiB`, `timeout=180 min`, **no GPU**.
- `torch` is hard‑pinned to the CPU wheel index (`[tool.uv.index] pytorch-cpu`
  and `[tool.uv.sources] torch` in `pyproject.toml`). A GPU box needs a CUDA
  build instead.
- The Lightning `Trainer` already passes `accelerator="auto"`,
  `devices="auto"`. If CUDA torch is installed and a GPU is attached, training
  uses it with **no change to the training code** — the work is all in packaging
  and the Modal function config.
- The model is small: ~377 K parameters in the encoder + heads, and the GAN
  generator/critic are tiny MLPs. Char‑level input, sequence length 42, batch
  256 (stage 1) / 512 (stage 2).
- The data loaders use `num_workers=0` and no `pin_memory`
  (`model/data.py`).

## The real question: will a GPU actually help here?

For a model this small the per‑step math is trivial. The likely bottlenecks are
**not** the parts a GPU accelerates:

- **Data loading.** With `num_workers=0`, the Python‑side collate and the
  965‑wide multi‑hot construction run on one thread. On a GPU box this becomes
  the bottleneck almost immediately and the GPU sits idle waiting for batches.
- **Kernel‑launch overhead.** A 377 K‑param model with batch 256 issues many
  tiny GPU ops; the card is latency‑bound, not compute‑bound. Speedups for
  models this size are often modest (roughly 1.5–4×) and occasionally negative.
- **Validation `cdist`.** `energy_distance` / `roc_auc` do O(n²) work over the
  2000‑row eval set every validation. This part *is* GPU‑friendly and is the
  most likely place to see a clear win.

**Cost is not guaranteed to go down.** Modal bills the GPU by the second *on top
of* the host CPU/RAM. If wall‑clock only halves but the hourly rate more than
doubles, the run gets more expensive. A 16‑core CPU run already costs a
non‑trivial amount per hour; an entry GPU (T4/L4) is in a similar range, and
A10G/A100 are several times more. Pull the actual per‑run cost from the Modal
dashboard before and after — don't estimate.

## Pros

- Faster validation loop (`cdist`‑heavy metrics) and faster GAN stage, which is
  the longer of the two.
- Head‑room to grow the model (bigger encoder, larger batch, more epochs)
  without wall‑clock becoming painful.
- Shorter iteration time on data/label changes if the speedup is real.
- `deterministic=True` and `accelerator="auto"` mean the code mostly "just
  works" on CUDA once torch is installed.

## Cons / risks

- **Packaging.** The CPU pin has to become a `cpu` / `gpu` extra split (uv's
  conflicting‑extras pattern) so local dev and the post‑run report/export stay
  on CPU wheels while the Modal image installs `pytorch-cu124`. Re‑locking and a
  larger image (~5–7 GiB vs ~2 GiB) follow.
- **Uncertain payoff.** The model may be too small to keep a GPU busy; the win
  could be marginal after the data loader is fixed on CPU anyway.
- **Cost may rise**, not fall, if the GPU is underutilised.
- **Determinism.** Some CUDA kernels have no deterministic implementation.
  `deterministic=True` may need to drop to `"warn"`, and GPU runs will not be
  bit‑identical to the CPU runs we have baselines for.
- **Cold starts.** First run after a dependency change pays a longer image
  build/pull; amortised afterwards via layer caching.
- GPU capacity/queueing on Modal can add latency vs always‑available CPU.

## Cheaper things to try first (no GPU)

- Set `num_workers` > 0, `persistent_workers=True`, `pin_memory=True` in the
  loaders — helps CPU and is a prerequisite for GPU anyway.
- Revisit `EARLY_STOP_PATIENCE` (20) × `VAL_CHECK_INTERVAL` (100) — the real
  epoch ceiling is set by early stopping, not `EPOCHS_TASK`.
- Drop `cpu=16` if the box is not CPU‑bound (direct cost saving).

## Rough estimates (unmeasured — order‑of‑magnitude only)

No baseline run has been timed yet, so these are back‑of‑envelope from the model
shape (~377 K params, 3 conv1d layers, batch 256/512, seq 42).

**Per‑step compute** is ~2–4e10 FLOP for stage 1 and mostly frozen‑encoder
forward for stage 2. On 16 CPU cores that is roughly 40–100 ms/step; on a T4 it
is launch‑bound at roughly 10–25 ms/step.

| | Speedup vs current CPU | Notes |
|---|---|---|
| Pessimistic | ~1.5× | `num_workers=0` kept; short run dominated by container start / image pull |
| Best guess | **~3×** | after `num_workers>0`; entry GPU (T4/L4) |
| Optimistic | ~5× | loader fixed, GAN stage and `cdist` validation benefit most |

**Cost.** An entry GPU on Modal (T4 ≈ $0.6/hr, L4 ≈ $0.8/hr on top of host
CPU/RAM) costs about the same per hour as the current 16‑core CPU box **if** we
also drop the GPU box to ~4 host cores. Under that assumption cost reduction
roughly tracks the speedup:

| | Cost reduction vs current | Assumption |
|---|---|---|
| Pessimistic | ~1× (break‑even) | A10G+ or 16 cores kept, speedup only ~1.5× |
| Best guess | **~3×** | T4/L4 + ~4 host cores, ~3× speedup |
| Optimistic | ~5× | ~5× speedup, cheap GPU, trimmed host cores |

Rates move and Modal's CPU per‑core price in particular has a wide public
spread; confirm against `modal.com/pricing` and the dashboard cost of a real
run. The single biggest lever on the cost factor is dropping host CPU cores on
the GPU box, not the GPU tier.

## Recommended path

1. Record baseline wall‑clock and cost for the current CPU `enc` and `gan`
   runs; add coarse timing to see if the data loader is the bottleneck.
2. Fix the data loader (`num_workers`, `pin_memory`) and re‑measure on CPU.
3. Only if still wall‑clock‑bound: add the `cpu`/`gpu` extra split and a `gpu=`
   kwarg (default `T4`), keep a CPU‑only remote fallback.
4. Run one GPU `enc` + one GPU `gan` on T4/L4/A10G, compare wall‑clock **and**
   Modal cost against step 1, then keep or revert.

Net: worth a measured experiment, not an obvious win. The data‑loader fix is
the low‑risk change; the GPU itself should be gated on numbers from step 2.
