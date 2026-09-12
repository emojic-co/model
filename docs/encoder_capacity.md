# Encoder block capacity

`tools/report.py`'s **"Model — Encoder block capacity"** table (source:
`_section_block_capacity`, `tools/report.py:332`; standalone CLI:
`tools/block_capacity.py`) answers one question: of the encoder's stacked
conv blocks, which ones actually move the emoji prediction, versus which
ones are just burning channel budget?

Channel count alone can't tell you this. `ENCODER_CHANNELS` says how many
dimensions each block *has*; this table says how many of those dimensions
the model actually *uses* by the time a decision gets made.

## Where the numbers come from

`TextEncoder` (`model/model.py`) is a stack of dilated `Conv1d` blocks. Each
block's output gets masked-max-pooled over time, and the pooled vectors are
concatenated into one `emb` vector of size `TEXT_EMBED_SIZE =
sum(ENCODER_CHANNELS)` — see `docs/model.md`. Block *i* owns a fixed,
contiguous slice of `emb`, e.g. with `ENCODER_CHANNELS = [200, 200, 200,
200]`, block 0 is `emb[0:200]`, block 1 is `emb[200:400]`, and so on.

`EmojiHead` turns `emb` into a query vector via one linear layer:
`q = weight @ emb` (`weight` is `EmojiHead.net[1].weight`). Because matrix
multiplication distributes over the concatenation, this splits *exactly*,
with no approximation, into one additive term per block:

```
q = weight @ emb
  = weight[:, 0:200]   @ emb[0:200]      (block 0's contribution, v0)
  + weight[:, 200:400] @ emb[200:400]    (block 1's contribution, v1)
  + weight[:, 400:600] @ emb[400:600]    (block 2's contribution, v2)
  + weight[:, 600:800] @ emb[600:800]    (block 3's contribution, v3)
```

The table reports, per block, one row per source (`keywords.jsonl` /
`terms.jsonl` / `eval.jsonl`), each row averaged over every valid sample in
that source's file — `N` says how many:

| Column | What it is | Depends on the sample? |
|---|---|---|
| `N` | how many samples from this source the row is averaged over | — |
| `range` | the block's slice of `emb` | no — fixed by `ENCODER_CHANNELS` |
| `dilation` | the block's dilation rate | no — fixed by `ENCODER_DILATION` |
| `W col-norm mean` / `max` | norm of `weight`'s columns restricted to this block's slice (mean / max over the block's channels) | no — a property of the *trained head weights* only |
| `Activation norm` | `‖emb[slice]‖` — how large this block's pooled output is, averaged over the `N` samples | **yes** — this is `v_i`'s input factor |
| `Contribution norm` | `‖v_i‖ = ‖weight[:, slice] @ emb[slice]‖` — the size of this block's exact additive term in `q`, averaged over the `N` samples | yes |
| `Contribution %` | `100 * ‖v_i‖ / ‖q‖`, averaged over the `N` samples | yes |

`W col-norm` tells you how strongly the head is *wired* to listen to a
block, independent of any input. `Activation norm` tells you how much that
block actually *said*, on average, across every sample in that source.
`Contribution norm`/`%` is the two multiplied together — how much the
block actually swayed the final answer, on average.

## Reading it

1. **Compare `Contribution %` to the block's fair share of channels.**
   With 4 equal 200-channel blocks, "fair share" is 25% each. A block
   sitting far below that is being paid for (in channels, and in every
   forward/backward pass) but isn't pulling its weight in the decision.

2. **Use `W col-norm` vs. `Activation norm` to tell head-side from
   encoder-side problems.** High `W col-norm` + low `Activation norm` on
   the same block means the head is *ready* to use that block but the
   block itself isn't producing signal — an encoder problem (undertrained,
   under-provisioned, or losing signal to depth). The opposite pattern
   (block outputs plenty, head barely wired to it) would instead point at
   the head/training, not the encoder.

3. **Look for a pattern that repeats across all three sources**, not one
   row. Each source is now averaged over hundreds to thousands of samples
   (see `N`), so a pattern that holds across `keywords`/`terms`/`eval`
   alike is a real, statistically solid signal — not a fluke from one
   lucky/unlucky sentence.

## Concrete example

From the current model (`ENCODER_CHANNELS = [200, 200, 200, 200]`,
`ENCODER_DILATION = [1, 2, 4, 8]`), the `eval` row — averaged over all 1999
valid samples in `data/eval.jsonl`:

| block | dilation | W col-norm mean | Activation norm | Contribution norm | Contribution % |
|---|---|---|---|---|---|
| 0 | 1 | 1.87 | 1.372 | 2.214 | 64.4% |
| 1 | 2 | 7.99 | 0.303 | 2.397 | 68.2% |
| 2 | 4 | 3.92 | 0.126 | 0.575 | 16.6% |
| 3 | 8 | 1.71 | 0.080 | 0.313 | 9.4% |

What this says:

- **Activation norm falls off steeply with depth**: `1.37 → 0.30 → 0.13 →
  0.08`, on average across ~2000 sentences. Each deeper, more-dilated
  block's pooled output is smaller — a classic vanishing-signal pattern
  for a plain stacked-conv encoder with no residual connections and no
  normalization between blocks.
- **Weight norms don't fall off the same way** — block 1 actually has the
  *strongest* weights (mean 7.99, nearly double block 2's and 4.7x block
  0's), meaning the head is wired to lean on block 1 hardest. But its
  small activation caps how much that can matter in practice.
- **Net effect**: blocks 0 and 1 end up contributing similarly (64–68%)
  despite very different weight norms — block 0's modest weights are
  offset by a large activation, block 1's strong weights are offset by a
  small one. Blocks 2 and 3 lose on *both* counts, and end up contributing
  16.6% and 9.4% on average — each despite owning the exact same
  200-channel, 25%-of-total budget as blocks 0 and 1.
- Block 3 — the widest-context block, the one meant to see the most of the
  sentence — is the cheapest 25% of the model, in every real sense, and
  this holds on average across the whole eval set, not just one sentence.

### Why the percentages add up to ~159%, not 100%

`64.4 + 68.2 + 16.6 + 9.4 = 158.6`. This isn't a bug: `q = v0+v1+v2+v3`
splits exactly, but `Contribution %` compares each `‖vᵢ‖` to `‖q‖`, and
vector norms don't add — only the triangle inequality holds:

```
‖v0 + v1 + v2 + v3‖  ≤  ‖v0‖ + ‖v1‖ + ‖v2‖ + ‖v3‖
```

Equality only happens if every block's contribution points in *exactly*
the same direction in query space. Any disagreement between blocks makes
the right side bigger than the left — some of each block's push gets
undone by the others when they're added together. So the amount the total
exceeds 100% is itself informative: a total near 100% means the blocks are
contributing nearly independent, aligned evidence; a total this far above
100% means there's real redundancy or outright tension between what
different blocks are pushing the answer toward, not four cleanly additive
pieces of evidence. Because each row is now averaged over hundreds to
thousands of samples, this overshoot is a stable property of the model,
not sampling noise from picking one sentence.

## What this is *not*

A low `Contribution %` is not by itself proof that a block should be made
smaller: in this project's history, narrowing deep/high-dilation blocks
has not been the fix that moved full-text eval accuracy — widening them,
or adding more depth, has. Treat this table as a diagnostic for *where*
signal is getting lost, not a direct instruction on which block to shrink.
