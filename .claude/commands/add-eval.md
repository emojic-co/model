---
description: Append ~50 cold-verified rows per feeling (~350 total) to eval.jsonl, leak-free vs data.jsonl
allowed-tools: [Bash, Read, Write]
---

# Add eval samples

Grow the gold holdout `eval.jsonl` by **~50 rows per feeling (~350 total)** of
**100%-correct** data, without ever reading the whole corpus, and without
leaking into or out of `data.jsonl`.

`eval.jsonl` is the fixed evaluation set (`data.py:split`). Every row you add
is scored against the model forever, so the bar is: a reader seeing only the
`text` — no context, feeling hidden — would name the row's `feeling` and no
other. Anything less does not go in.

## Steps

1. **Read the label sets + length cap — never the corpus.** Run:

   ```bash
   uv run python3 -c "import json; d=json.load(open('labels.json')); from config import MAX_TEXT_LEN; print('MAX_TEXT_LEN', MAX_TEXT_LEN); print('FEELINGS', d['feelings']); print('EMOJIS', ' '.join(d['emojis']))"
   ```

   `feelings` is the closed set of 7. `emojis` is the 300-emoji palette —
   the only emojis allowed in an `emoji` field.

2. **Sample a style seed — not the whole file.** Run:

   ```bash
   shuf -n 70 eval.jsonl
   ```

   (fallback: `sort -R eval.jsonl | head -n 70`). These rows are your only
   view of `eval.jsonl`. Infer from them: line schema (`{"text", "feeling",
   "emoji"}`, no other keys), the short WhatsApp-style voice, typical length,
   punctuation, and how unambiguous each `text` is about its `feeling`.

3. **Generate ~70 candidates per feeling** (350 → ~490; the surplus absorbs
   verification and dedup drops). For each feeling:
   - Short, natural, first-person-ish messages someone would actually send.
     Spread the voice — different speakers, situations, registers — so the
     set is not 70 paraphrases of one sentence.
   - The `feeling` must be the **single** defensible reading of the `text`.
     No mixed signals, no "could be Calm or Neutral". If you have to think
     about it, discard it.
   - `emoji`: pick one from the palette that a person would plausibly attach
     to that message. **Never name or describe the emoji in the `text`.**
   - Keep `text` inside `MAX_TEXT_LEN` after normalization (lowercased,
     whitespace-collapsed, run-length-trimmed, non-vocab chars dropped —
     `data.py:normalize`). Stay a few chars under to be safe.
   - Do not reuse a seed row's `text`; vary phrasing and topic.

4. **Cold-verify every candidate yourself.** Re-read each `text` with the
   feeling hidden. Keep it only if:
   - you independently land on the same `feeling`, **and**
   - no second feeling from the 7 is defensible for that `text`.

   Drop everything else. Aim to keep **~50 per feeling**. If a feeling drops
   below ~45, generate a fresh top-up batch for it and re-verify (repeat
   step 3–4 for that feeling only).

5. **Write survivors to a temp file** — one JSON object per line, keys
   `text`, `feeling`, `emoji`, in that order, `ensure_ascii=False`:
   `/tmp/eval_candidates.jsonl`.

6. **Dedup — leak-free vs both files.** Run:

   ```bash
   uv run python3 - <<'EOF'
   import json
   from config import MAX_TEXT_LEN
   from data import normalize

   seen = set()
   for path in ("eval.jsonl", "data.jsonl"):
       with open(path, encoding="utf-8") as f:
           for line in f:
               if line.strip():
                   seen.add(normalize(json.loads(line)["text"]))

   kept, dropped = [], 0
   with open("/tmp/eval_candidates.jsonl", encoding="utf-8") as f:
       for line in f:
           if not line.strip():
               continue
           r = json.loads(line)
           k = normalize(r["text"])
           if not k or len(k) > MAX_TEXT_LEN or k in seen:
               dropped += 1
               continue
           seen.add(k)
           kept.append(r)

   with open("/tmp/eval_new.jsonl", "w", encoding="utf-8") as f:
       for r in kept:
           f.write(json.dumps({"text": r["text"], "feeling": r["feeling"], "emoji": r["emoji"]}, ensure_ascii=False) + "\n")

   from collections import Counter
   print(f"kept {len(kept)}, dropped {dropped}")
   print("per feeling:", dict(Counter(r["feeling"] for r in kept)))
   EOF
   ```

   A normalized `text` that already exists in `eval.jsonl` **or** `data.jsonl`
   is dropped — the holdout must not overlap the training corpus. If any
   feeling is now well below ~50, go back to step 3 for that feeling.

7. **Append — never overwrite.** Run:

   ```bash
   wc -l eval.jsonl                 # record OLD count
   cat /tmp/eval_new.jsonl >> eval.jsonl
   wc -l eval.jsonl                 # NEW count
   ```

   Do not truncate or rewrite `eval.jsonl`.

8. **Verify and report.**
   - Confirm `eval.jsonl` grew by exactly the line count of
     `/tmp/eval_new.jsonl`.
   - `tail -n 1 eval.jsonl | python3 -c "import json,sys; json.loads(sys.stdin.read())"`
     — last line is valid JSON.
   - Print the new per-feeling totals:
     `python3 -c "import json,collections; print(collections.Counter(json.loads(l)['feeling'] for l in open('eval.jsonl') if l.strip()))"`
   - Refresh the stats report: `uv run python tools/eval/eval-stat.py`.
   - Report old count, new count, rows added per feeling.
