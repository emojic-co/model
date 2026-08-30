---
description: Rebuild eval.jsonl from scratch — fold any existing eval.jsonl back into train.jsonl, then sample 1,000 rows whose feeling and emoji are both in labels.json and keep only the correctly-labeled ones
allowed-tools: [Bash, Read, Write, Edit]
---

# Generate eval.jsonl

Rebuild the gold holdout `eval.jsonl` from scratch. If an `eval.jsonl` already
exists, every one of its rows is folded **back into** `train.jsonl` first, so the
new holdout is drawn from the full corpus and never silently accumulates stale
rows. Then sample **only from rows whose `feeling` and `emoji` are both present
in `labels.json`** — those are exactly the labels the model can predict, so a row
carrying an out-of-vocab label could never be scored in the validation step —
keep only rows whose **both** labels are correct and whose text length is in
range, and move every kept row **out** of `train.jsonl` so the holdout never
overlaps the training corpus.

Labels are open-set: `feeling` is any single capitalized word, `emoji` is any
emoji. You are judging whether the existing labels are right — **never rewrite a
label**. A row is keep-or-drop only.

Text-length range: **4–48 code points** on the raw `text` (no normalization).

## Steps

1. **Fold an existing `eval.jsonl` back into `train.jsonl`.** Skip this step if
   `eval.jsonl` does not exist. Record how many rows were folded back — you
   report it at the end.

   ```bash
   wc -l train.jsonl
   test -f eval.jsonl && python3 - <<'EOF' || echo "no eval.jsonl — nothing to fold back"
   import os

   def ensure_trailing_newline(path):
       if os.path.getsize(path) == 0:
           return
       with open(path, "rb+") as f:
           f.seek(-1, os.SEEK_END)
           if f.read(1) != b"\n":
               f.write(b"\n")

   norm = lambda s: s if s.endswith("\n") else s + "\n"
   rows = [norm(l) for l in open("eval.jsonl", encoding="utf-8") if l.strip()]
   ensure_trailing_newline("train.jsonl")
   with open("train.jsonl", "a", encoding="utf-8") as out:
       out.writelines(rows)
   os.remove("eval.jsonl")
   print(f"folded {len(rows)} rows from eval.jsonl back into train.jsonl; removed eval.jsonl")
   EOF
   wc -l train.jsonl
   ```

2. **Restrict to in-vocabulary rows, then sample 1,000 — do not read
   `train.jsonl` whole.** Only rows whose `feeling` **and** `emoji` both appear
   in `labels.json` are eligible. The filter parses each line only to test
   membership and writes the surviving line **verbatim**, so the byte-for-byte
   match in step 5 still holds.

   ```bash
   wc -l train.jsonl
   python3 - <<'EOF'
   import json

   labels = json.load(open("labels.json", encoding="utf-8"))
   feelings = set(labels["feelings"])
   emojis = set(labels["emojis"])
   kept = 0
   out = open("/tmp/eval_pool.jsonl", "w", encoding="utf-8")
   for line in open("train.jsonl", encoding="utf-8"):
       if not line.strip():
           continue
       row = json.loads(line)
       if row.get("feeling") in feelings and row.get("emoji") in emojis:
           out.write(line)
           kept += 1
   out.close()
   print(f"{kept} of the train.jsonl rows have both labels in labels.json")
   EOF
   wc -l /tmp/eval_pool.jsonl
   shuf -n 1000 /tmp/eval_pool.jsonl > /tmp/eval_candidates.jsonl
   wc -l /tmp/eval_candidates.jsonl
   ```

   Keep every candidate line **verbatim** — the kept lines are matched
   byte-for-byte against `train.jsonl` later, so do not re-serialize them.
   `eval.jsonl` no longer exists at this point (step 1 removed it), so there is
   nothing to dedup the sample against. If `/tmp/eval_pool.jsonl` holds fewer
   than 1,000 rows, `shuf` returns all of them — that is fine.

3. **Judge every candidate — in this session, in chunks of ~50.** Read
   `/tmp/eval_candidates.jsonl`. For each row, with the labels visible, decide:

   - **feeling** — is the row's `feeling` a correct label for `text`? It need
     not be the only word you would pick, but a reader seeing `text` alone must
     find it clearly right, not merely plausible. Mixed or contradicted signal
     → drop.
   - **emoji** — is the row's `emoji` a defensible best emoji for `text`? A
     reasonable person could attach it to that message. Off-topic or contrary
     → drop.
   - **length** — `4 <= len(text) <= 48` counting code points on the raw
     (un-normalized) `text`.

   Keep the row (its **exact original line**) only if all three hold. Do not
   fix, re-word, or re-label anything. Collect the kept lines into
   `/tmp/eval_kept.jsonl`, verbatim, one per line.

4. **Write the kept rows to a fresh `eval.jsonl`.**

   ```bash
   cp /tmp/eval_kept.jsonl eval.jsonl
   wc -l eval.jsonl /tmp/eval_kept.jsonl
   ```

5. **Remove the kept rows from `train.jsonl` — count-aware, byte-exact.**

   ```bash
   python3 - <<'EOF'
   from collections import Counter
   norm = lambda s: s if s.endswith("\n") else s + "\n"
   remove = Counter(norm(l) for l in open("/tmp/eval_kept.jsonl", encoding="utf-8") if l.strip())
   out, removed = [], 0
   for line in open("train.jsonl", encoding="utf-8"):
       if line.strip() and remove[line] > 0:
           remove[line] -= 1
           removed += 1
           continue
       out.append(line)
   open("train.jsonl", "w", encoding="utf-8").writelines(out)
   missing = sum(remove.values())
   print(f"removed {removed} from train.jsonl; unmatched kept lines: {missing}")
   EOF
   ```

   `unmatched kept lines` must be `0`. If not, a kept line was altered in
   step 3 — fix it to the original and re-run this step.

6. **Verify and report.**

   - `eval.jsonl` has exactly `wc -l /tmp/eval_kept.jsonl` rows (it was created
     fresh in step 4).
   - `train.jsonl`'s net change equals `(rows folded back in step 1) − (rows
     removed in step 5)`.
   - Every row in `eval.jsonl` has its `feeling` and `emoji` in `labels.json`
     (guaranteed by the step 2 filter — spot-check a few).
   - `tail -n 1 eval.jsonl | python3 -c "import json,sys; json.loads(sys.stdin.read())"`
     — last line is valid JSON.
   - Refresh the stats report: `bun run tools/data/stat.ts`.
   - Report: rows folded back from a prior `eval.jsonl` (0 if none), the
     in-vocabulary pool size, rows sampled, dropped by feeling / emoji / length,
     rows kept, and the old/new line counts of both files.
