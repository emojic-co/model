---
description: Sample 1,000 rows from train.jsonl, keep only the correctly-labeled ones as eval.jsonl, and remove those from train.jsonl
allowed-tools: [Bash, Read, Write, Edit]
---

# Generate eval.jsonl

Build (or grow) the gold holdout `eval.jsonl` by sampling `train.jsonl`, keeping
only rows whose **both** labels are correct and whose text length is in range,
and moving every kept row **out** of `train.jsonl` so the holdout never overlaps
the training corpus.

Labels are open-set: `feeling` is any single capitalized word, `emoji` is any
emoji. You are judging whether the existing labels are right — **never rewrite a
label**. A row is keep-or-drop only.

Text-length range: **4–48 code points** on the raw `text` (no normalization).

## Steps

1. **Sample 1,000 rows — do not read `train.jsonl` whole.**

   ```bash
   wc -l train.jsonl
   shuf -n 1000 train.jsonl > /tmp/eval_sample.jsonl
   wc -l /tmp/eval_sample.jsonl
   ```

   Keep every line **verbatim** — the kept lines are matched byte-for-byte
   against `train.jsonl` later, so do not re-serialize them.

2. **Drop rows already in `eval.jsonl`** (guard for re-runs; skip if the file
   does not exist yet).

   ```bash
   test -f eval.jsonl && python3 - <<'EOF' || cp /tmp/eval_sample.jsonl /tmp/eval_candidates.jsonl
   import json
   seen = {json.loads(l)["text"] for l in open("eval.jsonl", encoding="utf-8") if l.strip()}
   with open("/tmp/eval_sample.jsonl", encoding="utf-8") as f, \
        open("/tmp/eval_candidates.jsonl", "w", encoding="utf-8") as out:
       kept = dropped = 0
       for line in f:
           if not line.strip():
               continue
           if json.loads(line)["text"] in seen:
               dropped += 1
               continue
           out.write(line if line.endswith("\n") else line + "\n")
           kept += 1
   print(f"candidates {kept}, already-in-eval {dropped}")
   EOF
   ```

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

4. **Append kept rows to `eval.jsonl`** (create if missing, never truncate).

   ```bash
   wc -l eval.jsonl 2>/dev/null || echo "eval.jsonl: 0 (new)"
   cat /tmp/eval_kept.jsonl >> eval.jsonl
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

   - `eval.jsonl` grew by exactly `wc -l /tmp/eval_kept.jsonl`.
   - `train.jsonl` shrank by the same amount.
   - `tail -n 1 eval.jsonl | python3 -c "import json,sys; json.loads(sys.stdin.read())"`
     — last line is valid JSON.
   - Refresh the stats report: `bun run tools/data/stat.ts`.
   - Report: rows sampled, dropped as already-in-eval, dropped by feeling /
     emoji / length, rows kept, and the old/new line counts of both files.
