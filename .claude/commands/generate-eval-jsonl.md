---
description: Rebuild eval.jsonl from scratch — fold any existing eval.jsonl back into train.jsonl, then sample 1,000 rows whose feeling and emoji are both in labels.json and which carry a full bg/fg palette, and keep only the ones whose labels and colors are all correct
allowed-tools: [Bash, Read, Write, Edit]
---

# Generate eval.jsonl

Rebuild the gold holdout `eval.jsonl` from scratch. If an `eval.jsonl` already
exists, every one of its rows is folded **back into** `train.jsonl` first, so the
new holdout is drawn from the full corpus and never silently accumulates stale
rows. Then sample **only from rows whose `feeling` and `emoji` are both present
in `labels.json`** — those are exactly the labels the model can predict, so a row
carrying an out-of-vocab label could never be scored in the validation step —
keep only rows whose **both** labels are correct, whose `bg`/`fg` palette is
present, valid, and on-mood, and whose text length is in range, and move every
kept row **out** of `train.jsonl` so the holdout never overlaps the training
corpus.

Labels are open-set: `feeling` is any single capitalized word, `emoji` is any
emoji, and the palette (`bg` = two `#rrggbb` strings, `fg` = one `#rrggbb`) is
free-form. You are judging whether the existing labels are right — **never
rewrite a label or a color**. A row is keep-or-drop only.

Text-length range: **4–48 code points** on the raw `text` (no normalization).

## Steps

1. **Fold an existing `eval.jsonl` back into `train.jsonl`.** Skip this step if
   `eval.jsonl` does not exist. Record how many rows were folded back — you
   report it at the end.

   The combined corpus is built in `train.jsonl.tmp` and swapped in with a single
   atomic `os.replace`, so a crash can never leave `train.jsonl` truncated or
   half-appended. `eval.jsonl` is removed only after the swap succeeds.

   ```bash
   wc -l train.jsonl
   test -f eval.jsonl && python3 - <<'EOF' || echo "no eval.jsonl — nothing to fold back"
   import os

   norm = lambda s: s if s.endswith("\n") else s + "\n"
   rows = [norm(l) for l in open("eval.jsonl", encoding="utf-8") if l.strip()]

   with open("train.jsonl", encoding="utf-8") as f:
       train = f.read()
   if train and not train.endswith("\n"):
       train += "\n"

   with open("train.jsonl.tmp", "w", encoding="utf-8") as out:
       out.write(train)
       out.writelines(rows)
   os.replace("train.jsonl.tmp", "train.jsonl")
   os.remove("eval.jsonl")
   print(f"folded {len(rows)} rows from eval.jsonl back into train.jsonl; removed eval.jsonl")
   EOF
   wc -l train.jsonl
   ```

2. **Restrict to in-vocabulary rows with a full palette, then sample 1,000 — do
   not read `train.jsonl` whole.** A row is eligible only when its `feeling`
   **and** `emoji` both appear in `labels.json` **and** it carries a two-element
   `bg` list plus a string `fg` (rows folded back from a pre-palette `eval.jsonl`
   lack these and drop out here). The filter parses each line only to test
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
       has_palette = (
           isinstance(row.get("bg"), list)
           and len(row["bg"]) == 2
           and isinstance(row.get("fg"), str)
       )
       if row.get("feeling") in feelings and row.get("emoji") in emojis and has_palette:
           out.write(line)
           kept += 1
   out.close()
   print(f"{kept} of the train.jsonl rows have both labels in labels.json and a full palette")
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
   - **colors** — `bg` is two `#rrggbb` strings that read as one gradient (not a
     clash), `fg` is one `#rrggbb` that stays clearly readable over both `bg`
     stops, and the palette plausibly reflects the mood or imagery of `text`.
     Malformed hex, garish, unreadable, or off-mood → drop.
   - **length** — `4 <= len(text) <= 48` counting code points on the raw
     (un-normalized) `text`.

   Every candidate goes to **exactly one** of two files, verbatim, one line
   each: `/tmp/eval_kept.jsonl` if all four hold, `/tmp/eval_dropped.jsonl`
   otherwise. Do not fix, re-word, re-label, or restyle anything. Judge every
   chunk — none skipped.

4. **Check the partition, then write a fresh `eval.jsonl`.** `kept ∪ dropped`
   must equal the candidate set exactly — this is what proves no candidate was
   silently missed or judged twice.

   ```bash
   sort /tmp/eval_candidates.jsonl > /tmp/eval_cand_sorted.jsonl
   cat /tmp/eval_kept.jsonl /tmp/eval_dropped.jsonl | sort > /tmp/eval_judged_sorted.jsonl
   if cmp -s /tmp/eval_judged_sorted.jsonl /tmp/eval_cand_sorted.jsonl; then
     echo "partition OK — every candidate judged exactly once"
     cp /tmp/eval_kept.jsonl eval.jsonl
     wc -l eval.jsonl /tmp/eval_kept.jsonl /tmp/eval_dropped.jsonl /tmp/eval_candidates.jsonl
   else
     echo "MISMATCH — kept+dropped != candidates; return to step 3, do NOT continue"
   fi
   ```

   If the partition check fails, do not proceed — go back to step 3, find the
   missing or duplicated candidate, and rebuild both files.

5. **Remove the kept rows from `train.jsonl` — count-aware, byte-exact, atomic.**
   `train.jsonl` is copied to `train.jsonl.bak` before the rewrite, the new
   content is written to `train.jsonl.tmp`, and `os.replace` swaps it in as one
   step — a crash leaves either the untouched original or the `.bak`, never a
   truncated file.

   ```bash
   python3 - <<'EOF'
   import os, shutil
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
   shutil.copyfile("train.jsonl", "train.jsonl.bak")
   with open("train.jsonl.tmp", "w", encoding="utf-8") as f:
       f.writelines(out)
   os.replace("train.jsonl.tmp", "train.jsonl")
   missing = sum(remove.values())
   print(f"removed {removed} from train.jsonl; unmatched kept lines: {missing}; backup at train.jsonl.bak")
   EOF
   ```

   `unmatched kept lines` must be `0`. If not, a kept line was altered in
   step 3 — restore from `train.jsonl.bak`, fix the line to its original, and
   re-run this step.

6. **Verify and report.**

   - `eval.jsonl` has exactly `wc -l /tmp/eval_kept.jsonl` rows (it was created
     fresh in step 4).
   - The step 4 partition check printed `partition OK` — `kept + dropped` equals
     the candidate count, so no candidate was skipped.
   - `train.jsonl`'s net change equals `(rows folded back in step 1) − (rows
     removed in step 5)`.
   - No `train.jsonl.tmp` is left behind; `train.jsonl.bak` holds the pre-removal
     corpus.
   - Every row in `eval.jsonl` has its `feeling` and `emoji` in `labels.json`
     and a two-stop `bg` plus an `fg` (guaranteed by the step 2 filter —
     spot-check a few).
   - `tail -n 1 eval.jsonl | python3 -c "import json,sys; json.loads(sys.stdin.read())"`
     — last line is valid JSON.
   - Refresh the stats report: `bun run tools/data/stat.ts`.
   - Report: rows folded back from a prior `eval.jsonl` (0 if none), the
     in-vocabulary pool size, rows sampled, dropped by feeling / emoji / colors
     / length, rows kept, and the old/new line counts of both files.
