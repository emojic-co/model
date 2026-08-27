---
description: Append 500 new synthetic samples to data.jsonl, seeded from 50 random existing rows
allowed-tools: [Bash, Read, Write]
---

# Add samples

Grow `data.jsonl` by **500 new samples** without ever reading the whole file.

## Steps

1. **Sample 50 seed rows** — do not read `data.jsonl` directly. Run:

   ```bash
   shuf -n 50 data.jsonl
   ```

   (fallback if `shuf` is unavailable: `sort -R data.jsonl | head -n 50`).
   These 50 rows are your only view of the dataset — use them to infer the
   schema, value ranges, tone, and label distribution.

2. **Study the format.** Each line is a JSON object with exactly these keys:
   - `text`: short lowercase-ish English phrase (roughly 1–10 words), only
     characters from `main.py`'s `vocab`.
   - `emoji`: a single emoji drawn from the `emojis` set in `main.py`.
   - `feeling`: one of `Happy`, `Excited`, `Calm`, `Sad`, `Angry`, `Anxious`.
   - `bg1`, `bg2`: Oklab `[L, a, b]` gradient colors — `L` in `0..1`,
     `a`/`b` in `-0.4..0.4`.
   - `text_color`: Oklab `[L, a, b]` for readable text over that gradient
     (high contrast vs. `bg1`/`bg2` — usually near-white on dark, near-black
     on light).

3. **Generate 500 new samples** that match the seed rows' style and
   conventions:
   - New `text` values, not copies of the seeds. Vary phrasing, topics, and
     which `feeling` they map to; keep the label mix roughly balanced across
     all 6 feelings.
   - Pick an `emoji` that fits the text and feeling.
   - Choose `bg1`/`bg2` that suit the mood (warm/bright for Happy/Excited,
     cool/muted for Calm/Sad, dark/saturated red for Angry, desaturated for
     Anxious) and a `text_color` with clear contrast against them.
   - Keep every value inside the ranges above; `text` must only use `vocab`
     characters.

4. **Append, don't overwrite.** Write all 500 objects as one JSON object per
   line to a temp file, then append:

   ```bash
   cat /tmp/new_samples.jsonl >> data.jsonl
   ```

   Do not truncate or rewrite `data.jsonl`.

5. **Verify:** run `wc -l data.jsonl` and confirm it grew by exactly 500, and
   `tail -n 1 data.jsonl | python3 -c "import json,sys; json.loads(sys.stdin.read())"`
   to confirm the last line is valid JSON. Report the old and new line counts.
