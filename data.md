# Data quality report — 2026-08-28 19:49

- Sample: 500 of 13754 rows (`report/data/08-28-19:49.sample.jsonl`)
- Label correctness: emoji 480/500 ok · 18 weak · 2 wrong; feeling 433/500 ok · 66 weak · 1 wrong
- Text quality: 476/500 clean · 1 broken · 21 normalize-fragile · 2 low-content
- Label coverage: feelings 8/8 present (+10 corpus rows carry off-vocab feelings, dropped at load) · emojis 134/134 palette present (corpus holds 456 distinct; 322 off-palette classes dropped at load) · imbalance 501x
- Style coverage: casual 1st-person present-tense feeling statements dominate; biggest gaps are dialogue/quotes, ALL-CAPS, and any message longer than ~10 words
- Fixes applied: 6 rows rewritten (1 broken · 2 labels · 2 low-content · 1 dedup) · 1 left unfixed

## 1. Label correctness

Rates: emoji 480 ok / 18 weak / 2 wrong; feeling 433 ok / 66 weak / 1 wrong. The
palette is large and full of near-synonyms, so emoji fit is strong — the two
outright wrong emojis are 🤔 on an angry line and 🥳 on a heartbreak line.

Feeling is the weak head. **66/500 (13%) feelings are defensible-but-not-best,
and the pattern is one-directional: deadpan / wry one-liners get one of the six
strong feelings when the actual tone is mild or affect-free.** The annotator
under-selects `Neutral`. Recurring shapes:

- Mild disappointment or being unimpressed → `Angry` ("I'm not impressed with
  the DJ", "not impressed by the replacement bus tbh", "New phone, same
  nonsense").
- Wry / mock-tragic complaints → `Sad` ("Payday feels suspiciously far away.",
  "Breaking: cafeteria has run out of samosas.", "The vet bill costs more than
  my textbook.", "Apparently, sleep is optional.").
- Flat observations or neutral logistics → `Anxious` ("I'm bringing a jacket.",
  "It's sunny but somehow still freezing", "The next one is delayed.", "Truck
  arrived early.").
- Polite acknowledgements → `Happy` ("Noted, nice work", "Okay, well done",
  "Understood, good work", "Sure, well done") — a fixed template that is really
  `Neutral`.
- "Wasn't excited tbh…" / "Not excited tbh…" openers → `Sad` when the content
  is neutral or even positive ("Wasn't excited tbh, but it was great").

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| My heart's broken, yay | 🥳 / Happy | 💔 / Sad | sarcasm; conveyed emotion is not happiness — **fixed** |
| Why'd you pick this spot | 🤔 / Angry | 😒 / Angry | 🤔 is neutral-valence, clashes with the Angry label — **fixed** |
| I'm not impressed with the DJ | 😤 / Angry | 😐 / Neutral | unimpressed ≠ raging (no `Annoyed` in vocab) |
| Truck arrived early. | 🚚 / Happy | 🚚 / Neutral | plain logistics, no affect |
| I'm bringing a jacket. | 🧥 / Calm | 🧥 / Neutral | statement of fact |
| Breaking: cafeteria has run out of samosas. | 😞 / Sad | 😤 / Angry or Neutral | mock-news bit, not grief |
| Noted, nice work | 😊 / Happy | 🙂 / Neutral | acknowledgement template |
| Apparently, sleep is optional. | 😴 / Sad | 😴 / Neutral | weary wry, not sad |

## 2. Text quality

- broken: 1 — `"Sad heart at this place"` (not natural English; templated
  emotion-slotting). **fixed** → "Being back here just makes my heart hurt".
  Also watched but **not** fixed: `"Nice and mellow, just chili"` — "chili" is
  probably a typo for "chill" but could be an intentional food reference;
  low confidence, left untouched.
- normalize-fragile: 21 — meaning rides on digits that `normalize` deletes.
  `"Mum booked the 6am flight. Absolute betrayal."` → `"mum booked the am
  flight absolute betrayal"`; `"Made it onto the 8:12. Miracles happen."` →
  `"made it onto the : miracles happen"`; `"My phone is on 2 percent"` → `"my
  phone is on percent"`; `"Score's 1-0, but my spreadsheet is still losing."`
  → `"scores but my spreadsheet is still losing"`. Per skill rules these were
  **not** edited (normalize/CHARS may change).
- low-content: 2 — `"This belongs there"` (vague, no concrete or emotional
  hook), `"Keeping it neutral"` (tautological with its label). Both **fixed**
  into concrete Neutral messages.
- exact/near duplicates: 0 exact-after-normalize in the sample. One near-dup
  pair split: `"I’m so excited for donuts"` / `"I’m so excited for donuts!"`
  (both 🍩 / Excited) — the `!` copy was rewritten.

## 3. Label coverage

### Feelings

| feeling | corpus count | corpus share | sample count |
| --- | --- | --- | --- |
| Anxious | 2651 | 19.3% | 96 |
| Happy | 2465 | 17.9% | 93 |
| Sad | 1944 | 14.1% | 69 |
| Neutral | 1833 | 13.3% | 65 |
| Angry | 1778 | 12.9% | 66 |
| Excited | 1489 | 10.8% | 44 |
| Calm | 1407 | 10.2% | 62 |
| Love | 177 | 1.3% | 5 |
| _off-vocab (Annoyed 4, Confused 4, Frustrated 1, Hopeful 1)_ | 10 | 0.07% | 0 |

`Love` is ~14x rarer than any other feeling — effectively untrainable as its
own class. 10 rows carry feelings not in `labels.json` (`Annoyed`, `Confused`,
`Frustrated`, `Hopeful`) and are silently dropped by `read()` at load;
`Annoyed` and `Frustrated` in particular would be useful given the Angry
over-labeling in §1.

### Emojis

- palette present 134/134; absent: none.
- corpus contains 456 distinct emoji classes — **322 are not in the 134-emoji
  palette and are dropped at load**, discarding real annotation labour.
- top 10: 😌 501 · 😤 490 · 😰 472 · 😠 420 · 😟 396 · 🎉 382 · 😬 362 · 😊 351 · 😞 332 · 😔 317
- bottom 10 (all off-palette singletons, dropped at load): 🍎 🤓 📮 🏙️ 😇 💝 🪙 🦊 👓 🃏 — each count 1
- imbalance max/min = 501/1 = 501x (inflated by off-palette singletons; within
  the 134-emoji palette the spread is smaller but still steep — a handful of
  face emojis carry most rows and the topical/object emojis are thin).

## 4. Text-style coverage

| axis | buckets (approx share) |
| --- | --- |
| register | formal ~2% · neutral ~30% · casual ~60% · slang/net-speak ~8% (tbh, rn, lowkey, deadass, u, "gooo", "slaps") |
| form | 1st-person feeling statement ~45% · narrative/recount ~20% · observation/aphorism ~15% · question ~20% · dialogue/quote ~0% |
| device | plain ~72% · exclamation ~24% · profanity ~2% (mild only: "pissed", "hell") · ALL-CAPS 0% · in-text emoji 0% |
| age register | child 0% · teen ~10% · adult ~65% · indeterminate ~25% |

Gaps:

- **No dialogue or quoted speech** anywhere in the sample.
- **No ALL-CAPS emphasis and no in-text emoji** — two of the most common real
  texting devices are entirely absent (partly by generator design, but it
  makes the training distribution unlike real WhatsApp text).
- **No long messages** — median 5 words, max 10; no multi-sentence venting.
- **Almost no formal register**, no child voice, thin teen voice.
- **Present-tense monoculture** — nearly every row is "right now"; little
  past-tense recounting of a finished event, no forward planning beyond a time.
- **Heavy template frames**: "Feeling X, Y-ing Z", "X makes me happy/furious",
  "Can't say I'm sad …", "Wasn't excited tbh, but …", "Not happy about X", "not
  calm rn, the X", "Noted/Okay, <praise>".
- **Narrow topic set**: coffee, tacos, donuts, pizza, printers, weddings,
  sports matches, harvest/farm, commuting, hospital shifts recur constantly.

## 5. Fixes applied

- rewritten: 6 rows (1 broken · 2 labels · 2 low-content · 1 dedup); fixes file
  `report/data/08-28-19:49.fixes.jsonl`
- unfixed (flagged but not confidently fixable): 1 — `"Nice and mellow, just
  chili"` (probable "chill" typo, but possibly deliberate); plus the 66 `_weak`
  feelings and 21 normalize-fragile rows, which are out of scope by the skill's
  rules.

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |
| Sad heart at this place — 💔 / Sad | Being back here just makes my heart hurt — 💔 / Sad | broken: unnatural templated phrasing |
| This belongs there — 📍 / Neutral | The parcel goes to the flat upstairs, not ours — 📍 / Neutral | low-content: no concrete or emotional hook |
| Keeping it neutral — 😐 / Neutral | The meeting ran long but nothing got decided — 😐 / Neutral | low-content: tautological with label |
| Why'd you pick this spot — 🤔 / Angry | Why'd you pick this spot — 😒 / Angry | emoji wrong: 🤔 is neutral-valence, clashes with Angry |
| My heart's broken, yay — 🥳 / Happy | My heart's broken, yay — 💔 / Sad | labels wrong: sarcasm, conveyed emotion is not happiness |
| I’m so excited for donuts! — 🍩 / Excited | Bakery opens in ten and I’m first in line — 🍩 / Excited | near-duplicate of "I’m so excited for donuts" |

## 6. Verdict & recommendations

1. **Fix the annotator's feeling bias toward the six strong labels.** 13% of
   feelings are over-strong; give `Neutral` explicit priority for wry,
   logistical, and acknowledgement texts, and add the "Wasn't excited tbh" /
   "Noted, nice work" templates as `Neutral` exemplars in the annotation prompt.
2. **Decide what to do about `Love` (1.3%) and the 4 off-vocab feelings.**
   Either drop `Love` from the head, or have the generator/annotator target it
   (and `Annoyed` / `Frustrated`, which the Angry over-labeling shows are
   needed) so every class is trainable.
3. **Reconcile the emoji palette.** `labels.json` has 134 emojis (CLAUDE.md
   says top 100) and 322 further emoji classes in `data.jsonl` are dropped at
   load. Re-run `gen_labels.ts`, or widen the cutoff, so annotation labour
   isn't discarded; consider constraining the annotator to the palette.
4. **Break the style monoculture in `raw_txt.ts`.** Add dialogue/quoted
   messages, ALL-CAPS emphasis, longer multi-sentence vents, past-tense
   recounts, a formal register, and a teen/child voice; widen the topic spread
   beyond food/printers/weddings/sports.
5. **Stop generating digit-dependent meaning.** ~4% of rows lose their point
   after `normalize` strips digits (times like "6am"/"the 8:12", scores "1-0",
   "2 percent", room/page numbers). Have the generator spell numbers out or
   avoid leaning on them.
6. Minor: teach the generator that emotion-slot templates ("Sad heart at this
   place", "Calm feels nice") read as artifacts, not messages.
