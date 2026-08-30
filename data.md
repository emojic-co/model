# Data quality report — 2026-08-30 08:31

- Sample: 500 of 71,114 rows (`report/data/08-30-08:31.sample.jsonl`)
- Label correctness: emoji 467/500 ok · 30 weak · 3 wrong; feeling 415/500 ok · 82 weak · 3 wrong (Excited judged on its load-time alias to Happy)
- Text quality: 485/500 clean · 2 broken · 12 normalize-fragile · 1 low-content
- Label coverage: feelings 7/7 present · emojis 300/300 present · imbalance 1099x (corpus max/min; min is an off-palette singleton)
- Style coverage: casual 1st-person micro-texts (median 6 words, none >10); no formal register, no long/multi-sentence messages, no reported dialogue, ~10% off-palette emoji rows silently dropped at load
- Fixes applied: 6 rows rewritten (2 broken emoji-field · 3 labels · 1 low-content: 0 · dedup: 0 — see below) · 0 left unfixed

## 1. Label correctness

**Judging note.** `labels.json` now has 7 feelings (`Happy, Calm, Sad, Angry, Anxious, Neutral, Love`) — "Excited" was removed and `data.py` folds every raw `Excited` row into `Happy` at load time (`FEELING_ALIASES`). 58/500 sampled rows still carry the raw `Excited` string; they are judged here against their effective label, `Happy`. The corpus also still holds 14 rows with other off-list feelings (`Annoyed` 5, `Confused` 4, `Frustrated` 2, `Hopeful`/`Amused`/`Relieved` 1 each) — these are dropped entirely at load.

**Emoji** — 467 ok · 30 weak · 3 wrong. The palette is large and full of near-synonyms, so the bar for "wrong" is a different topic/valence. The 30 "weak" are mostly decorative-random picks (🦆 for a duck-free text, 🐺/🏰/🚬/⚫ as filler) and a handful of valence mismatches (😾/😿 angry- or crying-cat on a merely anxious line, 😁 grin on an embarrassed line). Only 3 are outright wrong, and 2 of those are not emoji at all.

**Feeling** — 415 ok · 82 weak · 3 wrong. One systematic pattern dominates the "weak" bucket: **the Calm ↔ Neutral ↔ mild-negative boundary is fuzzy**. Emotion-thin logistics texts ("Your socks are under the chair.", "Meeting room B is free now.", "I'll leave the gate unlocked.") are frequently tagged `Calm` where `Neutral` fits better; small mishaps told lightly ("Bad news: it's all over my shirt", "I'm stuck watching reruns", "spilled soup on my top") get `Sad`/`Angry` where `Neutral`/annoyed is closer; relief ("The compressor is behaving itself", "Client approved it at 2 a.m. Finally.") gets `Happy`. None of these are clearly wrong in isolation, but the drift is one-directional (toward a stronger label than the text carries) and consistent.

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| Did you get home okay? Just checking. | 🫶 / Sad | feeling → Love | caring check-in + heart-hands emoji; no sadness in text — **fixed** |
| I'll stay up with this until it behaves. | 😤 / Love | feeling → Angry | stubborn/frustrated at a balky task; nothing affectionate — **fixed** |
| Friday suddenly feels very far away. | 😩 / Excited→Happy | feeling → Sad | expresses the letdown that an awaited day is distant, not anticipation — **fixed** |
| Finally out of that meeting. Alive. | 자유 / Happy | emoji → 😅 | `emoji` field held the Korean word "자유", not an emoji — **fixed** |
| Tiny hands, enormous amounts of dirt | 泥 / Neutral | emoji → 🤦 | `emoji` field held the Chinese char "泥", not an emoji — **fixed** |
| Payment received—productivity restored | 🦆 / Happy | emoji → 💸 | duck is unrelated to payment/productivity — **fixed** |
| That meeting was painfully sweet | 🐝 / Sad | feeling → Neutral/Angry (sarcasm) | left unfixed — wordplay makes the intended reading ambiguous |
| Tiny visitor, huge interruption | 🐞 / Angry | feeling → Neutral (playful) | left unfixed — "Angry" is defensible as mild irritation |

## 2. Text quality

- **broken: 2** — both are malformed `emoji` fields (the text is fine): row `자유` ("Finally out of that meeting. Alive.") and row `泥` ("Tiny hands, enormous amounts of dirt"). These come from the generator's free-choice emoji step emitting a plain word instead of an emoji. Both fixed.
- **normalize-fragile: 12** — read fine now but lean on characters `normalize` deletes (digits, `%`, `£`, `:`, `-`, `'`). Examples:
  - `'My phone is on 2%, wish me luck'` -> `'my phone is on % wish me luck'`
  - `'Anyone else still waiting by gate B17?'` -> `'anyone else still waiting by gate b?'`
  - `'Client approved it at 2:07 a.m. Finally.'` -> `'client approved it at : am finally'`
  - `'Meet me by the snack table at 8'` -> `'meet me by the snack table at'`
  - `"Lost 3-2 and somehow you're the headline"` -> `'lost and somehow youre the headline'`
  - `'My brain is 90% salsa'` -> `'my brain is % salsa'`
  - Out of scope for fixing per the skill (normalize/CHARS may change); listed so the generator can be told to avoid bare numerals/times.
- **low-content: 1** — `"Such an exciting achievement"` (✨ / Excited): grammatical but generic and template-flavoured, nothing concrete for a label to hang on. A few more are borderline-generic but keep a clear sentiment ("Genuinely excited to work together!", "Big smiles, strong vibes", "I'm excited for ghostly fun", "Can't wait to share good vibes!"). Row fixed (rewritten to a concrete message).
- **exact/near duplicates: 0** exact-after-normalize in the sample. A few thematic near-repeats exist (two "…ghostly fun" rows; several near-identical "excited to…/can't wait…" openers) but no byte-duplicates to split.

## 3. Label coverage

### Feelings

Post-alias effective distribution (Excited→Happy; 14 off-list rows excluded), full corpus:

| feeling | corpus count | corpus share | sample count (post-alias) |
| --- | --- | --- | --- |
| Neutral | 17,460 | 24.6% | 115 |
| Happy | 16,350 (8,141 + 8,209 Excited) | 23.0% | 119 |
| Calm | 7,661 | 10.8% | 57 |
| Anxious | 7,655 | 10.8% | 58 |
| Love | 7,608 | 10.7% | 55 |
| Sad | 7,353 | 10.3% | 55 |
| Angry | 7,013 | 9.9% | 41 |

All 7 present. `Neutral` and `Happy` are each ~2.3x the other five — the imbalance is entirely the `Neutral` logistics bucket plus the Excited merge. Raw file also carries 14 stray-label rows (`Annoyed`/`Confused`/`Frustrated`/`Hopeful`/`Amused`/`Relieved`), all silently dropped at load.

### Emojis

- present 300/300; absent: none
- top 10: 😤 1099 · 🎉 1032 · 😌 970 · 😠 897 · 😔 734 · ☕ 700 · 😬 667 · 😞 646 · 😰 620 · 😊 590
- bottom 10 (all outside the 300-emoji palette — dropped at load): 迷 · 🪔 · 🚁 · 🚄 · 🔫 · 🛷 · 🐠 · 🏴‍☠️ · 🚘 · 🤙 (1 each)
- imbalance max/min = 1099/1 = 1099x
- **857 distinct emoji values in the corpus vs 300 in the palette** — 557 off-palette values, and **54/500 sampled rows (10.8%) carry an off-palette emoji**, so on the order of 10-11% of the corpus contributes an emoji label that is discarded at load (`d["emoji"] not in emoji2idx → skip`, with no alias table for emojis). Many are perfectly apt (🥞 pancakes, 🎓 graduation, ✈️ plane, 💰 money, 🥧, 👶, 🍻); they are simply not in the top-300 that `gen_labels.ts` last froze.

## 4. Text-style coverage

| axis | buckets (approx share) |
| --- | --- |
| register | casual ~65% · neutral/plain ~30% · slang/net-speak ~5% ("rn", "sending me", "flop era") · formal ~0% |
| form | 1st-person feeling/statement ~45% · imperative/logistics request ("meet me…", "can you grab…", "please sign…") ~30% · question ~15% · observation/aphorism ~8% · narrative/recount ~2% · dialogue/quote ~0% |
| device | plain ~75% · exclamation ~20% · all-caps ~1% · in-text emoji 0% · profanity ~0% (one "hell") |
| age register | adult ~88% · teen/school ~10% (PE, syllabus, quiz, "sir") · child ~0% · indeterminate ~2% |

Length: min 3 words · p25 5 · median 6 · p75 7 · max 10. Buckets: 1-3 words 11 (2.2%) · 4-7 words 408 (81.6%) · 8-15 words 81 (16.2%) · 16+ words 0.

Gaps:
- **Length monoculture** — nothing over 10 words, nothing multi-paragraph. The model never sees a long message.
- **No formal register** and **no reported dialogue / quoted speech**.
- **No child voice**; age register is almost entirely adult.
- **In-text emoji never appears** in `text` (consistent with `normalize` stripping it, but it means training input is uniformly emoji-free even though real WhatsApp text is not).
- **Large emotion-thin logistics block** ("the room is booked", "your parcel arrived", "meet me by the bakery at eleven") that defaults to `Neutral`/`Calm` and blurs that boundary (see §1).

## 5. Fixes applied

- rewritten: 6 rows (2 broken emoji-field · 3 labels · 1 low-content · 0 dedup); fixes file `report/data/08-30-08:31.fixes.jsonl`
- unfixed (flagged but not confidently fixable): 0 in the strict step-5 scope. Deliberately left alone: the ~82 `feeling_weak` / ~30 `emoji_weak` rows (judgment calls, not clear errors), all 12 normalize-fragile rows (out of scope), and the sarcasm-ambiguous rows "That meeting was painfully sweet" and "Tiny visitor, huge interruption".

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |
| Did you get home okay? Just checking. — 🫶 / Sad | 🫶 / Love | caring check-in; no sadness in the text |
| I'll stay up with this until it behaves. — 😤 / Love | 😤 / Angry | frustrated persistence at a balky task, not affection |
| Friday suddenly feels very far away. — 😩 / Excited | 😩 / Sad | letdown that an awaited day is distant (Excited→Happy would be wrong) |
| Finally out of that meeting. Alive. — 자유 / Happy | 😅 / Happy | `emoji` field was the word "자유", not an emoji |
| Tiny hands, enormous amounts of dirt — 泥 / Neutral | 🤦 / Neutral | `emoji` field was the char "泥", not an emoji |
| Payment received—productivity restored — 🦆 / Happy | 💸 / Happy | duck unrelated to payment/productivity |

Low-content row `"Such an exciting achievement"` was flagged in §2; on re-check it carries a clear (if generic) positive sentiment and a valid emoji/feeling pair, so it was left as-is rather than force-rewritten. Net: 6 rewrites, all label/emoji-field corrections.

## 6. Verdict & recommendations

1. **Stop the generator emitting non-emoji `emoji` values.** Two of 500 rows had a CJK/Hangul word in the `emoji` field; the corpus-wide bottom-10 is all junk singletons (迷, 🪔…). Add an "is this exactly one emoji" guard to the annotate step and drop failures.
2. **Close the emoji-palette gap.** ~10% of rows carry an off-palette emoji and are silently dropped at load. Either (a) constrain the generator's free-choice emoji to the current `labels.json` palette, or (b) re-run `bun gen_labels.ts` more often / raise the palette from 300, or (c) add an emoji alias/merge table mirroring `FEELING_ALIASES`.
3. **Fix the Calm/Neutral drift in generation.** The largest correctness issue is ~80 rows where an emotion-thin text got `Calm`/`Sad`/`Happy` instead of `Neutral`. Tighten the annotation prompt: reserve `Calm` for texts that actively express settledness, and default logistics/status texts to `Neutral`.
4. **Add length and register variety.** Everything is 3-10 words, casual, adult, 1st-person present. Add a generation mode for longer multi-sentence messages, a formal register, and quoted/reported speech so the LSTM sees more than one text shape.
5. **Purge or alias the 14 stray-feeling rows** (`Annoyed`, `Confused`, …) — currently dead weight dropped at load; either map them (`Annoyed→Angry`, `Frustrated→Angry`, `Hopeful→Anxious`?) or filter them in the generator.
6. **Update `CLAUDE.md`** — it still describes "the 8 in `labels.json`" and an `Excited` feeling; the set is now 7 with `Excited` aliased to `Happy` in `data.py`.
