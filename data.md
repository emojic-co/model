# Data quality report — 2026-08-28 12:56

- Sample: 500 of 13537 rows (`report/data/08-28-12:56.sample.jsonl`)
- Label correctness: emoji 478/500 ok · 22 weak · 0 wrong; feeling 428/500 ok · 70 weak · 2 wrong
- Text quality: 469/500 clean · 1 broken · 15 normalize-fragile · 15 low-content
- Label coverage: feelings 8/8 present · emojis 133/133 present · imbalance 74.5x
- Style coverage: casual first-person WhatsApp one-liners in a single deadpan-irony voice; biggest gap is register/form monoculture (no formal writing, no long messages, no dialogue, one age register) plus a stub `Love` class (1.4%).

## 1. Label correctness

Rates (500 rows, judged in 10 batches of 50):

- **feeling**: 428 ok · 70 weak · 2 wrong
- **emoji**: 478 ok · 22 weak · 0 wrong

**Systematic patterns**

1. **Deadpan-irony house style inflates the feeling label.** A large share of rows are
   wry understatement ("Lunch is happening beside the shed, very glamorous.",
   "apparently I was optimistic", "Naturally."). The assigned feeling captures the
   *intended* undertone (annoyed→Angry, worried→Anxious, flat→Sad), which is
   usually defensible but a plain reader would often say `Neutral`. This is the
   source of nearly every one of the 70 "weak" feelings — it is a style problem,
   not scattered labeling noise.
2. **`Angry` over-applied to mild sarcasm / sports gripes.** Rows like "Your
   goodnight arrived at 5:12, very punctual" (😒/Angry), "The kick sounds like a
   cupboard falling downstairs" (😤/Angry), "who approved that chorus" (🙄/Angry)
   are irritation-flavored banter, not anger. `😒 🙄 😤` + `Angry` is the recurring
   template.
3. **Literal-pun emojis on non-matching feelings.** `🥶` is attached to
   "not impressed by these cold wings", "sunny but somehow still freezing",
   "Frost on the van this morning" — the emoji tracks the word "cold/frost",
   while the feeling label (Sad/Anxious) is a stretch. Same shape with `☕` +
   `Anxious` for "need coffee" rows.
4. **`Love` is barely exercised** (7 rows) and every one is a generic
   send-off ("Sending big hugs", "Sending moral support", "All is well, love") —
   no examples of affection toward a person in a concrete situation.

Worst individual rows:

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| Do we still have cumin, or did I dream that? | 🤔 / Anxious | 🤔 / Neutral | plain domestic question, zero anxiety — **wrong** |
| It's sunny but somehow still freezing | 🥶 / Anxious | 🌤️ / Neutral | weather observation, no affect — **wrong** |
| Not feeling this gloomy art | 🌧️ / Sad | 🙄 / Neutral | dismissive/bored, not sad |
| The cover promised drama. So far, mostly weather. | 🌧️ / Sad | 😐 / Neutral | mild disappointment / boredom |
| Your goodnight arrived at 5:12, very punctual | 😒 / Angry | 🙂 / Neutral | affectionate sarcasm, not anger |
| Call me after the finale, I need to discuss that betrayal. | 😡 / Angry | 🤯 / Excited | excited about a show, not angry |
| I'm not impressed by these cold wings. | 🥶 / Sad | 😒 / Angry | complaint/annoyance; emoji is a literal-cold pun |
| Hope your birthday is quieter than tonight's admissions. | 🤫 / Happy | 😮‍💨 / Neutral | tired healthcare worker, not happy |
| That certainly stands out | ✨ / Happy | 👀 / Neutral | bare observation, no valence |
| A rainbow showed up | 🌈 / Excited | 🌈 / Calm | pleasant, low-arousal |
| Hope luck pulls us through | 🍀 / Excited | 🤞 / Anxious | anxious hope, not excitement |
| That was unexpectedly lovely | 🥰 / Love | 😊 / Happy | pleasant surprise, not love |

## 2. Text quality

- **broken: 1** — only marginal case: `"Error: I'm seriously mad!"` (row 33) —
  the `Error:` prefix reads like a template/roleplay artifact. No JSON crumbs,
  `{feeling}` placeholders, truncation, or non-English anywhere in the sample.
- **normalize-fragile: 15** — rows whose meaning rides on digits `normalize`
  deletes (`CHARS` has no `0-9`, no `,` `.` `'` `-`). All 15 casualties are
  times or counts:
  - `Support said to wait 48 hours. It's been 49.` -> `support said to wait  hours its been ` (the whole joke is gone)
  - `Please don't spoil page 200. I'm only on 37.` -> `please dont spoil page  im only on `
  - `My appointment got moved to 4:30. Free after that?` -> `my appointment got moved to : free after that?`
  - `Driver says 5 mins, which means 25 in delivery time.` -> `driver says  mins which means  in delivery time`
  - `Score's 1-0, but my spreadsheet is still losing.` -> `scores  but my spreadsheet is still losing`
  - plus `5:15`, `5:12`, `8:30`, `5:30`, `3pm`, `3;`, `track 7`, `2,300 words`, `At 17.`
- **low-content: 15** — grammatical but nothing a label can attach to:
  `I'm at this spot` · `I'm at the computer` · `I'm outside` · `Watching quietly` ·
  `Reading for class` · `Reading something on screen` · `Sounds right to me` ·
  `Acknowledged, moving forward` · `Sure, that's fine` · `Everything seems in order` ·
  `AI is checking` · `I see what happened` · `It's pouring outside` ·
  `The lights flickered` · `That is worth noticing`. These cluster hard on
  `Neutral` and `Calm`.
- **exact/near duplicates: 0** extra rows after `normalize` in the sample
  (0 collisions across 500).
- Uniform apostrophe loss (`I'll`->`ill`, `don't`->`dont`, `heart's`->`hearts`)
  affects a large fraction of rows. It is symmetric train/inference so it is not
  corruption, but `ill` (for "I'll") now collides with `ill` (sick) — worth a
  note, not a fix.

## 3. Label coverage

### Feelings

| feeling | corpus count | corpus share | sample count |
| --- | --- | --- | --- |
| Anxious | 2757 | 20.4% | 92 |
| Happy | 2533 | 18.7% | 93 |
| Sad | 1975 | 14.6% | 75 |
| Angry | 1902 | 14.0% | 80 |
| Excited | 1486 | 11.0% | 68 |
| Neutral | 1390 | 10.3% | 45 |
| Calm | 1304 | 9.6% | 40 |
| Love | 190 | 1.4% | 7 |

8/8 present, no off-vocab labels. The 7 strong feelings are within ~2x of each
other; `Love` is a 14x-underrepresented stub.

### Emojis

- present 133/133; absent: **none**
- top 10: `😤`596 · `😰`492 · `😬`468 · `🎉`420 · `😌`414 · `😠`399 · `😟`379 · `😊`362 · `😞`347 · `😔`324
- bottom 10: `🍨`20 · `😿`20 · `🚆`19 · `💸`19 · `🍼`18 · `😏`17 · `🪑`15 · `🥶`14 · `🎟️`13 · `😑`8
- imbalance max/min = 596/8 = **74.5x**

Face emojis for the high-frequency negative feelings dominate; object/scene
emojis (`🎟️ 🚆 💸 🍼 🪑`) sit in a long thin tail.

## 4. Text-style coverage

| axis | buckets (approx share) |
| --- | --- |
| register | formal ~1% · neutral ~25% · casual ~65% · slang/net-speak ~9% |
| form | 1st-person feeling ~45% · narrative/recount ~30% · question ~15% · observation/aphorism ~8% · dialogue/quote ~2% |
| device | plain ~62% · exclamation ~35% · profanity ~2% (mild: "damn", "hell", "pissed") · all-caps ~0% · in-text emoji 0% |
| age register | adult ~75% · indeterminate ~17% · teen ~8% · child ~0% · elderly (1st-person) ~0% |

Length: min 2 · p25 4 · median 5 · p75 8 · max 13 words. Buckets: 1-3 words 70,
4-7 words 286, 8-15 words 144, 16+ words 0.

**Gaps**

- **No formal register at all** — no work email, no announcement, no customer-service
  or official tone. Everything is a mate texting a mate.
- **One narrative voice.** Deadpan ironic understatement ("very glamorous.",
  "Naturally.", "apparently I was optimistic", "medically significant") recurs in
  dozens of rows and reads as a single author. Style monoculture even where each
  row is individually fine.
- **Length is pinned to 4-8 words.** No 3+ sentence messages, no rambling vents,
  nothing over 13 words in the sample.
- **No device variety.** Zero ALL-CAPS emphasis, zero in-text emoji, zero
  letter-elongation, profanity only the mildest tokens.
- **Age monoculture.** Adult professional / parent life (invoices, landlords,
  ranked matches, medication rounds, freelance). No children, almost no teens,
  no first-person elderly.
- **Form skew to present-tense first-person statement + "Update:" recounts.**
  Dialogue and aphorism barely present.
- **Emotional intensity is narrow.** Almost everything is mild-to-moderate;
  extreme joy/distress rows are rare and formulaic ("I'm furious", "so excited").
- `Love` is effectively unpopulated and the 7 examples are all generic
  send-offs.

## 5. Verdict & recommendations

1. **Fix the `normalize` / digit collision (generator side).** 3% of rows lose
   their point when digits are stripped. Cheapest fix: instruct `gen_data.ts` to
   avoid digits (spell "half four", "page two hundred", "forty-eight hours") — or
   widen `CHARS` to include `0-9` and retrain. Until then these rows are training
   noise.
2. **Break the style monoculture (generator prompt).** Explicitly request a mix
   of registers (formal notes, blunt statements, plain logistics), lengths
   (some 15-40 word multi-sentence messages), forms (dialogue, aphorism), and
   age registers (teen, older adult). Cap the "wry understatement" template to a
   minority. This is the single highest-impact change for model generality.
3. **Rebalance `Love` or drop it.** At 1.4% it cannot train a head reliably.
   Either target ~10% in the generator with concrete, situated affection texts,
   or remove it from `labels.json`.
4. **Tighten the `Angry` vs `Neutral`/`Excited` boundary.** Add generator
   guidance: sarcasm and sports gripes are `Neutral` unless there is real heat;
   show-related intensity ("that betrayal", "the finale") is usually `Excited`.
   Consider a light relabel pass over existing `😒 🙄 😤 + Angry` rows.
5. **Suppress low-content rows.** ~3% are `I'm outside` / `Sounds right to me` —
   filter texts under ~4 content words at generation time, or route them all to
   `Neutral` deliberately rather than letting them smear across `Calm`/`Neutral`.
6. **Grow the thin emoji tail.** 74.5x imbalance with object emojis
   (`🎟️ 🚆 💸 🍼 🪑 😑`) in single/low-double digits corpus-wide. Seed
   `gen_data.ts` batches toward the bottom quartile of `emojis` until the tail
   clears ~100 each.
