# Data quality report — 2026-08-30 09:56

- Sample: 500 of 77,280 rows (`report/data/08-30-09:56.sample.jsonl`)
- Label correctness: emoji 475/500 ok · 20 weak · 5 wrong; feeling 440/500 ok · 58 weak · 2 wrong
- Text quality: 493/500 clean · 0 broken (text) · 5 normalize-fragile · 1 low-content · 1 non-English emoji field
- Label coverage: feelings 7/7 present · emojis 300/300 palette present · palette imbalance 9.2x
- Style coverage: near-monoculture of 4–7-word 1st-person present-tense phone messages; no long/multi-sentence text, almost no dialogue, formal register, or child/teen voice
- Fixes applied: 8 rows rewritten (1 non-English emoji · 6 labels · 1 low-content · 0 dedup) · 0 left unfixed

## 1. Label correctness

Rates over the 500-row sample (emoji judged generously — the palette is large and
full of near-synonyms):

- **emoji**: 475 ok · 20 weak · 5 wrong (1.0% wrong, 4.0% weak)
- **feeling**: 440 ok · 58 weak · 2 wrong (0.4% wrong, 11.6% weak)

### Systematic patterns (weak, not fixed)

1. **Affect-free logistics labeled `Calm` instead of `Neutral`** — the largest
   single source of weak feeling labels (~14 rows in the sample). Examples:
   "The room booking is confirmed." (✅), "Please resend the file; this one is
   blank." (🔄), "I'll send the notes after the call." (📝), "The appointment
   went smoothly." (✅), "I'm taking the smaller ticket." (🎟️). These are
   defensible (steady professional tone) but `Neutral` is the better fit and the
   corpus already leans heavily `Neutral`/`Calm`.
2. **Mild-positive logistics labeled `Happy`** (~12 rows): "I found your charger"
   (🔌), "The group chat is useful for once" (💬), "You were right about the
   shortcut." (✅), "The new schedule starts tonight!" (📅).
3. **Friendly banter / practical care labeled `Love`** with no romantic or
   familial affection expressed (~15 rows): "you were right, unfortunately.
   coffee?" (☕), "You forgot your umbrella, genius." (☔), "Drink water before I
   become annoying." (💧), "I'm annoyed for you, if that helps" (😤). This looks
   like an intentional project definition of `Love` = "caring nudge", but it
   blurs the boundary with `Neutral`/`Happy`.
4. **Grinning emoji on anxious/frustrated text** (minor): "I have three
   assignments due tonight" 😃, "Boss said "quick call." It's been an hour." 😹.

### Worst individual rows

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| Dinner's ready. Come before it gets cold. | 🔴 / Neutral | 🍽️ | red-circle emoji illustrates nothing (fixed) |
| My plans just got canceled. Tragic | 🔴 / Sad | 😞 | same red-circle artifact (fixed) |
| No plans after work, surprisingly | 自由 / Calm | 😌 | emoji field is a CJK word, not an emoji (fixed) |
| I have three assignments due tonight | 😃 / Anxious | 😰 | grin contradicts the anxious text + label (fixed) |
| Boss said "quick call." It's been an hour. | 😹 / Angry | 😤 | joy-tears cat contradicts the frustration (fixed) |
| Bring flowers to my desk; calculus won. | 🪦 / Happy | 🪦 / Sad | "calculus won" is wry defeat, not joy (fixed) |
| They replaced my chair without asking. | 😠 / Sad | 😠 / Angry | emoji already says Angry; "without asking" = Angry (fixed) |
| Why is the Wi-Fi dead right now | 😫 / Neutral | Angry/Anxious | frustrated rhetorical question; left as weak, not fixed |
| Come walk with us after the next feed? | 🎽 / Love | 🚶 / Neutral | running-shirt emoji, gentle-walk text; weak |
| The office feels unusually chaotic today. | 🐿️ / Anxious | — | squirrel is off-topic; weak |

## 2. Text quality

- **broken (text): 0** — no truncation, no `{feeling}`/template crumbs, no
  trailing `-`, no ungrammatical rows in the sample.
- **non-English emoji field: 1** — row `"No plans after work, surprisingly"` had
  `emoji: "自由"` (Chinese "freedom"). This is a recurring generator bug: **56
  rows** corpus-wide carry `自由` in the emoji field (plus `泪`, `空`, and other
  CJK/geometric junk). Fixed in the sampled row only.
- **normalize-fragile: 5** — lean on digits/currency/times that `normalize`
  deletes:
  - `Come by at 2:30 if you're free, love.` -> `come by at : if youre free love`
  - `Can someone resend the Q3 deck?` -> `can someone resend the q deck?`
  - `I found £20 behind the sofa, jackpot` -> `i found  behind the sofa jackpot`
  - `Doctor at 8:30. Wish me luck.` -> `doctor at : wish me luck`
  - `Walked 4,200 steps. Practically an athlete.` -> `walked  steps practically an athlete`
  Not fixed (per skill: `normalize`/`CHARS` may change).
- **low-content: 1** — `"Crown still present"` (👑 / Neutral): a two-word status
  with nothing for a label to grip. Rewritten to
  `"The paper crown is still on the table."` (labels kept).
- **exact/near-duplicate texts: 0** exact-after-normalize in the sample. One
  soft template repeat noticed but not flagged: "I saved your seat beside mine."
  / "I saved your seat by the window." / "I kept your seat…" (all `Love`).

## 3. Label coverage

### Feelings

| feeling | corpus count | corpus share | sample count |
| --- | --- | --- | --- |
| Neutral | 17,465 | 22.6% | 120 |
| Happy | 16,350 | 21.2% | 110 |
| Calm | 8,879 | 11.5% | 60 |
| Love | 8,861 | 11.5% | 51 |
| Sad | 8,761 | 11.3% | 44 |
| Angry | 8,521 | 11.0% | 57 |
| Anxious | 8,443 | 10.9% | 58 |

`Neutral` + `Happy` = 43.8% of the corpus, ~2x each of the other five. The
sample tracks this. No off-vocab or unused feeling labels.

### Emojis

- **Palette present: 300/300.** Every `labels.json` emoji has **≥146 rows**;
  none below 5, none below 20. Palette coverage is genuinely healthy.
- **Palette imbalance max/min = 1349/146 = 9.2x** (😤 1349 … 👖/🏝️/⚫/🚉 146).
  (Step 2's script reported 1349x only because it divided by an off-palette
  singleton — the real palette floor is 146.)
- top 10: 😤 1349 · 😠 1224 · 😌 1063 · 🎉 1035 · 😔 896 · 😞 779 · ☕ 740 · 😰 726 · 😬 704 · 😟 688
- bottom 10 (palette): 🥛 147 · 🪄 147 · 🚒 147 · ☮️ 146 · 🌮 146 · 👠 146 · 🚉 146 · ⚫ 146 · 🏝️ 146 · 👖 146
- **Off-palette leakage: 571 distinct emoji-field strings across 8,511 rows
  (11.0% of the corpus)** that `data.py`'s `read()` silently drops at load time.
  Most are real emojis just outside the top-300 (🤍 99, 💞 83, 🕰️ 72, 💛 71,
  🎯 67, 🫥 64, 🏡 62, 🎵 61, 🔁 58, ✈️ 57 …); some are junk (自由 56, plus
  `泪`, `空`, unassigned code points). Effective training corpus ≈ 68,800 rows,
  not 77,280.

## 4. Text-style coverage

Skim of ~150 sample rows.

| axis | buckets (approx share) |
| --- | --- |
| register | casual 70% · neutral 25% · slang/net-speak 5% · formal ~0% |
| form | 1st-person feeling/status statement 55% · observation/aphorism 20% · question 20% · dialogue/quote 3% · narrative/recount 2% |
| device | plain 80% · exclamation 15% · in-text emoji 0% · all-caps 0% · profanity <1% |
| age register | adult 75% · indeterminate 20% · teen 5% · child ~0% |

Length: min 3 words, median 6, max 10; 84% of rows are 4–7 words, 0 rows over
15 words.

Gaps:

- **No long messages.** Nothing over 10 words / one sentence-and-a-half. The
  model never sees multi-clause venting, storytelling, or a message with
  context + feeling.
- **Style monoculture.** Overwhelmingly one shape: a casual, adult, 1st-person,
  present-tense phone message of 4–7 words. Register barely varies.
- **Almost no dialogue or reported speech**, no formal/professional writing,
  no child or teen voice, no all-caps, no in-text emoji, near-zero profanity.
- **`Neutral`/`Happy` are 2x over-represented** vs the other five feelings.
- Recurring props narrow the world further: backyard chickens, prams/feeds,
  pretzels 🥨, nurse-shift lines, cricket 🏏 — the same few scenarios recur.

## 5. Fixes applied

- rewritten: **8** rows (0 broken-text · 6 labels · 1 low-content · 0 dedup · +1
  non-English emoji field). Fixes file: `report/data/08-30-09:56.fixes.jsonl`
- unfixed (flagged but not confidently fixable): **0**. Weak labels,
  normalize-fragile rows, and style gaps were deliberately left alone per skill
  scope.

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |
| No plans after work, surprisingly — 自由 / Calm | 😌 / Calm | emoji field held a CJK word, not an emoji |
| Dinner's ready. Come before it gets cold. — 🔴 / Neutral | 🍽️ / Neutral | red circle illustrates nothing |
| My plans just got canceled. Tragic — 🔴 / Sad | 😞 / Sad | red circle illustrates nothing |
| Boss said "quick call." It's been an hour. — 😹 / Angry | 😤 / Angry | joy-tears cat contradicts the frustration |
| I have three assignments due tonight — 😃 / Anxious | 😰 / Anxious | grin contradicts anxious text + label |
| Bring flowers to my desk; calculus won. — 🪦 / Happy | 🪦 / Sad | wry defeat, not joy |
| They replaced my chair without asking. — 😠 / Sad | 😠 / Angry | emoji + "without asking" both point at Angry |
| Crown still present — 👑 / Neutral | "The paper crown is still on the table." / 👑 / Neutral | low-content; rewritten to a concrete message |

## 6. Verdict & recommendations

1. **Stop the emoji-field leakage (highest impact).** 11% of generated rows
   (8,511) are discarded at load because the emoji isn't in the top-300 palette,
   and 56 of them contain the literal string `自由`. Fix the generators
   (`feeling2emoji.ts`, `emoji2feeling.ts`) to (a) validate the annotator output
   is a single emoji and (b) constrain / re-map it to a `labels.json` member, or
   re-run `gen_labels.ts` more often so heavy off-palette emojis (🤍, 💞, 🎯, 🔁)
   actually enter the palette.
2. **Move affect-free logistics from `Calm` to `Neutral` in the annotator
   prompt.** The single biggest labeling weakness is confirmations / "resend the
   file" / "appointment went smoothly" being tagged `Calm`. Tighten the
   `Calm` vs `Neutral` instruction and add contrast examples.
3. **Rebalance feelings.** `Neutral` + `Happy` are ~2x the others. Run
   `feeling2emoji.ts` targeting the five minority feelings until shares even out,
   since feeling accuracy is the priority metric.
4. **Break the style monoculture.** Add a length axis to generation (some 12–30
   word, multi-sentence messages) and widen register: dialogue, formal notes,
   teen/child voice, all-caps. Every row today is a 4–7-word casual adult phone
   message.
5. **Tighten the `Love` definition.** Either accept "practical care / banter" as
   `Love` explicitly with examples, or reserve `Love` for expressed affection and
   push the banter rows to `Neutral`/`Happy`. Right now the boundary is fuzzy.
6. **Add an output guard for near-empty texts** (`< 3 words` or no content noun)
   in the generators — "Crown still present" / "Seems nice overall" slip through.
