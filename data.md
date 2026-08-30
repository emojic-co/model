# Data quality report — 2026-08-30 14:47

- Sample: 500 of 88624 rows (`report/data/08-30-14:47.sample.jsonl`)
- Label correctness: emoji 463/500 ok · 35 weak · 2 wrong; feeling 397/500 ok · 100 weak · 3 wrong
- Text quality: 493/500 clean · 0 broken · 6 normalize-fragile · 1 low-content
- Label coverage: feelings 7/7 present · emojis 300/300 palette present (887 distinct strings in corpus, 587 off-palette) · imbalance 1712x
- Style coverage: short deadpan 1st-person one-liners in a casual/neutral register; biggest gap is length + register monoculture (median 6 words, 0 rows >10 words, no formal writing, no multi-sentence messages)
- Fixes applied: 6 rows rewritten (0 broken · 5 labels · 1 low-content · 0 dedup) · 0 left unfixed

## 1. Label correctness

Rates: emoji 463 ok / 35 weak / 2 wrong; feeling 397 ok / 100 weak / 3 wrong.

Emoji quality is high — the palette is large and abstract emojis are used loosely
by design, so almost anything in the right cluster reads as fine. Feeling quality
is the weak spot: ~20% of rows are defensible-but-not-best, concentrated in one
register (see below).

### Systematic patterns (most important first)

1. **Ironic / deadpan one-liners get a strong feeling forced onto them.** A large
   share of the corpus is wry observation — `"..., apparently"`, `"..., of
   course"`, `"living the dream"`, `"great combo"`, `"Naturally."`,
   `"approximately forever"`. The real affect is mild or neutral-negative, but
   the label is Sad / Angry / Anxious. This alone accounts for most of the 100
   weak feelings. Examples: `"Wet socks before 7am, living the dream"` → Sad;
   `"That meeting lasted approximately forever"` → Angry; `"Markets are down; our
   laundry pile is up."` → Sad; `"Lunch break vanished into a loading screen."`
   → Sad.
2. **Sad is over-applied to flat disappointment / mild dismissal.** `"The joke
   wasn't that funny."`, `"Made soup. Forgot the soup."`, `"not exactly thrilled
   about all the photos"`, `"The vendor cancelled."` — Neutral fits better than
   Sad for affect-light letdown.
3. **Happy bleeds onto neutral logistics.** Plain invitations / questions labeled
   Happy: `"are you free after school today"` (🤔), `"Should we make this a whole
   weekend?"`, `"Guess what's unloading right now?"`.
4. **Emoji/feeling contradictions occasionally slip through.** `"The printer
   works for everyone but me."` carries 😤 (angry face) but feeling Sad; row 51
   `"I saw your photo. You look happy."` uses 😊 with Sad (here intentional —
   bittersweet).
5. **Template-y "convey feeling X in an absurd context" lines.** `"I'm angry you
   won!"`, `"I'm mad at that octopus"`, `"Fuming over your beauty!"`, `"Happy as
   a wave!"`, `"I'm anxious, your highness"`. Labels are technically right (they
   name the feeling) but the texts are low-realism.
6. **Face-emoji concentration.** 8 of the top-10 emojis are expressive faces
   (😠 😤 😌 🎉 😔 😡 😞 😰 ☕ 😟). The "free choice" emoji step collapses toward a
   small expressive-face vocabulary despite a 300-emoji palette and 887 distinct
   strings seen.

### Worst individual rows

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| The hallway light is dead again | `पछ` / Angry | 💡 / Angry | emoji field is Devanagari garbage, not an emoji — **fixed** |
| The garden survived the storm | 📹 / Neutral | 🌿 / Neutral | video camera unrelated to garden/storm — **fixed** |
| I just saw the kitchen sink. Never mind. | 🤢 / Sad | 🤢 / Neutral | wry "ugh, dishes" — disgust/dismissal, not sadness — **fixed** |
| Made soup. Forgot the soup. | 😅 / Sad | 😅 / Neutral | self-deprecating joke; 😅 already signals amused — **fixed** |
| The joke wasn't that funny. | 😐 / Sad | 😐 / Neutral | deadpan/unimpressed, not sad — **fixed** |
| The printer works for everyone but me. | 😤 / Sad | 😤 / Angry | angry-face emoji contradicts Sad; frustration reads Angry | weak, not fixed |
| That scarecrow looks more alive than you | 🌾 / Angry | 🌾 / Neutral | playful jab, not anger | weak, not fixed |
| are you free after school today | 🤔 / Happy | 🤔 / Neutral | flat logistics question, no affect | weak, not fixed |

## 2. Text quality

- **broken: 0** — no truncation, template artifacts, JSON crumbs, or non-English
  in the `text` field. (The one garbage token in the sample, `पछ`, is in the
  `emoji` field — counted as a label fix, not broken text.)
- **normalize-fragile: 6** — read fine now but lean on digits/currency/dashes
  that `normalize` deletes:
  - `"Wet socks before 7am, living the dream"` -> `"wet socks before am living the dream"`
  - `"Rounds at 7. Please have the charts ready."` -> `"rounds at  please have the charts ready"`
  - `"Maya's recital starts at 6, allegedly"` -> `"mayas recital starts at  allegedly"`
  - `"Inflation up, sandwich still £4.20."` -> `"inflation up sandwich still "`
  - `"Found £5 in an old coat. Riches, apparently."` -> `"found  in an old coat riches apparently"`
  - `"Our team won 4–0 somehow"` -> `"our team won  somehow"`
- **low-content: 1** — `"Actually, loudly in my head."` (context-less fragment,
  nothing a Love label can hang on) — **fixed**.
- **exact/near duplicates: 0** — no exact-after-normalize duplicate texts in the
  sample.

## 3. Label coverage

### Feelings

| feeling | corpus count | corpus share | sample count |
| --- | --- | --- | --- |
| Neutral | 20383 | 23.0% | 99 |
| Happy | 16718 | 18.9% | 94 |
| Love | 11103 | 12.5% | 73 |
| Calm | 10849 | 12.2% | 52 |
| Angry | 10663 | 12.0% | 72 |
| Sad | 9530 | 10.8% | 53 |
| Anxious | 9378 | 10.6% | 57 |

7/7 present. Mild imbalance — Neutral ≈ 2.2x Anxious. Sample tracks the corpus.

### Emojis

- Palette present: 300/300; absent: none.
- Corpus contains **887 distinct emoji strings** — 587 of them outside the 300
  palette (dropped at load by `data.py`). The off-palette tail includes true
  garbage: `पछ`, `呼` (CJK char), variation-selector / ZWJ forms
  (`🗃️`, `🏃‍♂️`).
- top 10: 😠 1712 · 😤 1695 · 😌 1208 · 🎉 1044 · 😔 972 · 😡 950 · 😞 837 · 😰 823 · ☕ 800 · 😟 793
- bottom 10 (all off-palette, count 1): 🫴 · ⛪ · 📫 · 🧡 · 🗃️ · 🏃‍♂️ · 🚯 · 呼 · 🧿 · 🫤
- imbalance max/min = 1712/1 = 1712x (min=1 is off-palette junk; within the
  palette the tail is far shallower, but the head is still heavily face-weighted).

## 4. Text-style coverage

| axis | buckets (approx share) |
| --- | --- |
| register | formal ~5% · neutral ~45% · casual ~42% · slang/net-speak ~8% |
| form | 1st-person feeling statement ~40% · observation/aphorism ~25% · question ~15% · narrative/recount ~12% · imperative/request ~5% · dialogue/quote ~3% |
| device | plain ~72% · exclamation ~25% · all-caps ~0% · in-text emoji 0% · profanity ~2% (mild: "bloody") |
| age register | adult ~70% · teen ~12% · child ~0% · indeterminate ~18% |

Length (words): min 3 · p25 5 · median 6 · p75 7 · max 10. Buckets: 1-3: 8 · 4-7: 413 · 8-15: 79 · 16+: 0.

Gaps:

- **Length monoculture.** Every row is one short clause or two tiny sentences.
  Nothing over 10 words; no paragraph-length venting, storytelling, or
  multi-turn context. Browser inputs will routinely be longer than anything
  trained on.
- **No formal / written register.** No emails, notices, reports, academic or
  professional prose. The clinical/work lines that exist are still terse chat.
- **Device axis is flat.** Zero all-caps shouting, zero in-text emoji, almost no
  profanity — real WhatsApp-style text has all three.
- **Form skew.** 1st-person present-tense statements + wry observations dominate;
  reported dialogue and past-tense narrative are rare.
- **Age skew.** Adult-dominated; no child voice, thin teen presence.
- **Irony tic.** A very large minority are deadpan/ironic one-liners
  (`"..., apparently"`, `"Naturally."`, `"great combo"`). This single stylistic
  habit is the direct cause of the ~20% weak feeling labels.

## 5. Fixes applied

- rewritten: 6 rows (0 broken · 5 labels · 1 low-content · 0 dedup); fixes file
  `report/data/08-30-14:47.fixes.jsonl`
- unfixed (flagged but not confidently fixable): 0

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |
| The hallway light is dead again — `पछ` / Angry | 💡 / Angry | emoji field held Devanagari garbage, not an emoji |
| The garden survived the storm — 📹 / Neutral | 🌿 / Neutral | video camera points at a different topic than garden/storm |
| I just saw the kitchen sink. Never mind. — 🤢 / Sad | 🤢 / Neutral | wry disgust/dismissal, not sadness |
| Made soup. Forgot the soup. — 😅 / Sad | 😅 / Neutral | self-deprecating joke; Sad is clearly not the best label |
| The joke wasn't that funny. — 😐 / Sad | 😐 / Neutral | deadpan/unimpressed reads Neutral |
| Actually, loudly in my head. — 📣 / Love | "I'm cheering you on, loudly, even from here." — 📣 / Love | context-less fragment rewritten to a concrete message fitting both labels |

## 6. Verdict & recommendations

1. **Break the irony monoculture in the generators.** The deadpan one-liner
   (`"..., apparently"` / `"Naturally."` / `"great combo"`) is overrepresented
   and is the main source of stretched feeling labels. Add explicit
   voice-spread buckets for plain-sincere and plain-flat statements, and cap the
   ironic share.
2. **Widen the length band.** Every sampled row is 3–10 words. Add a length axis
   to `feeling2emoji.ts` / `emoji2feeling.ts` (e.g. one-liner / two-sentence /
   short-paragraph) so training sees inputs as long as real browser text.
3. **Tighten the feeling annotation prompt toward Neutral.** Instruct the
   annotator that affect-light letdown, dry logistics, and wry observations are
   Neutral — not Sad/Angry/Anxious/Happy. This is the highest-leverage label
   fix given `feeling` matters more than `emoji`.
4. **Validate the emoji field at generation time.** 587/887 distinct emoji
   strings are off-palette and some are pure garbage (`पछ`, `呼`, ZWJ/VS forms).
   Reject any annotation whose emoji isn't a single palette codepoint before it
   reaches `data.jsonl`.
5. **Encourage emoji diversity.** 8 of the top-10 emojis are expressive faces.
   In the emoji-free-choice step, nudge toward object/scene emojis that match
   the text's content, not just a face that matches its mood.
6. **Add register + device variety.** Seed some formal/written samples and some
   all-caps / heavier-punctuation / mild-profanity samples so the model isn't
   blind to those input styles.
