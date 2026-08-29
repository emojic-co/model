# Data quality report — 2026-08-29 09:30

- Sample: 500 of 56,041 rows (`report/data/08-29-09:30.sample.jsonl`)
- Label correctness: emoji 473/500 ok · 25 weak · 2 wrong; feeling 403/500 ok · 90 weak · 7 wrong
- Text quality: 470/500 clean · 1 broken · 18 normalize-fragile · 11 low-content
- Label coverage: feelings 8/8 present · emojis 300/300 present · imbalance 954x (raw count, incl. off-palette values)
- Style coverage: one voice — casual 1st-person present-tense WhatsApp one-liners, 84% are 4–7 words; biggest gap is total absence of formal register and any multi-sentence / longer text.
- Fixes applied: 19 rows rewritten (1 broken · 7 labels · 11 low-content · 0 dedup) · 0 left unfixed

## 1. Label correctness

Emoji fit is strong. The palette is used mostly as a **literal illustration of the
text's topic** (beans → 🫘, moose → 🫎, charger → 🔌, clock-time → 🕝), and that
works. ~25/500 are "weak": an arbitrary decorative glyph that neither illustrates
nor contradicts (🟡 "putting the kettle on", ⭐ "code brown", 👽 "bring coffee and
a clean shirt", 🐺 "found the blockage", ➡️ "found the typo", ⛲ "the plan is
moving"). Only 2 are outright wrong:

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |
| Send help, towels, and a clean onesie | 泥 / Anxious | 🚼 | `emoji` field holds a CJK character (泥 "mud"), not an emoji — drops at load |
| Vitals are stable, spirits are not. | 📸 / Sad | 😔 | camera points at a different topic; left unfixed (weak-vs-wrong borderline) |

Feeling fit has one **systematic** problem: the generators reach for a strong
feeling on flat, affect-free logistics text.

- **`Love` as a catch-all for any domestic / caring / practical message.** "You
  left your headphones in my car." · "I'm outside. Open the door?" · "I brought
  the charger you forgot." · "Your socks are on the radiator." · "Wear the blue
  jumper, it's cold out." · "I'm bringing coffee. Your usual." — all `Love`, all
  better read as `Neutral`. ~15 rows in the sample.
- **`Excited` / `Calm` on pure scheduling text.** "I'm opening the files right
  now." → `Excited`. "Meeting moved to four. Breathe." → `Love`. "The new
  template is in the shared folder." → `Calm`. "I put your flyer in the right
  box." → `Calm`.
- **`Sad` for wry self-directed annoyance.** "Back home. Forgot the bread,
  naturally." · "My hamstrings filed a formal complaint" · "I missed the goal
  while making tea. Typical." · "Bad news: I still have to work" — these are
  `Neutral`/annoyed in tone, not sad. (There is no `Annoyed` label, so `Neutral`
  is the correct sink; instead they land on `Sad`.)
- **`Angry` inflation.** Mild irritation ("The elevator is broken again", "The
  cafeteria ran out of everything decent", movie-quality gripes) is routinely
  labeled `Angry` where `Neutral` would be defensible.

Net: 90/500 feelings are "weak" (defensible but not the best label) and 7 are
clearly wrong (fixed — see §5). The weak rate is high enough that the feeling
head is being trained toward "any warm text = Love, any downbeat text = Sad".

| text | labeled feeling | better fit | note |
| --- | --- | --- | --- |
| This is me being responsible. Terrifying. | Love | Neutral | self-mocking aside, no 2nd person, no affection |
| How dare you look gorgeous! | Angry | Love | mock-outrage compliment / flirtation |
| I'm fine to present. Probably. | Sad | Anxious | the "Probably." undercut is nerves, not sadness |
| I'm opening the files right now. | Excited | Neutral | zero excitement markers |
| Meeting moved to four. Breathe. | Love | Calm | "Breathe" = self-reassurance about a schedule change |
| My hamstrings filed a formal complaint | Sad | Neutral | joke about soreness |
| Back home. Forgot the bread, naturally. | Sad | Neutral | wry self-annoyance |

## 2. Text quality

- **broken: 1** — row with `泥` (a CJK char) in the `emoji` field (fixed). Two
  borderline-but-left: `"Bad news: I just said "good news."` (unbalanced quote)
  and `"Cant say im sad, the song was awful"` (clumsy double-negative, meaning
  recoverable).
- **normalize-fragile: 18** — text whose meaning leans on characters `normalize`
  deletes (digits, `.`, `,`, `-`, `:`). Worst cases:
  - `2-1! I take back everything I said` -> `! i take back everything i said` (score gone)
  - `The deadline says 11:59. Which timezone?` -> `the deadline says : which timezone?`
  - `Who thought an 8 a.m. exam was reasonable?` -> `who thought an  am exam was reasonable?`
  - `Taco night at mine, 7ish?` -> `taco night at mine ish?`
  - `I accidentally liked their post from 2021` -> `i accidentally liked their post from`
  - clock times (`2:30`, `8:30`, `7am`, `8am`) recur — ~11 of the 18 are a time-of-day.
  Left untouched per skill scope (normalize/CHARS may change).
- **low-content: 11** — grammatical but nothing a label hangs on: "Working on a
  task", "That's quite something", "Stay awesome", "Guess it is what it is",
  "Resting in peaceful energy", "Calmly reading the signs", "All feels fine",
  "I'm glowing", "Peace in every color", "Savoring this peaceful moment",
  "Feeling happy and glowing". All rewritten (§5).
- **exact/near duplicates: 0 exact** after normalize. Near-duplicate *templates*
  are common though (see §4): `"missed it by ~two seconds"` (rows 108 & 227,
  both Sad), `"...move our call to 2:30"` / `"...meet around 2:30"`, `"Your
  <garment> is <place>."` ×3, `"Not (that) calm rn, <x>"` ×3.

## 3. Label coverage

### Feelings

| feeling | corpus count | corpus share | sample count |
| --- | --- | --- | --- |
| Neutral | 8,495 | 15.2% | 83 |
| Anxious | 7,197 | 12.8% | 65 |
| Calm | 7,096 | 12.7% | 76 |
| Happy | 7,050 | 12.6% | 60 |
| Excited | 6,831 | 12.2% | 48 |
| Love | 6,787 | 12.1% | 47 |
| Sad | 6,431 | 11.5% | 54 |
| Angry | 6,140 | 11.0% | 67 |

Feelings are well balanced (max/min = 1.4x). **15 corpus rows carry an off-vocab
feeling** — `Annoyed` ×5, `Confused` ×4, `Frustrated` ×2, `Hopeful`/`Amused`/
`Relieved` ×1 — leftovers from an older label set; they are silently dropped at
load by `data.py`'s `read()`. Harmless but worth a one-time `sed` cleanup.

### Emojis

- present: **300/300** palette emojis appear at least once; absent: none.
- The corpus contains **829 distinct emoji values** — 529 of them are outside the
  300-emoji palette and drop at load (row 86's `泥`, plus a long tail of
  count-1 glyphs: 💜 🕥 🥊 🤱 👹 🟣 🫒 🍥 👛 🛞 …).
- top 10: 😤 954 · 😌 933 · 🎉 904 · ☕ 677 · 😬 666 · 😠 651 · 😰 620 · 😔 614 · 😩 566 · 😟 561
- bottom 10 (of all values, non-palette): 💜 🕥 🥊 🤱 👹 🟣 🫒 🍥 👛 🛞 — all count 1
- imbalance max/min = 954/1 = **954x** (raw). Even confined to the palette the
  spread is steep: a cluster of face-emojis at 500–950 vs. many object emojis in
  low double digits. The emoji head has ~187 rows/class on average but a heavy
  head and a thin tail.

## 4. Text-style coverage

| axis | buckets (approx share) |
| --- | --- |
| register | casual ~70% · neutral ~25% · slang/net-speak ~5% (`rn`, `lol`, `im`, `bro`, `mate`) · formal ~0% |
| form | 1st-person feeling/statement ~55% · observation/quip ~20% · question ~20% (`Can we…?`, `Want to…?`) · narrative/recount ~5% · dialogue/quote ~0% |
| device | plain ~75% · exclamation ~24% · all-caps ~0% · in-text emoji 0% · profanity ~1% (mild: "damn", "crap") |
| age register | adult ~85% · teen ~10% (group-chat, school, "bro") · child ~0% · indeterminate ~5% |

Gaps:

- **No formal register at all** — no workplace-formal, customer-service, email,
  or written-correspondence voice.
- **Length monoculture** — 84% of rows are 4–7 words, max is 10; nothing
  multi-sentence, no longer vents or recounts with an arc. Driven by the
  generators' ≤50-char cap.
- **Tense/POV monoculture** — almost everything is 1st-person present tense.
  Little past-tense storytelling, no 3rd-person narration.
- **Age monoculture** — overwhelmingly adult; essentially no child voice.
- **No emphasis devices** — zero ALL-CAPS, zero repeated punctuation, zero
  in-text emoji (all three would be flattened by `normalize` anyway, so this is
  arguably fine).
- **Persona / template reuse** — the same handful of scenes recur: domestic
  partner, hospital ward / nurse, student with a deadline, tradesperson mid-job.
  Templates `"Not that calm rn, <x>"`, `"Your <garment> is <place>."`, `"Bad
  news: <x>"` / `"Good news: <x>"`, `"Meet me by <x>"`, `"<time> reschedule"`
  each appear multiple times in a 500-row sample.

## 5. Fixes applied

- rewritten: **19 rows** (1 broken · 7 labels · 11 low-content · 0 dedup); fixes
  file `report/data/08-29-09:30.fixes.jsonl`
- unfixed (flagged but not confidently fixable): **0**. Deliberately left:
  normalize-fragile rows (out of scope), all `*_weak` labels, the `📸`/"vitals"
  emoji (weak-vs-wrong borderline), the two borderline-broken texts in §2.

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |
| Send help, towels, and a clean onesie — 泥 / Anxious | emoji → 🚼 | `emoji` field was a CJK character, not an emoji |
| This is me being responsible. Terrifying. — 😅 / Love | feeling → Neutral | self-mocking aside; no affection / 2nd person |
| How dare you look gorgeous! — 😠 / Angry | feeling → Love | mock-outrage compliment |
| I'm fine to present. Probably. — 😬 / Sad | feeling → Anxious | the "Probably." is nerves |
| I'm opening the files right now. — 📂 / Excited | feeling → Neutral | flat logistics, no excitement |
| Meeting moved to four. Breathe. — 😮‍💨 / Love | feeling → Calm | self-reassurance about a schedule change |
| My hamstrings filed a formal complaint — 🤸 / Sad | feeling → Neutral | joke about soreness |
| Back home. Forgot the bread, naturally. — 🙄 / Sad | feeling → Neutral | wry self-annoyance |
| Working on a task — 📝 / Neutral | "I'm drafting the status update now." | low-content |
| That's quite something — 🙂 / Neutral | "The new schedule just came through." | low-content |
| Stay awesome — 😎 / Happy | "You crushed that presentation today." | low-content |
| Guess it is what it is — 🤷 / Neutral | "The order slipped to next week, not much we can do." | low-content |
| Resting in peaceful energy — ☮️ / Calm | "Lying in the garden with nowhere to be." | low-content |
| Calmly reading the signs — 📖 / Calm | "Reading the trail signs, taking my time." | low-content |
| All feels fine — 🙂 / Happy | "Everything went smoothly today, no complaints." | low-content |
| I'm glowing — ✨ / Happy | "Nailed the interview and I feel unstoppable." | low-content |
| Peace in every color — 🌈 / Calm | "Watching the sunset spread across the sky." | low-content |
| Savoring this peaceful moment — 🕊️ / Calm | "Sitting by the lake before the house wakes up." | low-content |
| Feeling happy and glowing — ✨ / Happy | "Got the offer this morning and I can't stop smiling." | low-content |

## 6. Verdict & recommendations

1. **Fix the feeling-annotation prompt in `feeling2emoji.ts` / `emoji2feeling.ts`
   to allow `Neutral` for flat text.** The dominant defect (90/500 weak
   feelings) is strong-feeling inflation: `Love` for any caring/domestic line,
   `Sad` for wry self-annoyance, `Excited`/`Calm` for scheduling. Add explicit
   instruction + few-shot examples that logistics/practical messages are
   `Neutral`.
2. **Add an `Annoyed` sink or fold it explicitly into `Neutral`.** A large share
   of "weak `Sad`" and "inflated `Angry`" rows are really mild annoyance with no
   home label, so the annotator scatters them.
3. **Flatten the emoji distribution.** 954x imbalance and a 529-value off-palette
   tail. Run `emoji2feeling.ts` (its whole job) a few times, then `gen_labels.ts`;
   consider capping per-emoji rows on the high end.
4. **Broaden text style.** Everything is a 4–7-word 1st-person present-tense
   one-liner. Raise the char cap for a subset, add a formal/workplace voice, add
   past-tense recounts, and de-duplicate the recurring templates (`"Not that
   calm rn…"`, `"Your <garment> is <place>"`, `"Bad news: …"`).
5. **One-time corpus hygiene.** Drop or relabel the 15 off-vocab-feeling rows;
   strip the 529 off-palette emoji rows (or let `read()` keep doing it, but they
   inflate `data.jsonl` size for nothing).
6. **Reduce normalize-fragile generation.** ~11 of 18 casualties are clock times
   (`2:30`, `7am`). Nudge the generator away from digit-dependent phrasing, or
   teach it to spell times out ("half two", "seven in the morning").
