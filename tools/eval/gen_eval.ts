import { existsSync } from "node:fs"
import { readFile, writeFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const DATA = "./data.jsonl"
const LABELS = "./labels.json"
const EVAL = "./eval.jsonl"

const EVAL_SIZE = 900

const MAX_RAW_LEN = 50
const MAX_TEXT_LEN = 42

const OVERGEN = 8
const MAX_ROUNDS = 6

const GEN_BATCH_SIZE = 100
const GEN_CONCURRENCY = 20

const VERIFY_VOTES = 3
const VERIFY_BATCH_SIZE = 10
const VERIFY_CONCURRENCY = 20

const EMOJI_BATCH_SIZE = 10
const EMOJI_CONCURRENCY = 20

const VOCAB = new Set("abcdefghijklmnopqrstuvwxyz!?:()@$%&* ")
function normalize(text: string): string {
  const t = text
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/(.)\1{2,}/g, "$1$1")
  return [...t].filter((c) => VOCAB.has(c)).join("")
}

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

const FEELING_GUIDANCE = [
  "Label the emotion a typical reader would actually feel from the words, not",
  "one that is merely plausible for the situation.",
  '- "Neutral" is the right answer for flat, practical or informational',
  "  messages: logistics, scheduling, quick factual updates, plain questions,",
  "  low-effort replies. Do not upgrade these to a stronger feeling.",
  '- Do not inflate. A caring or domestic line ("your socks are on the',
  '  radiator") is Neutral unless it openly states affection -- only then Love.',
  "  Mild irritation with no heat is Neutral, not Angry. A dry or self-mocking",
  "  complaint is Neutral, not Sad. Plainly stated anticipation is Neutral, not",
  "  Happy.",
  "- Reserve Love, Sad and Angry for messages where that feeling is",
  "  unmistakably on the surface.",
  "- If two feelings fit, pick the milder; if none clearly fits, pick Neutral.",
].join("\n")

function conveyInstruction(feeling: string, feelings: string[]): string[] {
  if (feeling === "Neutral") {
    return [
      "Every message must be genuinely emotion-free: a plain, practical or",
      "informational note (logistics, scheduling, a quick fact, a low-key",
      "question), with no detectable mood, positive or negative. A reader",
      `choosing from [${feelings.join(", ")}] should land on Neutral and on`,
      "no other feeling.",
    ]
  }
  return [
    `Every message must unmistakably convey the feeling "${feeling}" -- someone`,
    "reading it cold, with no context, should name that feeling. Convey it",
    "through what is said and how, never by naming the feeling.",
    "A flat or logistical message that a person in that mood merely could have",
    "sent does not count: the feeling has to be visible in the words.",
    `A reader choosing from [${feelings.join(", ")}] should overwhelmingly pick`,
    `"${feeling}"; no other feeling on that list should be a reasonable second`,
    "choice.",
  ]
}

type Row = { emoji: string; feeling: string; text: string }
type Candidate = { feeling: string; text: string }

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

async function genBatch(
  voice: string,
  feeling: string,
  feelings: string[],
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${GEN_BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
      ...conveyInstruction(feeling, feelings),
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary the wording, situation and sentence shape, but keep every line firmly in the target feeling -- do not drift toward a different mood for the sake of variety.`,
      `Sound real and specific.`,
    ].join("\n"),
  })
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

const Vote = z.object({ id: z.number(), feeling: z.string() })

async function voteBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, string>> {
  const ids = new Set(batch.map((b) => b.id))
  const valid = new Set(feelings)
  const byId = new Map<number, string>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Vote) }),
        }),
        prompt: [
          "You are an annotator. For each message below, choose the single feeling that best fits it.",
          `Choose exactly one from this list: ${feelings.join(", ")}.`,
          "",
          FEELING_GUIDANCE,
          "",
          "Return exactly one annotation object per input message, echoing its id.",
          "Do not add, drop, reorder, or merge items.",
          `Format: {"annotations": [{"id": 0, "feeling": "${feelings[0]}"}]}`,
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        const f = a.feeling.trim()
        if (ids.has(a.id) && valid.has(f)) byId.set(a.id, f)
      }
    } catch (err) {
      if (attempt === 1) console.warn(`\n  vote batch of ${batch.length} failed: ${err}`)
    }
  }
  return byId
}

const Adv = z.object({
  id: z.number(),
  feeling: z.string(),
  ambiguous: z.boolean(),
})

async function adversarialBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, { feeling: string; ambiguous: boolean }>> {
  const ids = new Set(batch.map((b) => b.id))
  const valid = new Set(feelings)
  const byId = new Map<number, { feeling: string; ambiguous: boolean }>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Adv) }),
        }),
        prompt: [
          "You are a strict annotator checking whether each message has a single",
          "unambiguous feeling.",
          `The feeling list is: ${feelings.join(", ")}.`,
          "",
          FEELING_GUIDANCE,
          "",
          "For each message: pick the single best feeling, then decide whether a",
          "DIFFERENT feeling from the list could also be justifiably assigned by a",
          "reasonable reader. Set ambiguous=true if it could, false if the best",
          "feeling is clearly the only defensible one.",
          "",
          "Return exactly one object per input message, echoing its id.",
          `Format: {"annotations": [{"id": 0, "feeling": "${feelings[0]}", "ambiguous": false}]}`,
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        const f = a.feeling.trim()
        if (ids.has(a.id) && valid.has(f)) {
          byId.set(a.id, { feeling: f, ambiguous: a.ambiguous })
        }
      }
    } catch (err) {
      if (attempt === 1) console.warn(`\n  adversarial batch failed: ${err}`)
    }
  }
  return byId
}

async function verify(
  cands: Candidate[],
  feelings: string[],
): Promise<Candidate[]> {
  if (!cands.length) return []
  const items = cands.map((c, id) => ({ id, feeling: c.feeling, text: c.text }))
  const batches = chunk(items, VERIFY_BATCH_SIZE)

  const bar = new cliProgress.SingleBar(
    { format: "verifying  |{bar}| {percentage}% | {value}/{total} passes | ETA: {eta}s" },
    cliProgress.Presets.shades_classic,
  )
  bar.start(batches.length * (VERIFY_VOTES + 1), 0)

  const votes = new Map<number, string[]>()
  const adv = new Map<number, { feeling: string; ambiguous: boolean }>()

  const q = new PQueue({ concurrency: VERIFY_CONCURRENCY })
  for (let v = 0; v < VERIFY_VOTES; v++) {
    q.addAll(
      batches.map((batch) => async () => {
        const got = await voteBatch(
          batch.map((b) => ({ id: b.id, text: b.text })),
          feelings,
        )
        for (const [id, f] of got) {
          const arr = votes.get(id) ?? []
          arr.push(f)
          votes.set(id, arr)
        }
        bar.increment()
      }),
    )
  }
  q.addAll(
    batches.map((batch) => async () => {
      const got = await adversarialBatch(
        batch.map((b) => ({ id: b.id, text: b.text })),
        feelings,
      )
      for (const [id, rec] of got) adv.set(id, rec)
      bar.increment()
    }),
  )
  await q.onIdle()
  bar.stop()

  const kept: Candidate[] = []
  for (const it of items) {
    const vs = votes.get(it.id)
    const a = adv.get(it.id)
    if (!vs || vs.length < VERIFY_VOTES || !a) continue
    if (!vs.every((f) => f === it.feeling)) continue
    if (a.ambiguous || a.feeling !== it.feeling) continue
    kept.push({ feeling: it.feeling, text: it.text })
  }
  return kept
}

const EmojiPick = z.object({ id: z.number(), emoji: z.string() })

async function emojiBatch(
  batch: { id: number; text: string }[],
  palette: string[],
): Promise<Map<number, string>> {
  const ids = new Set(batch.map((b) => b.id))
  const valid = new Set(palette)
  const byId = new Map<number, string>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(EmojiPick) }),
        }),
        prompt: [
          "For each message below, pick the single emoji that best fits it.",
          "Choose the most fitting emoji, not a safe default.",
          `You must choose from exactly this list (copy the character verbatim): ${palette.join(" ")}`,
          "",
          "Return exactly one object per input message, echoing its id.",
          `Format: {"annotations": [{"id": 0, "emoji": "${palette[0]}"}]}`,
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        const e = a.emoji.trim()
        if (ids.has(a.id) && valid.has(e)) byId.set(a.id, e)
      }
    } catch (err) {
      if (attempt === 1) console.warn(`\n  emoji batch failed: ${err}`)
    }
  }
  return byId
}

const FALLBACK_EMOJI: Record<string, string> = {
  Happy: "😊", Calm: "😌", Sad: "😔",
  Angry: "😠", Anxious: "😰", Neutral: "✅", Love: "🥰",
}

if (import.meta.main) {
  const force = process.argv.includes("--force")
  if (existsSync(EVAL) && !force) {
    throw new Error(`${EVAL} exists; pass --force to overwrite it`)
  }
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  if (!existsSync(DATA)) throw new Error(`${DATA} not found`)

  const labels = z
    .object({ feelings: z.array(z.string()), emojis: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8")))
  const feelings = labels.feelings
  const palette = labels.emojis

  const base = Math.floor(EVAL_SIZE / feelings.length)
  const rem = EVAL_SIZE % feelings.length
  const target = new Map(feelings.map((f, i) => [f, base + (i < rem ? 1 : 0)]))

  const inCorpus = new Set<string>()
  for (const l of (await readFile(DATA, "utf8")).split("\n")) {
    const line = l.trim()
    if (!line) continue
    const n = normalize((JSON.parse(line) as Row).text)
    if (n) inCorpus.add(n)
  }
  console.log(`${inCorpus.size} corpus texts to dedup against`)
  console.log(
    "targets: " + feelings.map((f) => `${f}=${target.get(f)}`).join(" "),
  )

  const accepted = new Map<string, Candidate[]>(feelings.map((f) => [f, []]))
  const seen = new Set<string>()

  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const deficit = feelings
      .map((f) => ({ f, need: target.get(f)! - accepted.get(f)!.length }))
      .filter((d) => d.need > 0)
    if (!deficit.length) break

    console.log(
      `\n=== round ${round} ===  need: ` +
      deficit.map((d) => `${d.f}=${d.need}`).join(" "),
    )

    const jobs: { feeling: string }[] = []
    for (const d of deficit) {
      const nBatches = Math.max(
        1,
        Math.ceil((d.need * OVERGEN) / GEN_BATCH_SIZE),
      )
      for (let i = 0; i < nBatches; i++) jobs.push({ feeling: d.f })
    }

    const genBar = new cliProgress.SingleBar(
      { format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s" },
      cliProgress.Presets.shades_classic,
    )
    genBar.start(jobs.length, 0)

    const fresh: Candidate[] = []
    const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
    genQ.addAll(
      jobs.map((job) => async () => {
        try {
          for (const text of await genBatch(pickVoice(), job.feeling, feelings)) {
            const n = normalize(text)
            if (!n || n.length > MAX_TEXT_LEN) continue
            if (inCorpus.has(n) || seen.has(n)) continue
            seen.add(n)
            fresh.push({ feeling: job.feeling, text })
          }
        } catch (err) {
          console.warn(`\n  gen batch (${job.feeling}) failed: ${err}`)
        }
        genBar.increment()
      }),
    )
    await genQ.onIdle()
    genBar.stop()
    console.log(`  ${fresh.length} fresh candidates`)

    const kept = await verify(fresh, feelings)
    let added = 0
    for (const c of kept) {
      const bucket = accepted.get(c.feeling)!
      if (bucket.length < target.get(c.feeling)!) {
        bucket.push(c)
        added++
      }
    }
    console.log(
      `  ${kept.length} passed verification, ${added} added  ` +
      `(totals: ${feelings.map((f) => `${f}=${accepted.get(f)!.length}`).join(" ")})`,
    )
  }

  const short = feelings.filter((f) => accepted.get(f)!.length < target.get(f)!)
  if (short.length) {
    throw new Error(
      `ran out of rounds; short: ` +
      short.map((f) => `${f}=${accepted.get(f)!.length}/${target.get(f)}`).join(" "),
    )
  }

  const rows = feelings.flatMap((f) =>
    accepted.get(f)!.slice(0, target.get(f)!).map((c) => ({ ...c })),
  )
  const emojiItems = rows.map((r, id) => ({ id, text: r.text }))
  const emojiBatches = chunk(emojiItems, EMOJI_BATCH_SIZE)

  const emBar = new cliProgress.SingleBar(
    { format: "emoji      |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s" },
    cliProgress.Presets.shades_classic,
  )
  emBar.start(emojiBatches.length, 0)

  const emojiById = new Map<number, string>()
  const emQ = new PQueue({ concurrency: EMOJI_CONCURRENCY })
  emQ.addAll(
    emojiBatches.map((batch) => async () => {
      const got = await emojiBatch(batch, palette)
      for (const [id, e] of got) emojiById.set(id, e)
      emBar.increment()
    }),
  )
  await emQ.onIdle()
  emBar.stop()

  const paletteSet = new Set(palette)
  let fallbacks = 0
  const out: Row[] = rows.map((r, id) => {
    let emoji = emojiById.get(id)
    if (!emoji || !paletteSet.has(emoji)) {
      emoji = paletteSet.has(FALLBACK_EMOJI[r.feeling])
        ? FALLBACK_EMOJI[r.feeling]
        : palette[0]
      fallbacks++
    }
    return { text: r.text, feeling: r.feeling, emoji }
  })

  await writeFile(EVAL, out.map((r) => JSON.stringify(r)).join("\n") + "\n")

  const perFeeling = new Map<string, number>(feelings.map((f) => [f, 0]))
  let maxLen = 0
  let overlap = 0
  for (const r of out) {
    perFeeling.set(r.feeling, perFeeling.get(r.feeling)! + 1)
    const n = normalize(r.text)
    maxLen = Math.max(maxLen, n.length)
    if (inCorpus.has(n)) overlap++
  }

  console.log("\n--- eval.jsonl ---")
  console.log(`rows                 : ${out.length}`)
  console.log(
    `per feeling          : ${feelings.map((f) => `${f}=${perFeeling.get(f)}`).join(" ")}`,
  )
  console.log(`max normalized len   : ${maxLen} (limit ${MAX_TEXT_LEN})`)
  console.log(`overlap w/ data.jsonl: ${overlap} (must be 0)`)
  console.log(`emoji fallbacks used : ${fallbacks}`)
  process.exit(0)
}
