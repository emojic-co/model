import { existsSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { cac } from "cac"
import { PINTEREST_STATE_JSON, PREVIEW_DIR } from "../../files.ts"
import { LATIN, SCRIPT_FONT_QUERY, scriptForLang } from "../../web/src/scriptFonts.js"
import { ensureFonts, FONTS_XDG_DATA_HOME, type FontCache } from "../cli/fonts.ts"
import { cardSvg, type Row } from "../cli/svg-card.ts"
import { LANGS, type Lang } from "../data/langs.ts"
import { writeFileAtomic } from "../data/io.ts"
import { cardHtml, page, stamp } from "../data/preview-card.ts"

const PINTEREST_API = "https://api.pinterest.com/v5"
const LINK = "https://emojify.ing"
const RESOLUTION = 1024
const JPEG_QUALITY = 90

type EngagingCard = {
  id: string
  style: string
  emoji: string
  bg: [string, string]
  fg: string
  text: Record<Lang, string>
}

const CARDS: EngagingCard[] = [
  {
    id: "favorite-chaos",
    style: "Joyful",
    emoji: "😂",
    bg: ["#ffe29a", "#ff9a5a"],
    fg: "#3a1c02",
    text: { en: "This is my favorite kind of chaos", he: "זה סוג הכאוס האהוב עליי" },
  },
  {
    id: "main-character",
    style: "Playful",
    emoji: "✨",
    bg: ["#c9a7ff", "#7b5cff"],
    fg: "#1c0f3a",
    text: { en: "Main character energy", he: "אנרגיה של דמות ראשית" },
  },
  {
    id: "one-breath",
    style: "Serene",
    emoji: "🌿",
    bg: ["#bfe8d8", "#7fcbb0"],
    fg: "#0d2b21",
    text: { en: "Taking it one breath at a time", he: "לוקחים את זה נשימה אחת בכל פעם" },
  },
  {
    id: "doing-it-scared",
    style: "Determined",
    emoji: "💪",
    bg: ["#ffd166", "#ef8354"],
    fg: "#3a1c02",
    text: { en: "Doing it scared", he: "עושים את זה למרות הפחד" },
  },
  {
    id: "little-things",
    style: "Tender",
    emoji: "🥹",
    bg: ["#ffc9de", "#ff8fb1"],
    fg: "#3a0a1c",
    text: { en: "Grateful for the little things", he: "אסירי תודה על הדברים הקטנים" },
  },
  {
    id: "plot-twist",
    style: "Whimsical",
    emoji: "🌀",
    bg: ["#a7f0e5", "#5cd6c0"],
    fg: "#062e28",
    text: { en: "Plot twist: I'm actually fine", he: "טוויסט בעלילה: אני בעצם בסדר" },
  },
  {
    id: "old-version",
    style: "Wistful",
    emoji: "🌙",
    bg: ["#c7d2ff", "#8a9cff"],
    fg: "#0d1440",
    text: { en: "Missing an old version of myself", he: "מתגעגעים לגרסה ישנה של עצמנו" },
  },
  {
    id: "cant-stop-smiling",
    style: "Excited",
    emoji: "😆",
    bg: ["#ffe5a0", "#ffb84d"],
    fg: "#3a2400",
    text: { en: "Can't stop smiling today", he: "לא מפסיקים לחייך היום" },
  },
]

const TAGLINE: Record<Lang, string> = {
  en: "Turn any text into an emoji mood card — free at emojify.ing",
  he: "הופכים כל טקסט לכרטיס אימוג׳י — בחינם באתר emojify.ing",
}

type StateEntry = { pinId: string; url: string; boardId: string; publishedAt: string }
type State = Record<string, StateEntry>

function stateKey(card: EngagingCard, lang: Lang): string {
  return `${card.id}:${lang}`
}

async function loadState(): Promise<State> {
  if (!existsSync(PINTEREST_STATE_JSON)) return {}
  return JSON.parse(await readFile(PINTEREST_STATE_JSON, "utf8")) as State
}

async function saveState(state: State): Promise<void> {
  await mkdir(PINTEREST_STATE_JSON.slice(0, PINTEREST_STATE_JSON.lastIndexOf("/")), { recursive: true })
  await writeFileAtomic(PINTEREST_STATE_JSON, `${JSON.stringify(state, null, 2)}\n`)
}

function scriptQueriesFor(langs: readonly Lang[]): string[] {
  const scripts = new Set(langs.map(scriptForLang).filter((s) => s !== LATIN))
  return [...scripts].map((s) => SCRIPT_FONT_QUERY[s]).filter((q): q is string => Boolean(q))
}

async function renderJpeg(card: EngagingCard, lang: Lang, fonts: FontCache): Promise<Uint8Array> {
  const row: Row = {
    text: card.text[lang],
    emojis: card.emoji,
    styles: [card.style],
    colors: [{ bg: card.bg, fg: card.fg }],
    lang,
  }
  const svg = cardSvg(row, RESOLUTION, fonts)
  const proc = Bun.spawn(["rsvg-convert", "-f", "png", "-"], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, XDG_DATA_HOME: FONTS_XDG_DATA_HOME },
  })
  proc.stdin.write(svg)
  proc.stdin.end()
  const [png, code] = await Promise.all([new Response(proc.stdout).arrayBuffer(), proc.exited])
  if (code !== 0) {
    const err = await new Response(proc.stderr).text()
    throw new Error(`rsvg-convert exited ${code}: ${err}`)
  }
  return await new Bun.Image(png).jpeg({ quality: JPEG_QUALITY }).bytes()
}

async function publishPin(opts: {
  token: string
  boardId: string
  imageJpeg: Uint8Array
  title: string
  description: string
  altText: string
}): Promise<{ id: string; url: string }> {
  const res = await fetch(`${PINTEREST_API}/pins`, {
    method: "POST",
    headers: { Authorization: `Bearer ${opts.token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      board_id: opts.boardId,
      media_source: {
        source_type: "image_base64",
        content_type: "image/jpeg",
        data: Buffer.from(opts.imageJpeg).toString("base64"),
      },
      title: opts.title,
      description: opts.description,
      link: LINK,
      alt_text: opts.altText,
    }),
  })
  if (!res.ok) throw new Error(`pinterest: ${res.status} ${await res.text()}`)
  const body = (await res.json()) as { id: string }
  return { id: body.id, url: `https://www.pinterest.com/pin/${body.id}/` }
}

function boardIdFor(lang: Lang): string | undefined {
  return process.env[`PINTEREST_BOARD_ID_${lang.toUpperCase()}`] ?? process.env.PINTEREST_BOARD_ID
}

const PREVIEW_EXTRA_CSS = `
body { place-items: start center; padding: 2em 1em; }
.preview-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1em;
  width: 100%;
  max-width: 1200px;
}
.preview-grid .card { max-width: none; aspect-ratio: 1; }
`

async function buildPreviewHtml(): Promise<string> {
  const cards = LANGS.flatMap((lang) =>
    CARDS.map((c) =>
      cardHtml({
        text: c.text[lang],
        emoji: c.emoji,
        feeling: c.style,
        lang,
        colors: { bg1: c.bg[0], bg2: c.bg[1], text_color: c.fg },
      }),
    ),
  ).join("\n")
  const extraHead = scriptQueriesFor(LANGS)
    .map((q) => `<link href="https://fonts.googleapis.com/css2?${q}&display=swap" rel="stylesheet" />`)
    .join("\n")
  return page({
    title: `pinterest preview — ${CARDS.length} cards × ${LANGS.length} languages`,
    extraCss: PREVIEW_EXTRA_CSS,
    body: `<div class="preview-grid">\n${cards}\n</div>`,
    extraHead,
  })
}

async function runPreview(): Promise<void> {
  await mkdir(PREVIEW_DIR, { recursive: true })
  const dest = `${PREVIEW_DIR}/pinterest-${stamp()}.html`
  await writeFile(dest, await buildPreviewHtml())
  console.log(dest)
}

async function runStatus(): Promise<void> {
  const state = await loadState()
  for (const card of CARDS) {
    for (const lang of LANGS) {
      const entry = state[stateKey(card, lang)]
      console.log(entry ? `${stateKey(card, lang)}  published  ${entry.url}` : `${stateKey(card, lang)}  pending`)
    }
  }
}

async function runPublish(opts: { dry: boolean; force: boolean; lang?: string; card?: string }): Promise<void> {
  const langs = opts.lang ? [opts.lang as Lang] : [...LANGS]
  for (const lang of langs) {
    if (!LANGS.includes(lang)) throw new Error(`pinterest: unsupported --lang ${JSON.stringify(lang)}, expected one of ${LANGS.join(", ")}`)
  }
  const targets = opts.card ? CARDS.filter((c) => c.id === opts.card) : CARDS
  if (opts.card && !targets.length) throw new Error(`pinterest: no card with id ${JSON.stringify(opts.card)}`)

  const token = process.env.PINTEREST_ACCESS_TOKEN
  if (!opts.dry && !token) throw new Error("pinterest: set PINTEREST_ACCESS_TOKEN")

  const state = await loadState()
  const fonts = await ensureFonts(scriptQueriesFor(langs))

  let published = 0
  let skipped = 0
  for (const card of targets) {
    for (const lang of langs) {
      const key = stateKey(card, lang)
      if (state[key] && !opts.force) {
        skipped++
        continue
      }
      const boardId = boardIdFor(lang)
      if (!opts.dry && !boardId) {
        throw new Error(`pinterest: set PINTEREST_BOARD_ID (or PINTEREST_BOARD_ID_${lang.toUpperCase()})`)
      }
      const jpeg = await renderJpeg(card, lang, fonts)
      if (opts.dry) {
        console.log(`dry-run: would publish ${key} (${jpeg.length} bytes)`)
        continue
      }
      const result = await publishPin({
        token: token!,
        boardId: boardId!,
        imageJpeg: jpeg,
        title: card.text[lang],
        description: TAGLINE[lang],
        altText: `${card.text[lang]} ${card.emoji} — emojify.ing mood card`,
      })
      state[key] = { pinId: result.id, url: result.url, boardId: boardId!, publishedAt: new Date().toISOString() }
      await saveState(state)
      published++
      console.log(`published ${key} -> ${result.url}`)
    }
  }
  console.log(`done: ${published} published, ${skipped} already published`)
}

const cli = cac("pinterest")

cli
  .command("preview", "render every predefined card in every supported language to an HTML preview page")
  .action(runPreview)

cli
  .command("publish", "publish predefined cards to pinterest.com in every supported language")
  .option("--dry", "render and validate without calling the Pinterest API")
  .option("--force", "republish cards already marked as published")
  .option("--lang <lang>", `limit to one language (${LANGS.join(", ")})`)
  .option("--card <id>", "limit to one predefined card id")
  .action((options) =>
    runPublish({
      dry: Boolean(options.dry),
      force: Boolean(options.force),
      lang: options.lang,
      card: options.card,
    }),
  )

cli.command("status", "list which predefined cards have been published, per language").action(runStatus)

cli.help()

if (import.meta.main) {
  cli.parse()
}
