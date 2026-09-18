import { existsSync } from "node:fs"

import { cac } from "cac"
import { Box, render, Text } from "ink"
import React from "react"

import { langForText } from "../../web/src/scriptFonts.js"
import { LANGS, LANG_SET } from "./langs.ts"
import { splitEmojis } from "./emoji.ts"
import { readJsonl } from "./io.ts"
import { normalize } from "./normalize.ts"
import { STYLE_SET, STYLES } from "./styles.ts"

const h = React.createElement

const pct = (n: number, d: number) => (d ? Math.round((n / d) * 10000) / 100 : 0)

export type Disagreement = {
  text: string
  records: number
  uniqueEmojis: number
  emojis: string[]
}

export type ColorFit = {
  colorTexts: number
  multiColorTexts: number
  withinTextVariance: number
  datasetVariance: number
  r2Ceiling: number
}

export type Stats = {
  rawRows: number
  texts: number
  zeroEmojiTexts: number
  meanEmojisPerText: number
  meanStylesPerText: number
  meanColorsPerText: number
  distinctEmojis: number
  distinctStyles: number
  emojiPerText: { count: number; texts: number; pct: number; cumPct: number; tailPct: number }[]
  topEmojis: { emoji: string; texts: number; pct: number }[]
  styleCounts: { style: string; texts: number; pct: number }[]
  langCounts: { lang: string; texts: number; pct: number; tagged: number }[]
  textLen: { min: number; median: number; p90: number; max: number }
  lenHistogram: { len: number; texts: number }[]
  repeatBuckets: { label: string; texts: number }[]
  disagreement: Disagreement[]
  extraFields: { neg: number; meta: number; singleEmoji: number }
  colorFit: ColorFit
}

type Acc = {
  text: string
  emojis: Set<string>
  styles: Set<string>
  colors: Set<string>
  records: number
  taggedLang?: string
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0
  const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length))
  return sorted[i]
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace(/^#/, ""), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function colorVector(bg0: string, bg1: string, fg: string): number[] {
  return [...hexToRgb(bg0), ...hexToRgb(bg1), ...hexToRgb(fg)]
}

function vectorSetVariance(vecs: number[][]): number {
  const n = vecs.length
  const dims = vecs[0]?.length ?? 0
  let total = 0
  for (let d = 0; d < dims; d++) {
    let mean = 0
    for (const v of vecs) mean += v[d]
    mean /= n
    let variance = 0
    for (const v of vecs) variance += (v[d] - mean) ** 2
    total += variance / n
  }
  return total
}

export function computeStats(rawRows: unknown[]): Stats {
  const acc = new Map<string, Acc>()
  const extraFields = { neg: 0, meta: 0, singleEmoji: 0 }

  for (const raw of rawRows) {
    const row = (raw ?? {}) as Record<string, unknown>
    const text = typeof row.text === "string" ? row.text : ""
    const key = normalize(text)
    if (!key) continue

    let a = acc.get(key)
    if (!a) {
      a = { text, emojis: new Set(), styles: new Set(), colors: new Set(), records: 0 }
      acc.set(key, a)
    }
    a.records++
    if (typeof row.lang === "string" && LANG_SET.has(row.lang)) a.taggedLang = row.lang
    if (typeof row.emojis === "string") {
      for (const e of splitEmojis(row.emojis)) a.emojis.add(e)
    }
    if (Array.isArray(row.styles)) {
      for (const s of row.styles) {
        if (typeof s === "string" && STYLE_SET.has(s)) a.styles.add(s)
      }
    }
    if (Array.isArray(row.colors)) {
      for (const c of row.colors as unknown[]) {
        if (!c || typeof c !== "object") continue
        const bg = (c as Record<string, unknown>).bg
        const fg = (c as Record<string, unknown>).fg
        if (Array.isArray(bg) && bg.length >= 2 && typeof fg === "string") {
          a.colors.add(JSON.stringify([bg[0], bg[1], fg]))
        }
      }
    }

    if ("neg" in row) extraFields.neg++
    const meta = row.meta
    if (meta && typeof meta === "object") {
      extraFields.meta++
      if ((meta as Record<string, unknown>)["single-emoji"] === true) {
        extraFields.singleEmoji++
      }
    }
  }

  const keys = [...acc.values()]
  const texts = keys.length

  const emojiTexts = new Map<string, number>()
  const perCount = new Map<number, number>()
  const lenCount = new Map<number, number>()
  const styleTexts = new Map<string, number>()
  const langTexts = new Map<string, number>()
  const langTagged = new Map<string, number>()
  const repeat = { "1×": 0, "2×": 0, "3–5×": 0, "6+×": 0 }
  const lengths: number[] = []
  let totalEmojiTags = 0
  let totalStyleTags = 0
  let totalColorTags = 0
  let zeroEmojiTexts = 0

  for (const a of keys) {
    const n = a.emojis.size
    perCount.set(n, (perCount.get(n) ?? 0) + 1)
    totalEmojiTags += n
    if (n === 0) zeroEmojiTexts++
    for (const e of a.emojis) emojiTexts.set(e, (emojiTexts.get(e) ?? 0) + 1)

    totalStyleTags += a.styles.size
    for (const s of a.styles) styleTexts.set(s, (styleTexts.get(s) ?? 0) + 1)

    totalColorTags += a.colors.size

    const lang = a.taggedLang ?? langForText(a.text)
    langTexts.set(lang, (langTexts.get(lang) ?? 0) + 1)
    if (a.taggedLang) langTagged.set(lang, (langTagged.get(lang) ?? 0) + 1)

    const len = normalize(a.text).length
    lengths.push(len)
    lenCount.set(len, (lenCount.get(len) ?? 0) + 1)

    if (a.records === 1) repeat["1×"]++
    else if (a.records === 2) repeat["2×"]++
    else if (a.records <= 5) repeat["3–5×"]++
    else repeat["6+×"]++
  }

  let running = 0
  const emojiPerText = [...perCount.entries()]
    .sort((x, y) => x[0] - y[0])
    .map(([count, t]) => {
      const below = running
      running += t
      return {
        count,
        texts: t,
        pct: pct(t, texts),
        cumPct: pct(below, texts),
        tailPct: pct(texts - below, texts),
      }
    })

  const topEmojis = [...emojiTexts.entries()]
    .sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1))
    .map(([emoji, t]) => ({ emoji, texts: t, pct: pct(t, texts) }))

  const styleCounts = STYLES.map((style) => ({
    style,
    texts: styleTexts.get(style) ?? 0,
    pct: pct(styleTexts.get(style) ?? 0, texts),
  })).sort((x, y) => y.texts - x.texts || STYLES.indexOf(x.style) - STYLES.indexOf(y.style))

  const langCounts = LANGS.map((lang) => ({
    lang,
    texts: langTexts.get(lang) ?? 0,
    pct: pct(langTexts.get(lang) ?? 0, texts),
    tagged: langTagged.get(lang) ?? 0,
  })).sort((x, y) => y.texts - x.texts || LANGS.indexOf(x.lang) - LANGS.indexOf(y.lang))

  lengths.sort((x, y) => x - y)
  const textLen = {
    min: lengths[0] ?? 0,
    median: quantile(lengths, 0.5),
    p90: quantile(lengths, 0.9),
    max: lengths[lengths.length - 1] ?? 0,
  }

  const lenHistogram = [...lenCount.entries()]
    .sort((x, y) => y[0] - x[0])
    .map(([len, t]) => ({ len, texts: t }))

  const repeatBuckets = Object.entries(repeat).map(([label, t]) => ({ label, texts: t }))

  const datasetColorVecs: number[][] = []
  const withinTextVariances: number[] = []
  let colorTexts = 0
  let multiColorTexts = 0

  for (const a of keys) {
    if (a.colors.size === 0) continue
    colorTexts++
    const vecs = [...a.colors].map((s) => {
      const [bg0, bg1, fg] = JSON.parse(s) as [string, string, string]
      return colorVector(bg0, bg1, fg)
    })
    for (const v of vecs) datasetColorVecs.push(v)
    if (vecs.length >= 2) {
      multiColorTexts++
      withinTextVariances.push(vectorSetVariance(vecs))
    }
  }

  const withinTextVariance = withinTextVariances.length
    ? withinTextVariances.reduce((x, y) => x + y, 0) / withinTextVariances.length
    : 0
  const datasetVariance = vectorSetVariance(datasetColorVecs)
  const colorFit: ColorFit = {
    colorTexts,
    multiColorTexts,
    withinTextVariance,
    datasetVariance,
    r2Ceiling: datasetVariance ? Math.max(0, 1 - withinTextVariance / datasetVariance) : 0,
  }

  const disagreement = keys
    .map((a) => ({
      text: a.text,
      records: a.records,
      uniqueEmojis: a.emojis.size,
      emojis: [...a.emojis].sort(),
    }))
    .sort(
      (x, y) =>
        y.uniqueEmojis - x.uniqueEmojis ||
        y.records - x.records ||
        (x.text < y.text ? -1 : 1),
    )

  return {
    rawRows: rawRows.length,
    texts,
    zeroEmojiTexts,
    meanEmojisPerText: texts ? totalEmojiTags / texts : 0,
    meanStylesPerText: texts ? totalStyleTags / texts : 0,
    meanColorsPerText: texts ? totalColorTags / texts : 0,
    distinctEmojis: emojiTexts.size,
    distinctStyles: styleTexts.size,
    emojiPerText,
    topEmojis,
    styleCounts,
    langCounts,
    textLen,
    lenHistogram,
    repeatBuckets,
    disagreement,
    extraFields,
    colorFit,
  }
}

type Cell = string | number
type Align = "l" | "r"

function Table({
  head,
  rows,
  align = [],
}: {
  head: string[]
  rows: Cell[][]
  align?: Align[]
}) {
  const widths = head.map((hd, i) =>
    Math.max(String(hd).length, ...rows.map((r) => String(r[i] ?? "").length)),
  )
  const fmt = (v: Cell, i: number) => {
    const s = String(v ?? "")
    const pad = " ".repeat(Math.max(0, widths[i] - s.length))
    return (align[i] ?? "l") === "r" ? pad + s : s + pad
  }
  return h(
    Box,
    { flexDirection: "column" },
    h(
      Text,
      { color: "cyan", bold: true },
      head.map((hd, i) => fmt(hd, i)).join("  "),
    ),
    h(Text, { dimColor: true }, widths.map((w) => "─".repeat(w)).join("  ")),
    ...rows.map((r, ri) =>
      h(Text, { key: ri }, r.map((v, i) => fmt(v, i)).join("  ")),
    ),
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return h(
    Box,
    { flexDirection: "column", marginBottom: 1 },
    h(Text, { color: "yellow", bold: true }, title),
    children,
  )
}

function pctStr(n: number) {
  return `${n.toFixed(1)}%`
}

function App({
  stats,
  file,
  long,
}: {
  stats: Stats
  file: string
  long: boolean
}) {
  const s = stats

  const overviewRows: Cell[][] = [
    ["raw rows", s.rawRows, ""],
    ["distinct normalized texts", s.texts, pctStr(pct(s.texts, s.rawRows)) + " of raw"],
    ["duplicate rows", s.rawRows - s.texts, pctStr(pct(s.rawRows - s.texts, s.rawRows))],
    ["texts with 0 emojis", s.zeroEmojiTexts, pctStr(pct(s.zeroEmojiTexts, s.texts))],
    ["mean unique emojis / text", s.meanEmojisPerText.toFixed(2), ""],
    ["mean styles / text", s.meanStylesPerText.toFixed(2), ""],
    ["mean colors / text", s.meanColorsPerText.toFixed(2), ""],
    ["distinct emojis seen", s.distinctEmojis, ""],
    ["distinct styles seen", `${s.distinctStyles} / ${STYLES.length}`, ""],
    ["normalized length min/median/p90/max", `${s.textLen.min} / ${s.textLen.median} / ${s.textLen.p90} / ${s.textLen.max}`, ""],
  ]

  const emojiPerTextRows: Cell[][] = s.emojiPerText.map((r) => [
    r.count,
    r.texts,
    pctStr(r.pct),
    pctStr(r.cumPct),
    pctStr(r.tailPct),
  ])

  const styleRows: Cell[][] = s.styleCounts.map((r) => [r.style, r.texts, pctStr(r.pct)])

  const langRows: Cell[][] = s.langCounts.map((r) => [
    r.lang,
    r.texts,
    pctStr(r.pct),
    r.tagged,
    pctStr(pct(r.tagged, r.texts)),
  ])

  const colorFitRows: Cell[][] = [
    ["texts with colors", s.colorFit.colorTexts, ""],
    ["texts with ≥2 distinct colors", s.colorFit.multiColorTexts, pctStr(pct(s.colorFit.multiColorTexts, s.colorFit.colorTexts))],
    ["within-text color variance (RGB²)", s.colorFit.withinTextVariance.toFixed(1), "noise, from disagreeing texts"],
    ["dataset color variance (RGB²)", s.colorFit.datasetVariance.toFixed(1), "spread of all tagged colors"],
    ["color regressor R² ceiling", s.colorFit.r2Ceiling.toFixed(3), "1 − within/dataset variance"],
  ]

  const children = [
    h(Section, { key: "overview", title: `Corpus (${file}, collapsed by normalized text)` }, h(Table, { head: ["metric", "value", ""], rows: overviewRows, align: ["l", "r", "l"] })),
    h(Section, { key: "colorfit", title: "Color regressor R² ceiling (text-level label disagreement vs. dataset spread)" }, h(Table, { head: ["metric", "value", ""], rows: colorFitRows, align: ["l", "r", "l"] })),
    h(Section, { key: "epr", title: "Unique emojis per normalized text" }, h(Table, { head: ["# emojis", "# texts", "%", "cum %", "100−cum %"], rows: emojiPerTextRows, align: ["r", "r", "r", "r", "r"] })),
    h(Section, { key: "styles", title: "Styles by # texts" }, h(Table, { head: ["style", "# texts", "%"], rows: styleRows, align: ["l", "r", "r"] })),
    h(Section, { key: "langs", title: "Languages by # texts (tagged = explicit row.lang, rest detected from script)" }, h(Table, { head: ["lang", "# texts", "%", "tagged", "tagged %"], rows: langRows, align: ["l", "r", "r", "r", "r"] })),
  ]

  if (long) {
    const topEmojiRows: Cell[][] = s.topEmojis.map((r) => [r.emoji, r.texts, pctStr(r.pct)])
    const repeatRows: Cell[][] = s.repeatBuckets.map((r) => [r.label, r.texts, pctStr(pct(r.texts, s.texts))])
    const lenRows: Cell[][] = s.lenHistogram.map((r) => [
      r.len,
      r.texts,
      "▏".padEnd(Math.max(1, Math.round((r.texts / (s.lenHistogram[0]?.texts || 1)) * 40)), "█"),
    ])
    const extraRows: Cell[][] = [
      ['rows with "neg"', s.extraFields.neg, pctStr(pct(s.extraFields.neg, s.rawRows))],
      ['rows with "meta"', s.extraFields.meta, pctStr(pct(s.extraFields.meta, s.rawRows))],
      ["meta.single-emoji rows", s.extraFields.singleEmoji, pctStr(pct(s.extraFields.singleEmoji, s.rawRows))],
    ]
    const disRows: Cell[][] = s.disagreement
      .slice(0, 30)
      .map((r) => [
        r.text.length > 44 ? r.text.slice(0, 43) + "…" : r.text,
        r.records,
        r.uniqueEmojis,
        r.emojis.join(" "),
      ])
    children.push(
      h(Section, { key: "top", title: "Emoji frequency (all)" }, h(Table, { head: ["emoji", "# texts", "%"], rows: topEmojiRows, align: ["l", "r", "r"] })),
      h(Section, { key: "repeat", title: "Records per normalized text" }, h(Table, { head: ["bucket", "# texts", "%"], rows: repeatRows, align: ["l", "r", "r"] })),
      h(Section, { key: "len", title: "Normalized text length" }, h(Table, { head: ["len", "# texts", ""], rows: lenRows, align: ["r", "r", "l"] })),
      h(Section, { key: "extra", title: "Extra row fields" }, h(Table, { head: ["field", "count", "%"], rows: extraRows, align: ["l", "r", "r"] })),
      h(Section, { key: "dis", title: "Top 30 texts by emoji-label disagreement" }, h(Table, { head: ["text", "recs", "uniq", "emojis"], rows: disRows, align: ["l", "r", "r", "l"] })),
    )
  }

  return h(Box, { flexDirection: "column", marginTop: 1 }, ...children)
}

if (import.meta.main) {
  const cli = cac("stat")
  cli.usage("<file>  — statistics for one data JSONL file (e.g. data/data.jsonl)")
  cli.option("--long", "print more statistics")
  cli.help()
  const parsed = cli.parse(process.argv, { run: false })
  if (parsed.options.help) process.exit(0)

  const file = parsed.args[0] as string | undefined
  if (!file) {
    console.error("usage: bun tools/data/stat.ts <file>  (e.g. data/data.jsonl)")
    process.exit(1)
  }
  if (!existsSync(file)) {
    console.error(`${file} not found`)
    process.exit(1)
  }

  const rows = await readJsonl(file)
  const stats = computeStats(rows)

  const { unmount, waitUntilExit } = render(
    h(App, {
      stats,
      file,
      long: Boolean(parsed.options.long),
    }),
  )
  unmount()
  await waitUntilExit()
}
