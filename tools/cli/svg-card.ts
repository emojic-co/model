import { openSync } from "fontkit"

import { resolveFeeling } from "../../web/src/feelings.js"
import { fitCanvasFont, wrapLines } from "../../web/src/fit.js"
import { patternTint } from "../../web/src/model.js"
import { patternLayers } from "../../web/src/patterns.js"
import type { FontCache } from "./fonts.ts"
import { resolveEmojiSvg } from "./emoji-svg.ts"

const S = 512
const PAD = 0.07 * S
const GAP = 0.03 * S
const EMOJI_PX = 0.32 * S
const EMOJI_DY = 0.065 * S
const TEXT_BOX_PAD_X = 0.03 * S
const TEXT_BOX_PAD_Y = 0.05 * S
const TEXT_LINE_HEIGHT = 1.5
const TEXT_MIN_PX = Math.round(0.05 * S)
const TEXT_MAX_PX = Math.round(0.13 * S)
const MAX_LINES = 10
const REFERENCE_PX = 600

export type Row = {
  text: string
  emojis: string
  styles: string[]
  colors: { bg: [string, string]; fg: string }[]
  lang?: string
}

export function firstEmoji(field: string): string {
  return field.trim().split(/\s+/)[0] ?? ""
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

function primaryFamily(fontStack: string): string {
  return fontStack.match(/^"([^"]+)"/)?.[1] ?? fontStack.split(",")[0].trim()
}

const fontCache = new Map<string, ReturnType<typeof openSync>>()

function openFont(path: string): ReturnType<typeof openSync> {
  let font = fontCache.get(path)
  if (!font) {
    font = openSync(path)
    fontCache.set(path, font)
  }
  return font
}

function widthAtFn(fontPath: string | undefined, letterSpacingPx: (px: number) => number) {
  if (!fontPath) return (str: string, px: number) => str.length * px * 0.55 + letterSpacingPx(px)
  const font = openFont(fontPath)
  return (str: string, px: number) => {
    const run = font.layout(str)
    return (run.advanceWidth / font.unitsPerEm) * px + letterSpacingPx(px) * str.length
  }
}

function decodePatternTile(cssUrl: string): { href: string; w: number; h: number } {
  const encoded = cssUrl.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)?.[2] ?? ""
  const decoded = decodeURIComponent(encoded)
  const tag = decoded.match(/<svg[^>]*>/)?.[0]
  const w = Number(tag?.match(/width="([\d.]+)"/)?.[1] ?? REFERENCE_PX)
  const h = Number(tag?.match(/height="([\d.]+)"/)?.[1] ?? REFERENCE_PX)
  return { href: `data:image/svg+xml,${encoded}`, w, h }
}

export function cardSvg(row: Row, resolution: number, fonts: FontCache): string {
  const feeling = row.styles[0] ?? "Neutral"
  const r = resolveFeeling(feeling, row.lang)
  const st = r.style ?? {}
  const fontWeight = st.fontWeight ?? 600
  const fontStyle = st.fontStyle === "italic" ? "italic" : "normal"
  const family = primaryFamily(r.font)
  const headline = st.textTransform === "uppercase" ? row.text.toUpperCase() : row.text
  const letterSpacingEm = st.letterSpacing ? Number.parseFloat(st.letterSpacing) : 0
  const opacity = st.opacity ?? 1

  const palette = row.colors[0]
  const colors = { bg1: palette.bg[0], bg2: palette.bg[1], text_color: palette.fg }
  const layers = patternLayers(feeling, undefined, undefined, patternTint(colors.bg1, colors.bg2))

  const fontPath = fonts.fileFor(family, fontWeight, fontStyle)
  const widthAt = widthAtFn(fontPath, (px) => letterSpacingEm * px)

  const emojiBoxBottom = PAD + EMOJI_PX
  const emojiCenterY = PAD + EMOJI_PX / 2 + EMOJI_DY
  const textBoxTop = emojiBoxBottom + GAP
  const textBoxBottom = S - PAD
  const textCenterY = (textBoxTop + textBoxBottom) / 2
  const maxWidth = S - 2 * PAD - 2 * TEXT_BOX_PAD_X
  const maxHeight = textBoxBottom - textBoxTop - 2 * TEXT_BOX_PAD_Y

  const fpx = fitCanvasFont({
    text: headline,
    maxWidth,
    maxHeight,
    min: TEXT_MIN_PX,
    max: TEXT_MAX_PX,
    lineHeight: TEXT_LINE_HEIGHT,
    widthAt,
  })
  const lines: string[] = wrapLines((s: string) => widthAt(s, fpx), headline, maxWidth, MAX_LINES)
  const ty0 = textCenterY - ((lines.length - 1) * fpx * TEXT_LINE_HEIGHT) / 2

  const textEls = lines
    .map((line, i) => {
      const y = ty0 + i * fpx * TEXT_LINE_HEIGHT
      return (
        `<text x="${S / 2}" y="${y}" text-anchor="middle" dominant-baseline="central" `
        + `font-family="${esc(r.font)}" font-weight="${fontWeight}" font-style="${fontStyle}" `
        + `letter-spacing="${letterSpacingEm}em" font-size="${fpx}" fill="${colors.text_color}" `
        + `opacity="${opacity}">${esc(line)}</text>`
      )
    })
    .join("\n")

  const emoji = firstEmoji(row.emojis)
  const emojiSvg = resolveEmojiSvg(emoji)
  const ex = S / 2 - EMOJI_PX / 2
  const ey = emojiCenterY - EMOJI_PX / 2
  const emojiEl = emojiSvg
    ? `<svg x="${ex}" y="${ey}" width="${EMOJI_PX}" height="${EMOJI_PX}" `
      + `viewBox="0 0 ${emojiSvg.width} ${emojiSvg.height}">${emojiSvg.body}</svg>`
    : `<text x="${S / 2}" y="${emojiCenterY}" text-anchor="middle" dominant-baseline="central" `
      + `font-size="${EMOJI_PX * 0.8}">${esc(emoji)}</text>`

  const patternDefs = layers
    .map((layer, i) => {
      const tile = decodePatternTile(layer.image)
      return (
        `<pattern id="pat${i}" patternUnits="objectBoundingBox" width="${layer.w}" height="${layer.h}" `
        + `viewBox="0 0 ${tile.w} ${tile.h}">`
        + `<image href="${tile.href}" x="0" y="0" width="${tile.w}" height="${tile.h}"/>`
        + `</pattern>`
      )
    })
    .join("\n")
  const patternRects = layers.map((_, i) => `<rect width="${S}" height="${S}" fill="url(#pat${i})"/>`).join("\n")

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${resolution}" height="${resolution}" viewBox="0 0 ${S} ${S}">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="${colors.bg1}"/>
<stop offset="1" stop-color="${colors.bg2}"/>
</linearGradient>
${patternDefs}
</defs>
<rect width="${S}" height="${S}" fill="url(#bg)"/>
${patternRects}
${emojiEl}
${textEls}
</svg>
`
}
