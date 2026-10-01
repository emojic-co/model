import { useCallback, useRef, useState } from 'react'
import { fitCanvasFont, wrapLines } from '../fit'
import { resolveFeeling } from '../feelings'
import { contrastRatio, patternTint, toCssOklab, toHexColor, BLACK, WHITE } from '../model'
import { lineUnits, shimmerTimeline, textTimeline } from '../cardAnim'
import { patternLayers } from '../patterns'
import { downloadBlob, renderGif } from '../cardGif'
import { mp4Supported, renderMp4 } from '../cardMp4'
import { ensureScriptFontsLoaded, scriptForLang } from '../scriptFonts'

const S = 512
const WATERMARK = 'emojify.ing'
const EMOJI_STACK = '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif'

// Card layout ratios, relative to card size. Shared with tools/data/export-style.ts,
// which bakes these into style.yml for the Android app to mirror.
export const RATIOS = {
  padRatio: 0.07,
  gapRatio: 0.06,
  emojiRatio: 0.32,
  emojiDyRatio: 0.065,
  textBoxPadXRatio: 0.03,
  textBoxPadYRatio: 0.05,
  textLineHeight: 1.5,
  textMinRatio: 0.05,
  textMaxRatio: 0.21,
  maxLines: 10,
  watermarkPxRatio: 0.044,
  watermarkOpacity: 0.28,
  watermarkMarginRatio: 0.055,
}

// CSS custom properties for the live DOM card, derived from the same ratios (see .card in styles.css).
export const CARD_CSS_VARS = {
  '--card-pad': `${RATIOS.padRatio * 100}%`,
  '--card-gap': `${RATIOS.gapRatio * 100}%`,
  '--card-emoji': `${RATIOS.emojiRatio * 100}cqw`,
  '--card-emoji-dy': `${RATIOS.emojiDyRatio * 100}cqw`,
  '--card-text-box-pad': `${RATIOS.textBoxPadYRatio * 100}% ${RATIOS.textBoxPadXRatio * 100}%`,
  '--card-text-box-max-h': `${(1 - 2 * RATIOS.padRatio - RATIOS.emojiRatio - RATIOS.gapRatio) * 100}cqw`,
}

const WATERMARK_PX = Math.round(RATIOS.watermarkPxRatio * S)
const PAD = RATIOS.padRatio * S
const GAP = RATIOS.gapRatio * S
const EMOJI_PX = RATIOS.emojiRatio * S
const EMOJI_DY = RATIOS.emojiDyRatio * S
const TEXT_BOX_PAD_X = RATIOS.textBoxPadXRatio * S
const TEXT_BOX_PAD_Y = RATIOS.textBoxPadYRatio * S
const TEXT_LINE_HEIGHT = RATIOS.textLineHeight
const TEXT_MIN_PX = Math.round(RATIOS.textMinRatio * S)
const TEXT_MAX_PX = Math.round(RATIOS.textMaxRatio * S)
const MAX_LINES = RATIOS.maxLines

async function ensureFonts(stack, emoji) {
  if (!document.fonts) return
  const jobs = [
    document.fonts.load(`400 120px "Noto Color Emoji"`, emoji),
    document.fonts.load(`700 ${WATERMARK_PX}px "Caveat"`, WATERMARK),
  ]
  const name = stack.match(/"([^"]+)"/)?.[1]
  if (name) {
    jobs.push(document.fonts.load(`600 24px "${name}"`))
    jobs.push(document.fonts.load(`400 13px "${name}"`))
  }
  try {
    await Promise.all(jobs)
  } catch {}
}

function patternUrl(cssValue) {
  return cssValue.match(/^url\((['"]?)(.*)\1\)$/)[2]
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

function scaledTile(img, w, h) {
  const tile = document.createElement('canvas')
  tile.width = Math.max(1, Math.round(w))
  tile.height = Math.max(1, Math.round(h))
  tile.getContext('2d').drawImage(img, 0, 0, tile.width, tile.height)
  return tile
}

// Lays the card out once and returns paint(drawEmoji): draws the static background, text and watermark,
// with the emoji drawn by the callback (default: the plain glyph) so animated exports can vary it per frame.
export async function createPainter({ text, emoji, feeling, lang, colors }, scale = 2) {
  ensureScriptFontsLoaded(scriptForLang(lang))
  const stack = resolveFeeling(feeling, lang).font
  const st = resolveFeeling(feeling, lang).style
  const fw = st.fontWeight ?? 600
  const fitalic = st.fontStyle === 'italic' ? 'italic ' : ''
  const headline = st.textTransform === 'uppercase' ? text.toUpperCase() : text
  await ensureFonts(stack, emoji)

  const canvas = document.createElement('canvas')
  canvas.width = S * scale
  canvas.height = S * scale
  const ctx = canvas.getContext('2d')
  ctx.scale(scale, scale)

  const grad = ctx.createLinearGradient(0, 0, S, S)
  grad.addColorStop(0, toCssOklab(colors.bg1))
  grad.addColorStop(1, toCssOklab(colors.bg2))
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, S, S)

  const layers = patternLayers(feeling, toHexColor(patternTint(colors.bg1, colors.bg2)))
  for (const layer of layers) {
    const img = await loadImage(patternUrl(layer.image))
    const tile = scaledTile(img, layer.w * S, layer.h * S)
    ctx.fillStyle = ctx.createPattern(tile, 'repeat')
    ctx.fillRect(0, 0, S, S)
  }

  const bg = canvas
  const out = document.createElement('canvas')
  out.width = canvas.width
  out.height = canvas.height
  const octx = out.getContext('2d')

  // Animated export state: built on first animated paint, once the text has been laid out.
  let motion = null

  const paint = (drawEmoji, opts) => {
    octx.setTransform(1, 0, 0, 1, 0, 0)
    octx.clearRect(0, 0, out.width, out.height)
    octx.drawImage(bg, 0, 0)
    octx.scale(scale, scale)
    paintContent(octx, drawEmoji, opts)
    return out
  }
  // Timings of the entrance + shimmer for the laid-out text (needs one paint to measure).
  paint.size = { width: canvas.width, height: canvas.height }
  paint.motion = () => {
    if (!motion) paint(null, { animate: true, timeMs: 0 })
    return motion
  }

  const paintContent = (ctx, drawEmoji, opts) => {
    ctx.fillStyle = toCssOklab(colors.text_color)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    const textBoxTop = PAD + EMOJI_PX + GAP
    const maxWidth = S - 2 * PAD - 2 * TEXT_BOX_PAD_X
    const maxHeight = S - PAD - textBoxTop - 2 * TEXT_BOX_PAD_Y

    const widthAt = (str, px) => {
      ctx.font = `${fitalic}${fw} ${px}px ${stack}`
      return ctx.measureText(str).width
    }
    if ('letterSpacing' in ctx) ctx.letterSpacing = st.letterSpacing ?? '0px'
    const fpx = fitCanvasFont({
      text: headline,
      maxWidth,
      maxHeight,
      min: TEXT_MIN_PX,
      max: TEXT_MAX_PX,
      lineHeight: TEXT_LINE_HEIGHT,
      widthAt,
    })

    ctx.font = `${fitalic}${fw} ${fpx}px ${stack}`
    const lines = wrapLines((str) => ctx.measureText(str).width, headline, maxWidth, MAX_LINES)
    // Emoji + text form one group, centered vertically on the card.
    const blockH = lines.length * fpx * TEXT_LINE_HEIGHT
    const top = (S - (EMOJI_PX + GAP + 2 * TEXT_BOX_PAD_Y + blockH)) / 2
    const emojiCenterY = top + EMOJI_PX / 2 + EMOJI_DY
    const textCenterY = top + EMOJI_PX + GAP + TEXT_BOX_PAD_Y + blockH / 2
    const animate = !!opts?.animate
    let unitLines = null
    if (animate) {
      unitLines = lines.map(lineUnits)
      if (!motion) {
        const n = unitLines.reduce((k, u) => k + u.length, 0)
        const r = resolveFeeling(feeling, lang)
        const text = textTimeline(r.entrance, feeling, n)
        const shimmer = shimmerTimeline(r.cluster, feeling)
        motion = { text, shimmer, shimmerStartMs: text.totalMs + shimmer.startDelayMs }
      }
      motion.shimmer.draw(ctx, S, opts.timeMs - motion.shimmerStartMs)
    }
    if (drawEmoji) drawEmoji(ctx, S / 2, emojiCenterY, EMOJI_PX)
    else {
      ctx.font = `${EMOJI_PX}px ${EMOJI_STACK}`
      ctx.fillText(emoji, S / 2, emojiCenterY)
    }
    ctx.font = `${fitalic}${fw} ${fpx}px ${stack}`
    let ty = textCenterY - ((lines.length - 1) * fpx * TEXT_LINE_HEIGHT) / 2
    ctx.globalAlpha = st.opacity ?? 1
    if ('direction' in ctx) ctx.direction = lang === 'he' ? 'rtl' : 'ltr'
    const rtl = lang === 'he'
    let unitIndex = 0
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li]
      if (!animate) {
        ctx.fillText(line, S / 2, ty)
      } else {
        // Each unit is drawn at its natural position in the line, posed by the entrance animation.
        const width = ctx.measureText(line).width
        for (const [a, b] of unitLines[li]) {
          const mid = (ctx.measureText(line.slice(0, a)).width + ctx.measureText(line.slice(0, b)).width) / 2
          const cx = rtl ? S / 2 + width / 2 - mid : S / 2 - width / 2 + mid
          const p = motion.text.pose(unitIndex++, opts.timeMs)
          ctx.save()
          ctx.translate(cx + p.x * fpx, ty + p.y * fpx)
          ctx.rotate((p.rotate * Math.PI) / 180)
          ctx.scale(p.scale, p.scale * p.scaleY)
          ctx.globalAlpha = (st.opacity ?? 1) * p.opacity
          ctx.fillText(line.slice(a, b), 0, 0)
          ctx.restore()
        }
      }
      ty += fpx * TEXT_LINE_HEIGHT
    }
    if ('direction' in ctx) ctx.direction = 'ltr'
    ctx.globalAlpha = 1
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0px'

    ctx.save()
    ctx.textAlign = 'right'
    ctx.textBaseline = 'alphabetic'
    ctx.font = `700 ${WATERMARK_PX}px "Caveat", ui-sans-serif, sans-serif`
    ctx.fillStyle =
      contrastRatio(BLACK, colors.bg2) >= contrastRatio(WHITE, colors.bg2)
        ? '#000000'
        : '#ffffff'
    ctx.globalAlpha = RATIOS.watermarkOpacity
    ctx.fillText(WATERMARK, S - RATIOS.watermarkMarginRatio * S, S - RATIOS.watermarkMarginRatio * S)
    ctx.restore()
  }

  return paint
}

async function render(data) {
  const out = (await createPainter(data))()
  return new Promise((resolve, reject) => {
    out.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png')
  })
}

export function useCardImage(cardData, showToast) {
  return useCallback(async () => {
    if (!cardData) {
      showToast('nothing to copy yet')
      return
    }
    try {
      const blob = await render(cardData)
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      showToast('copied to clipboard ✓')
    } catch (err) {
      console.error(err)
      showToast('copy failed')
    }
  }, [cardData, showToast])
}

// GIF/MP4 export with progress and cancel. `busy` is the format being made (null when idle).
export function useCardExport(cardData, showToast) {
  const [state, setState] = useState({ busy: null, progress: 0 })
  const abort = useRef(null)

  const start = useCallback(
    async (format) => {
      if (!cardData || abort.current) return
      const ctl = new AbortController()
      abort.current = ctl
      setState({ busy: format, progress: 0 })
      const render = format === 'gif' ? renderGif : renderMp4
      try {
        const blob = await render(cardData, {
          signal: ctl.signal,
          onProgress: (progress) => setState({ busy: format, progress }),
        })
        downloadBlob(blob, `emojify.${format}`)
        showToast(`${format} saved ✓`)
      } catch (err) {
        if (err?.name === 'AbortError') showToast('cancelled')
        else {
          console.error(err)
          showToast(`${format} failed`)
        }
      } finally {
        abort.current = null
        setState({ busy: null, progress: 0 })
      }
    },
    [cardData, showToast],
  )

  const cancel = useCallback(() => abort.current?.abort(), [])
  return { ...state, saveGif: useCallback(() => start('gif'), [start]), saveMp4: useCallback(() => start('mp4'), [start]), cancel }
}

export { mp4Supported }
