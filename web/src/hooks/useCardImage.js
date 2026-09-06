import { useCallback } from 'react'
import { fitCanvasFont, wrapLines } from '../fit'
import { resolveFeeling } from '../feelings'

const S = 512
const EMOJI_STACK = '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif'

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

async function ensureFonts(stack, emoji) {
  if (!document.fonts) return
  const jobs = [document.fonts.load(`400 120px "Noto Color Emoji"`, emoji)]
  const name = stack.match(/"([^"]+)"/)?.[1]
  if (name) {
    jobs.push(document.fonts.load(`600 24px "${name}"`))
    jobs.push(document.fonts.load(`400 13px "${name}"`))
  }
  try {
    await Promise.all(jobs)
  } catch {}
}

async function render({ text, emoji, feeling, colors }) {
  const stack = resolveFeeling(feeling).font
  const st = resolveFeeling(feeling).style
  const fw = st.fontWeight ?? 600
  const fitalic = st.fontStyle === 'italic' ? 'italic ' : ''
  const headline = st.textTransform === 'uppercase' ? text.toUpperCase() : text
  await ensureFonts(stack, emoji)

  const scale = 2
  const canvas = document.createElement('canvas')
  canvas.width = S * scale
  canvas.height = S * scale
  const ctx = canvas.getContext('2d')
  ctx.scale(scale, scale)

  const grad = ctx.createLinearGradient(0, 0, S, S)
  grad.addColorStop(0, colors.bg1)
  grad.addColorStop(1, colors.bg2)
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, S, S)

  ctx.fillStyle = colors.text_color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  const emojiBoxBottom = PAD + EMOJI_PX
  const emojiCenterY = PAD + EMOJI_PX / 2 + EMOJI_DY
  const textBoxTop = emojiBoxBottom + GAP
  const textBoxBottom = S - PAD
  const textCenterY = (textBoxTop + textBoxBottom) / 2
  const maxWidth = S - 2 * PAD - 2 * TEXT_BOX_PAD_X
  const maxHeight = textBoxBottom - textBoxTop - 2 * TEXT_BOX_PAD_Y

  ctx.font = `${EMOJI_PX}px ${EMOJI_STACK}`
  ctx.fillText(emoji, S / 2, emojiCenterY)

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
  let ty = textCenterY - ((lines.length - 1) * fpx * TEXT_LINE_HEIGHT) / 2
  ctx.globalAlpha = st.opacity ?? 1
  for (const line of lines) {
    ctx.fillText(line, S / 2, ty)
    ty += fpx * TEXT_LINE_HEIGHT
  }
  ctx.globalAlpha = 1
  if ('letterSpacing' in ctx) ctx.letterSpacing = '0px'

  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png')
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
