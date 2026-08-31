import { useCallback } from 'react'
import { fitCanvasFont, wrapLines } from '../fit'
import { resolveFeeling } from '../feelings'

const S = 512
const EMOJI_STACK = '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif'

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

async function render({ text, emoji, feeling, pal }) {
  const stack = resolveFeeling(feeling).font
  const st = resolveFeeling(feeling).style
  const fw = st.fontWeight ?? 600
  const fitalic = st.fontStyle === 'italic' ? 'italic ' : ''
  const headline = st.textTransform === 'uppercase' ? text.toUpperCase() : text
  await ensureFonts(stack, emoji)

  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')

  const r = Math.round((20 / 600) * S)
  ctx.beginPath()
  if (ctx.roundRect) ctx.roundRect(0, 0, S, S, r)
  else ctx.rect(0, 0, S, S)
  ctx.clip()

  const grad = ctx.createLinearGradient(0, 0, S, S)
  grad.addColorStop(0, pal.bg1)
  grad.addColorStop(1, pal.bg2)
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, S, S)

  ctx.fillStyle = pal.text_color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  ctx.font = `120px ${EMOJI_STACK}`
  ctx.fillText(emoji, S / 2, S * 0.4)

  const lineHeight = 1.33
  const maxWidth = S - 96
  const widthAt = (str, px) => {
    ctx.font = `${fitalic}${fw} ${px}px ${stack}`
    return ctx.measureText(str).width
  }
  if ('letterSpacing' in ctx) ctx.letterSpacing = st.letterSpacing ?? '0px'
  const fpx = fitCanvasFont({
    text: headline,
    maxWidth,
    maxHeight: S * 0.26,
    min: Math.round((32 * S) / 600),
    max: Math.round((104 * S) / 600),
    lineHeight,
    widthAt,
  })

  ctx.font = `${fitalic}${fw} ${fpx}px ${stack}`
  const lines = wrapLines((str) => ctx.measureText(str).width, headline, maxWidth, 4)
  let ty = S * 0.62 - ((lines.length - 1) * fpx * lineHeight) / 2
  ctx.globalAlpha = st.opacity ?? 1
  for (const line of lines) {
    ctx.fillText(line, S / 2, ty)
    ty += fpx * lineHeight
  }
  ctx.globalAlpha = 1

  if ('letterSpacing' in ctx) ctx.letterSpacing = '0px'
  ctx.font = `600 13px ${stack}`
  if ('letterSpacing' in ctx) ctx.letterSpacing = '3.5px'
  ctx.globalAlpha = 0.85
  ctx.fillText(feeling.toUpperCase(), S / 2, S * 0.84)
  ctx.globalAlpha = 1

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
