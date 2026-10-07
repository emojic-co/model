import { useEffect, useRef } from 'react'
import { createPainter, emojiPixels } from './useCardImage'
import { emojiLayer } from '../cardGif'
import { prefersReducedMotion } from '../notoLottie'

export const playerKey = ({ emoji, feeling, lang, text, dim, colors }) =>
  [emoji, feeling, lang, text, dim ? 1 : 0, JSON.stringify(colors)].join('|')

// What a card "is" apart from its text and colors: a change here blanks the canvas at once, while a text or
// color change keeps the old frame on screen until the rebuilt one draws.
export const shapeKey = ({ emoji, feeling, lang }) => [emoji, feeling, lang].join('|')

const REBUILD_DEBOUNCE_MS = 150

// Preview player: draws the shared clip (same painter + timeline as GIF/MP4 export) in a loop that
// restarts from t=0 whenever the card changes. While `paused` (an export is running) the loop stops and the last
// frame stays on screen, so the preview doesn't compete with the encoder for the main thread. `scale` is the painter's backing-resolution factor.
export function useCardPlayer(cardData, canvasRef, scale = 1, paused = false) {
  const key = cardData ? playerKey(cardData) : ''
  const shape = cardData ? shapeKey(cardData) : ''
  const lastShape = useRef('')

  useEffect(() => {
    if (shape !== lastShape.current) {
      lastShape.current = shape
      const canvas = canvasRef.current
      canvas?.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    }
    if (!cardData || paused) return
    let cancelled = false
    let raf = 0
    let emoji
    const timer = setTimeout(async () => {
      try {
        const paint = await createPainter(cardData, scale)
        const layer = await emojiLayer(cardData.emoji, emojiPixels(scale))
        if (cancelled) return layer.destroy()
        emoji = layer
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const blit = (src) => {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          ctx.drawImage(src, 0, 0, canvas.width, canvas.height)
        }
        if (prefersReducedMotion()) return blit(paint.poster(emoji))
        const tl = paint.timeline(emoji.loopMs)
        const start = performance.now()
        const tick = (now) => {
          if (cancelled) return
          blit(paint.frame(tl, emoji, (now - start) % tl.durationMs))
          raf = requestAnimationFrame(tick)
        }
        raf = requestAnimationFrame(tick)
      } catch (err) {
        console.error(err)
      }
    }, REBUILD_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
      cancelAnimationFrame(raf)
      emoji?.destroy()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, scale, paused])
}
