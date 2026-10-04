import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { continueRender, delayRender, staticFile } from 'remotion'
import { createPainter } from '../../../web/src/hooks/useCardImage.js'
import { springScale } from '../../../web/src/clip.js'
import { loadFonts } from './fonts'

import type { CardConfig } from './config'

const SCALE = 2 // painter backing resolution: 512 * 2 = 1024px

// Mirrors web/src/cardGif.js emojiLayer, but loads the clip from Remotion's public dir.
async function emojiLayer(emoji: string) {
  const index = await fetch(staticFile('noto/index.json')).then((r) => r.json())
  const stem = index[emoji]
  if (stem) {
    const [{ default: lottie }, data] = await Promise.all([
      import('lottie-web/build/player/lottie_canvas'),
      fetch(staticFile(`noto/${stem}.json`)).then((r) => r.json()),
    ])
    const canvas = document.createElement('canvas')
    canvas.width = data.w
    canvas.height = data.h
    const anim = lottie.loadAnimation({
      renderer: 'canvas',
      loop: false,
      autoplay: false,
      animationData: data,
      rendererSettings: { context: canvas.getContext('2d'), clearCanvas: true },
    })
    const frames = data.op - data.ip
    const loopMs = (frames / (data.fr || 30)) * 1000
    return {
      loopMs,
      draw(timeMs: number, ctx: CanvasRenderingContext2D, x: number, y: number, px: number) {
        anim.goToAndStop(data.ip + ((timeMs % loopMs) / loopMs) * (frames - 1), true)
        ctx.drawImage(canvas, x - px / 2, y - px / 2, px, px)
      },
    }
  }
  return {
    loopMs: 0,
    draw(timeMs: number, ctx: CanvasRenderingContext2D, x: number, y: number, px: number) {
      ctx.font = `${px * springScale(timeMs)}px "Noto Color Emoji", sans-serif`
      ctx.fillText(emoji, x, y)
    },
  }
}

type Player = { paint: any; tl: any; emoji: any }

// One real card (same painter + clip timeline as the web preview and GIF/MP4 export),
// drawn deterministically for `timeMs` since the card started playing.
export const CardClip: React.FC<{ card: CardConfig; lang: string; timeMs: number; size: number }> = ({
  card,
  lang,
  timeMs,
  size,
}) => {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [player, setPlayer] = useState<Player | null>(null)
  const [handle] = useState(() => delayRender(`card-${card.text}`))

  useEffect(() => {
    ;(async () => {
      await loadFonts()
      const paint = await createPainter(
        {
          text: card.text,
          emoji: card.emoji,
          feeling: card.feeling,
          lang: card.lang ?? lang,
          dim: false,
          colors: card.colors,
        },
        SCALE,
      )
      const emoji = await emojiLayer(card.emoji)
      setPlayer({ paint, tl: paint.timeline(emoji.loopMs), emoji })
    })()
      .catch((e) => {
        console.error(e)
        continueRender(handle)
      })
  }, [card, lang, handle])

  useLayoutEffect(() => {
    const el = canvas.current
    if (!el || !player) return
    const out = player.paint.frame(player.tl, player.emoji, Math.max(0, timeMs) % player.tl.durationMs)
    const ctx = el.getContext('2d')!
    ctx.clearRect(0, 0, el.width, el.height)
    ctx.drawImage(out, 0, 0, el.width, el.height)
    continueRender(handle) // only after the first real paint, so the screenshot never catches a blank canvas
  })

  return (
    <canvas
      ref={canvas}
      width={512 * SCALE}
      height={512 * SCALE}
      style={{ width: size, height: size, borderRadius: size * 0.06, display: 'block' }}
    />
  )
}

