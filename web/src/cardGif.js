import { loadNotoIndex, notoUrl } from './notoLottie'
import { CLIP, springScale } from './clip'

// Emoji layer: the Noto Lottie clone rendered to an offscreen canvas (looping over the timeline),
// or, for emojis without a clone, the spring scale at the start. `restMs` is the time the poster frame shows.
export async function emojiLayer(emoji) {
  const stem = (await loadNotoIndex())[emoji]
  if (stem) {
    const [{ default: lottie }, data] = await Promise.all([
      import('lottie-web/build/player/lottie_canvas'),
      fetch(notoUrl(stem)).then((r) => r.json()),
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
      restMs: 0,
      draw(timeMs, ctx, x, y, px) {
        anim.goToAndStop(data.ip + ((timeMs % loopMs) / loopMs) * (frames - 1), true)
        ctx.drawImage(canvas, x - px / 2, y - px / 2, px, px)
      },
      destroy: () => anim.destroy(),
    }
  }
  return {
    loopMs: 0,
    restMs: CLIP.spring.durationMs,
    draw(timeMs, ctx, x, y, px) {
      ctx.font = `${px * springScale(timeMs)}px "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif`
      ctx.fillText(emoji, x, y)
    },
    destroy() {},
  }
}

export const frameCount = (durationMs, fps) => Math.max(1, Math.ceil((durationMs * fps) / 1000))

export const tick = () => new Promise((r) => setTimeout(r))

export function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
