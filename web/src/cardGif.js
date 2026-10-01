import { GIFEncoder, quantize, applyPalette } from 'gifenc'
import { createPainter } from './hooks/useCardImage'
import { loadNotoIndex, notoUrl } from './notoLottie'
import { CLIP, springScale } from './clip'

// Output sizes tried in order until the file fits TARGET_BYTES (the last one is kept regardless).
const SIZES = [320, 256, 208, 160]
const TARGET_BYTES = 500 * 1024

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

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const DITHER = 5

// 4x4 ordered dither: nudges each pixel's RGB by a position-dependent offset before palette lookup.
function dither(data, width) {
  for (let p = 0, n = data.length / 4; p < n; p++) {
    const d = (BAYER[((((p / width) | 0) & 3) << 2) | (p % width & 3)] - 7.5) * (DITHER / 7.5)
    data[p * 4] += d
    data[p * 4 + 1] += d
    data[p * 4 + 2] += d
  }
}

// One palette for the whole clip, from a subsample of every few frames.
function sharedPalette(frames) {
  const picked = frames.filter((_, i) => i % 4 === 0)
  const stride = 5
  const sample = new Uint8Array(picked.reduce((n, f) => n + Math.ceil(f.length / 4 / stride) * 4, 0))
  let o = 0
  for (const f of picked) for (let p = 0; p < f.length; p += 4 * stride) sample.set(f.subarray(p, p + 4), (o += 4) - 4)
  return quantize(sample, 256)
}

export const tick = () => new Promise((r) => setTimeout(r))

// Encodes the card as a looping animated GIF. onProgress gets 0..1; aborting `signal` rejects with AbortError.
export async function renderGif(cardData, { onProgress, signal } = {}) {
  const emoji = await emojiLayer(cardData.emoji)
  try {
    let blob
    for (let a = 0; a < SIZES.length; a++) {
      const report = (p) => onProgress?.((a + p) / SIZES.length)
      blob = await encode(cardData, emoji, SIZES[a], report, signal)
      if (blob.size <= TARGET_BYTES) break
    }
    onProgress?.(1)
    return blob
  } finally {
    emoji.destroy()
  }
}

const grabPixels = (c) => c.getContext('2d').getImageData(0, 0, c.width, c.height).data

async function encode(cardData, emoji, size, onProgress, signal) {
  const paint = await createPainter(cardData, size / 512)
  const tl = paint.timeline(emoji.loopMs)
  const count = frameCount(tl.durationMs, CLIP.gifFps)
  // Pass 1: render every frame (poster first). Pass 2: encode with ONE shared palette (per-frame palettes make
  // the soft shimmer gradient band and flicker) plus ordered dithering to hide the banding.
  const frames = [grabPixels(paint.poster(emoji))]
  for (let i = 0; i < count; i++) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
    frames.push(grabPixels(paint.frame(tl, emoji, (i * 1000) / CLIP.gifFps)))
    onProgress?.(((i + 1) / count) * 0.8)
    await tick()
  }
  const palette = sharedPalette(frames)
  const { width, height } = paint.size
  const gif = GIFEncoder()
  for (let i = 0; i < frames.length; i++) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
    dither(frames[i], width)
    const delay = i === 0 ? CLIP.posterHoldMs : Math.round(1000 / CLIP.gifFps)
    gif.writeFrame(applyPalette(frames[i], palette), width, height, { palette, delay })
    frames[i] = null
    onProgress?.(0.8 + ((i + 1) / frames.length) * 0.2)
    await tick()
  }
  gif.finish()
  return new Blob([gif.bytes()], { type: 'image/gif' })
}

export function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
