import { ArrayBufferTarget, Muxer } from 'mp4-muxer'
import { emojiLayer, frameCount, tick } from './cardGif'
import { CLIP } from './clip'
import { createPainter } from './hooks/useCardImage'

const SIZE = 1024
const FPS = CLIP.mp4Fps
const BITRATE = 6_000_000
const CODEC = 'avc1.640028' // High profile, level 4.0 (1024x1024 exceeds level 3.x)

export const mp4Supported = () => typeof VideoEncoder === 'function' && typeof VideoFrame === 'function'

// Encodes the card (same timeline as the GIF) as an H.264 MP4 via WebCodecs. onProgress gets 0..1;
// aborting `signal` rejects with AbortError.
export async function renderMp4(cardData, { onProgress, signal } = {}) {
  if (!mp4Supported()) throw new Error('WebCodecs not supported')
  const config = { codec: CODEC, width: SIZE, height: SIZE, bitrate: BITRATE, framerate: FPS }
  if (!(await VideoEncoder.isConfigSupported(config)).supported) throw new Error('H.264 encoding not supported')

  const paint = await createPainter(cardData, SIZE / 512)
  const emoji = await emojiLayer(cardData.emoji)
  let encoder
  try {
    const muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: { codec: 'avc', width: SIZE, height: SIZE },
      fastStart: 'in-memory',
    })
    let failure
    encoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => (failure = e),
    })
    encoder.configure(config)

    const tl = paint.timeline(emoji.loopMs)
    const count = frameCount(tl.durationMs, FPS)
    const poster = new VideoFrame(paint.poster(emoji), { timestamp: 0, duration: CLIP.posterHoldMs * 1000 })
    encoder.encode(poster, { keyFrame: true })
    poster.close()
    for (let i = 0; i < count; i++) {
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
      if (failure) throw failure
      const timeMs = (i * 1000) / FPS
      const canvas = paint.frame(tl, emoji, timeMs)
      const frame = new VideoFrame(canvas, { timestamp: Math.round((CLIP.posterHoldMs + timeMs) * 1000), duration: Math.round(1e6 / FPS) })
      encoder.encode(frame, { keyFrame: i % FPS === 0 })
      frame.close()
      onProgress?.((i + 1) / count)
      // Let the encoder drain instead of queueing hundreds of raw frames.
      while (encoder.encodeQueueSize > 8) await tick()
      await tick()
    }
    await encoder.flush()
    if (failure) throw failure
    muxer.finalize()
    return new Blob([muxer.target.buffer], { type: 'video/mp4' })
  } finally {
    if (encoder && encoder.state !== 'closed') encoder.close()
    emoji.destroy()
  }
}
