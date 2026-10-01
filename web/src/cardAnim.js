// Canvas ports of the per-unit text entrance (textAnimations.yml) and background shimmer
// (shimmers.yml), evaluated at an explicit time so GIF/MP4 export can step frames deterministically.
// Semantics follow the headers of those yml files; textAnimation.js / shimmer.js are the live DOM players.
import { TEXT_ANIMATIONS, scheduleFor } from './textAnimation'
import { SHIMMERS, resolveShimmer } from './shimmer'

const CONNECTED_RE = /[\p{Script=Arabic}\p{Script=Devanagari}\p{Script=Thai}]/u
const segmenter = typeof Intl !== 'undefined' && Intl.Segmenter ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }) : null

// Animation units of one line as [start, end) string offsets (same unit rules as splitWords).
export function lineUnits(line) {
  const units = []
  for (const m of line.matchAll(/\S+/g)) {
    const word = m[0]
    if (!segmenter || CONNECTED_RE.test(word)) {
      units.push([m.index, m.index + word.length])
      continue
    }
    for (const s of segmenter.segment(word)) units.push([m.index + s.index, m.index + s.index + s.segment.length])
  }
  return units
}

function cubicBezier(x1, y1, x2, y2) {
  const sample = (a, b, t) => 3 * a * (1 - t) * (1 - t) * t + 3 * b * (1 - t) * t * t + t * t * t
  return (x) => {
    if (x <= 0) return 0
    if (x >= 1) return 1
    let lo = 0
    let hi = 1
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2
      if (sample(x1, x2, mid) < x) lo = mid
      else hi = mid
    }
    return sample(y1, y2, (lo + hi) / 2)
  }
}

const lerp = (a, b, t) => a + (b - a) * t

// Fills missing keyframe fields from the previous keyframe.
function carry(keyframes, defaults) {
  let cur = { ...defaults }
  return keyframes.map((k) => (cur = { ...cur, ...k }))
}

function sample(frames, p, fields) {
  if (p <= frames[0].at) return frames[0]
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1]
    const b = frames[i]
    if (p <= b.at) {
      const t = b.at === a.at ? 1 : (p - a.at) / (b.at - a.at)
      return Object.fromEntries(fields.map((f) => [f, lerp(a[f], b[f], t)]))
    }
  }
  return frames[frames.length - 1]
}

const TEXT_DEFAULTS = { opacity: 1, x: 0, y: 0, scale: 1, scaleY: 1, rotate: 0 }
const TEXT_FIELDS = Object.keys(TEXT_DEFAULTS)

// Builds timeMs -> per-unit pose for n units. Pose: {opacity, x, y (em), rotate (deg), scale, scaleY}.
export function textTimeline(motifName, feeling, n, anim = TEXT_ANIMATIONS) {
  const { motif, delays } = scheduleFor(anim, motifName, feeling, n)
  const frames = carry(motif.keyframes, TEXT_DEFAULTS)
  const ease = cubicBezier(...motif.easing)
  const totalMs = n ? Math.max(...delays) + motif.durationMs : 0
  const pose = (i, timeMs) => {
    const p = ease(Math.min(1, Math.max(0, (timeMs - delays[i]) / motif.durationMs)))
    const k = sample(frames, p, TEXT_FIELDS)
    const flip = motif.alternate && i % 2 === 1 ? -1 : 1
    return { ...k, x: k.x * flip, rotate: k.rotate * flip }
  }
  return { totalMs, pose }
}

const BLEND = { normal: 'source-over', screen: 'screen', overlay: 'overlay', softLight: 'soft-light', multiply: 'multiply' }

function rgba(hex) {
  const v = parseInt(hex.slice(1), 16)
  const a = hex.length > 7 ? (v & 255) / 255 : 1
  const rgb = hex.length > 7 ? v >>> 8 : v
  return `rgba(${(rgb >> 16) & 255}, ${(rgb >> 8) & 255}, ${rgb & 255}, ${+a.toFixed(4)})`
}

// Linear gradient along angleDeg using the yml's s(p) mapping; stops given as [s, color].
function axisGradient(ctx, S, angleDeg, stops) {
  const a = (angleDeg * Math.PI) / 180
  const dx = Math.cos(a)
  const dy = Math.sin(a)
  const half = ((Math.abs(dx) + Math.abs(dy)) * S) / 2
  const g = ctx.createLinearGradient(S / 2 - dx * half, S / 2 - dy * half, S / 2 + dx * half, S / 2 + dy * half)
  const clear = (c) => c.replace(/[\d.]+\)$/, '0)')
  const sorted = stops.map(([s, c]) => [Math.min(1, Math.max(0, s)), c]).sort((x, y) => x[0] - y[0])
  if (sorted[0][0] > 0) g.addColorStop(0, clear(sorted[0][1]))
  for (const [s, c] of sorted) g.addColorStop(s, c)
  if (sorted[sorted.length - 1][0] < 1) g.addColorStop(1, clear(sorted[sorted.length - 1][1]))
  return g
}

const SHIMMER_DEFAULTS = { c: 0, opacity: 1 }
const SHIMMER_FIELDS = ['c', 'opacity']

// Builds timeMs -> draw(ctx, S) for the cluster's shimmer. startMs = text entrance end + startDelayMs.
export function shimmerTimeline(cluster, feeling, anim = SHIMMERS) {
  const e = resolveShimmer(anim, cluster, feeling)
  const frames = carry(e.keyframes, { ...SHIMMER_DEFAULTS, ...(e.keyframes[0] ?? {}) })
  const ease = cubicBezier(...e.easing)
  const cycleMs = e.durationMs + e.pauseMs
  const colors = e.stops.map((s) => [s.at, rgba(s.color)])
  const draw = (ctx, S, t) => {
    const u = (t % cycleMs) / e.durationMs
    if (t < 0 || u >= 1) return
    // Eased progress per segment, then lerp c/opacity.
    let k = frames[frames.length - 1]
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1]
      const b = frames[i]
      if (u <= b.at) {
        const f = ease(b.at === a.at ? 1 : (u - a.at) / (b.at - a.at))
        k = Object.fromEntries(SHIMMER_FIELDS.map((n) => [n, lerp(a[n], b[n], f)]))
        break
      }
    }
    let fill
    if (e.kind === 'radial') {
      const r = k.c * e.radiusFrac * S
      if (r <= 0) return
      fill = ctx.createRadialGradient(e.cx * S, e.cy * S, 0, e.cx * S, e.cy * S, r)
      for (const [at, c] of colors) fill.addColorStop(Math.min(1, Math.max(0, at)), c)
    } else if (e.kind === 'flash') {
      fill = axisGradient(ctx, S, e.angleDeg, colors)
    } else {
      fill = axisGradient(ctx, S, e.angleDeg, colors.map(([at, c]) => [k.c + (at - 0.5) * e.widthFrac, c]))
    }
    ctx.save()
    ctx.globalAlpha = k.opacity
    ctx.globalCompositeOperation = BLEND[e.blend] ?? 'source-over'
    ctx.fillStyle = fill
    ctx.fillRect(0, 0, S, S)
    ctx.restore()
  }
  return { cycleMs, startDelayMs: anim.startDelayMs, draw }
}
