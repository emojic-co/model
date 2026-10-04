// Player for the background shimmer effects defined in shimmers.yml (the ground truth shared
// with Android; see the semantics header there).
import { parse } from 'yaml'
import raw from './shimmers.yml?raw'
import { prefersReducedMotion } from './textAnimation'

export const SHIMMERS = parse(raw)

const CSS_BLEND = { normal: 'normal', screen: 'screen', overlay: 'overlay', softLight: 'soft-light', multiply: 'multiply' }
const PROP = '--shimmer-c'

export function resolveShimmer(anim, cluster, feeling) {
  const base = anim.clusters[cluster] ?? anim.clusters.reflective
  const o = anim.styles?.[feeling] ?? {}
  return { ...base, ...o }
}

const pct = (v) => `${+(v * 100).toFixed(3)}%`

// CSS `background` for the overlay. The animated number lives in the registered --shimmer-c property.
export function shimmerBackground(e) {
  const list = (position) => e.stops.map((s) => `${s.color} ${position(s.at)}`).join(', ')
  if (e.kind === 'sweep') {
    return `linear-gradient(${e.angleDeg + 90}deg, ${list((at) => `calc(var(${PROP}) * 100% + ${pct((at - 0.5) * e.widthFrac)})`)})`
  }
  if (e.kind === 'radial') {
    return `radial-gradient(circle calc(var(${PROP}) * ${e.radiusFrac * 100}cqw) at ${pct(e.cx)} ${pct(e.cy)}, ${list((at) => pct(at))})`
  }
  return `linear-gradient(${e.angleDeg + 90}deg, ${list((at) => pct(at))})`
}

// Web Animations keyframes for the single pass; the layer is hidden again once it ends.
export function shimmerKeyframes(e) {
  const [a, b, c, d] = e.easing
  const easing = `cubic-bezier(${a}, ${b}, ${c}, ${d})`
  let cur = { c: 0, opacity: 1, ...(e.keyframes[0] ?? {}) }
  const frames = e.keyframes.map((k, i) => {
    cur = { ...cur, ...k }
    const last = i === e.keyframes.length - 1
    return { offset: k.at, opacity: cur.opacity, [PROP]: cur.c, easing: last ? 'linear' : easing }
  })
  const last = frames[frames.length - 1]
  frames.push({ ...last, offset: 1, opacity: 0 })
  return { frames, durationMs: e.durationMs }
}

// Starts the shimmer on `el` after `startAfterMs` (text entrance total + spec startDelayMs).
export function playShimmer(el, { anim = SHIMMERS, cluster, feeling, entranceMs }) {
  if (!el || prefersReducedMotion()) return () => {}
  const e = resolveShimmer(anim, cluster, feeling)
  const { frames, durationMs } = shimmerKeyframes(e)
  el.style.background = shimmerBackground(e)
  el.style.mixBlendMode = CSS_BLEND[e.blend] ?? 'normal'
  const an = el.animate(frames, {
    duration: durationMs,
    delay: entranceMs + anim.startDelayMs,
    fill: 'both',
  })
  return () => an.cancel()
}
