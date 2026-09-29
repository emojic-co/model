// Player for the per-character text animations defined in textAnimations.yml
// (the ground truth shared with Android; see the semantics header there).
import { parse } from 'yaml'
import raw from './textAnimations.yml?raw'

export const TEXT_ANIMATIONS = parse(raw)

const CONNECTED_RE = /[\p{Script=Arabic}\p{Script=Devanagari}\p{Script=Thai}]/u
const LONG_WORD = 14

export function hash(i) {
  return ((Math.imul(i + 1, 2654435761) >>> 0) % 1000) / 1000
}

// Words -> animation units. Connected scripts stay whole so letters keep joining.
export function splitWords(text) {
  const words = text.split(/\s+/).filter(Boolean)
  if (typeof Intl === 'undefined' || !Intl.Segmenter) return words.map((w) => [...w])
  const seg = new Intl.Segmenter(undefined, { granularity: 'grapheme' })
  return words.map((w) => (CONNECTED_RE.test(w) ? [w] : [...seg.segment(w)].map((s) => s.segment)))
}

export function countUnits(words) {
  return words.reduce((n, w) => n + w.length, 0)
}

export function isLongWord(units) {
  return units.length > LONG_WORD
}

export function resolveMotif(anim, motif, feeling) {
  const base = anim.motifs[motif] ?? anim.motifs.settle
  return { ...base, ...(anim.styles?.[feeling] ?? {}) }
}

export function scheduleFor(anim, motif, feeling, n) {
  const m = resolveMotif(anim, motif, feeling)
  const rank = (i) => {
    switch (m.order) {
      case 'reverse': return n - 1 - i
      case 'center': return Math.abs(i - (n - 1) / 2)
      case 'random': return hash(i) * (n - 1)
      default: return i
    }
  }
  const ranks = Array.from({ length: n }, (_, i) => rank(i))
  const maxRank = Math.max(0, ...ranks)
  const cap = (anim.timing.maxTotalMs - m.durationMs - m.jitterMs) / (maxRank || 1)
  const s = maxRank === 0 ? m.staggerMs : Math.max(0, Math.min(m.staggerMs, cap))
  return {
    motif: m,
    delays: ranks.map((r, i) => r * s + m.jitterMs * hash(i)),
  }
}

// Time until every unit has finished its entrance (0 when there is nothing to animate).
export function entranceTotalMs(anim, motif, feeling, n) {
  if (n === 0 || prefersReducedMotion()) return 0
  const { motif: m, delays } = scheduleFor(anim, motif, feeling, n)
  return Math.max(...delays) + m.durationMs
}

const DEFAULTS = { opacity: 1, x: 0, y: 0, scale: 1, scaleY: 1, rotate: 0 }

// Web Animations keyframes for unit i (carry-forward of missing fields, mirrored when alternate).
export function keyframesFor(motif, i) {
  const flip = motif.alternate && i % 2 === 1 ? -1 : 1
  let cur = { ...DEFAULTS }
  return motif.keyframes.map((k) => {
    cur = { ...cur, ...k }
    const x = cur.x * flip
    const rot = cur.rotate * flip
    return {
      offset: k.at,
      opacity: cur.opacity,
      transform: `translate(${x}em, ${cur.y}em) rotate(${rot}deg) scale(${cur.scale}, ${cur.scale * cur.scaleY})`,
    }
  })
}

export function prefersReducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}

// Plays the animation on every [data-ci] unit under `root`; returns a cancel function.
export function playText(root, { anim = TEXT_ANIMATIONS, motif, feeling }) {
  if (!root || prefersReducedMotion()) return () => {}
  const els = [...root.querySelectorAll('[data-ci]')].sort((a, b) => a.dataset.ci - b.dataset.ci)
  const { motif: m, delays } = scheduleFor(anim, motif, feeling, els.length)
  const [a, b, c, d] = m.easing
  const running = els.map((el, i) =>
    el.animate(keyframesFor(m, i), {
      duration: m.durationMs,
      delay: delays[i],
      easing: `cubic-bezier(${a}, ${b}, ${c}, ${d})`,
      fill: 'both',
    }),
  )
  return () => running.forEach((an) => an.cancel())
}
