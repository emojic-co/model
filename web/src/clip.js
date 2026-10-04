// Shared clip timeline for preview and export (rules: clip.yml, the ground truth shared with Android).
import { parse } from 'yaml'
import raw from './clip.yml?raw'
import { shimmerTimeline, textTimeline } from './cardAnim'

export const CLIP = parse(raw)

export function clipFor({ entranceMs, startDelayMs, passMs, loopMs }, spec = CLIP) {
  const content = entranceMs + startDelayMs + passMs
  if (!loopMs) return { durationMs: content, loops: 0 }
  return { durationMs: Math.max(content, Math.min(loopMs, spec.maxClipMs)), loops: 1 }
}

export function springScale(tMs, s = CLIP.spring) {
  if (tMs >= s.durationMs) return 1
  const t = Math.max(0, tMs) / 1000
  const wd = s.omega * Math.sqrt(1 - s.zeta * s.zeta)
  const decay = Math.exp(-s.zeta * s.omega * t)
  const osc = Math.cos(wd * t) + ((s.zeta * s.omega) / wd) * Math.sin(wd * t)
  return 1 - (1 - s.from) * decay * osc
}

// Pose sources for one card on the clip's time axis (t = ms since the clip started, looping at durationMs).
export function createTimeline({ motif, feeling, cluster, unitCount, loopMs }) {
  const text = textTimeline(motif, feeling, unitCount)
  const shimmer = shimmerTimeline(cluster, feeling)
  const shimmerStartMs = text.totalMs + shimmer.startDelayMs
  const { durationMs, loops } = clipFor({
    entranceMs: text.totalMs,
    startDelayMs: shimmer.startDelayMs,
    passMs: shimmer.passMs,
    loopMs,
  })
  return {
    durationMs,
    loops,
    loopMs,
    text,
    shimmer: {
      ...shimmer,
      pose: (t) => shimmer.pose(t - shimmerStartMs),
      draw: (ctx, S, t) => shimmer.draw(ctx, S, t - shimmerStartMs),
    },
    shimmerStartMs,
    emojiMs: (t) => t,
  }
}
