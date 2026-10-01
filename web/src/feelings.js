import { scriptForLang, fontForScript, latinFont, LATIN } from './scriptFonts'
import { hexToOklab } from './model'

export const DEFAULT_COLORS = {
  bg1: hexToOklab('#a8e2f4'),
  bg2: hexToOklab('#78c9f4'),
  text_color: hexToOklab('#282e36'),
}

export const CLUSTERS = {
  anger: { entrance: 'slam', emoji: 'shake', driftSec: 10 },
  joy: { entrance: 'pop', emoji: 'hop', driftSec: 12 },
  play: { entrance: 'spin', emoji: 'wobble', driftSec: 11 },
  calm: { entrance: 'settle', emoji: 'breathe', driftSec: 22 },
  sad: { entrance: 'drop', emoji: 'sink', driftSec: 20 },
  anxiety: { entrance: 'jitter', emoji: 'tremor', driftSec: 9 },
  tender: { entrance: 'bloom', emoji: 'heartbeat', driftSec: 16 },
  drive: { entrance: 'rise', emoji: 'lift', driftSec: 14 },
  reflective: { entrance: 'fadeTilt', emoji: 'tilt', driftSec: 18 },
}

export const ENTRANCE_MOTIFS = ['slam', 'pop', 'spin', 'settle', 'drop', 'jitter', 'bloom', 'rise', 'fadeTilt', 'droop', 'shrinkBack']
export const EMOJI_MOTIFS = ['shake', 'hop', 'wobble', 'breathe', 'sink', 'tremor', 'heartbeat', 'lift', 'tilt', 'droop', 'shrinkBack']
export const MOTIF_DEFAULT_MS = { entrance: 650, emoji: 2400 }

export const FEELINGS = {
  Joyful: { cluster: 'joy', style: { fontWeight: 600 }, dur: { entrance: 560, emoji: 900 } },
  Excited: { cluster: 'joy', style: { textTransform: 'uppercase', letterSpacing: '0.05em' }, dur: { entrance: 460, emoji: 380 } },
  Hopeful: { cluster: 'drive', style: { fontWeight: 500 }, dur: { entrance: 780, emoji: 3000 } },
  Serene: { cluster: 'calm', style: { fontWeight: 500 }, dur: { entrance: 900, emoji: 4200 } },
  Tender: { cluster: 'tender', style: { fontWeight: 700 }, dur: { entrance: 700, emoji: 1300 } },
  Playful: { cluster: 'play', style: {}, dur: { entrance: 600, emoji: 1100 } },
  Whimsical: { cluster: 'play', style: { letterSpacing: '0.02em' }, dur: { entrance: 640, emoji: 1500 } },
  Awed: { cluster: 'reflective', style: { letterSpacing: '0.04em' }, dur: { entrance: 520, emoji: 2600 } },
  Earnest: { cluster: 'tender', style: { letterSpacing: '0.01em' }, dur: { entrance: 720, emoji: 1600 } },
  Determined: { cluster: 'drive', style: { textTransform: 'uppercase', fontWeight: 700 }, dur: { entrance: 560, emoji: 1400 } },
  Proud: { cluster: 'drive', style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 700, emoji: 2600 } },
  Wistful: { cluster: 'sad', style: { fontStyle: 'italic', letterSpacing: '0.05em', opacity: 0.9 }, dur: { entrance: 1050, emoji: 4200 } },
  Melancholy: { cluster: 'sad', style: { fontStyle: 'italic' }, dur: { entrance: 1000, emoji: 3200 } },
  Anxious: { cluster: 'anxiety', style: {}, dur: { entrance: 560, emoji: 220 } },
  Tense: { cluster: 'anxiety', style: { letterSpacing: '-0.01em' }, dur: { entrance: 500, emoji: 420 } },
  Furious: { cluster: 'anger', style: { textTransform: 'uppercase', letterSpacing: '0.06em' }, dur: { entrance: 420, emoji: 450 } },
  Irritated: { cluster: 'anger', style: { textTransform: 'uppercase' }, dur: { entrance: 520, emoji: 600 } },
  Disgusted: { cluster: 'anger', style: { fontStyle: 'italic', letterSpacing: '0.03em' }, dur: { entrance: 480, emoji: 700 } },
  Startled: { cluster: 'play', entrance: 'shrinkBack', emoji: 'shrinkBack', style: {}, dur: { entrance: 420, emoji: 2600 } },
  Sarcastic: { cluster: 'reflective', style: { fontStyle: 'italic' }, dur: { entrance: 800, emoji: 4200 } },
  Deadpan: { cluster: 'reflective', entrance: 'droop', emoji: 'droop', style: {}, dur: { entrance: 700, emoji: 6000 } },
  Neutral: { cluster: 'reflective', style: { fontWeight: 600 }, dur: { entrance: 650, emoji: 3200 } },
}

export function resolveFeeling(feeling, lang = 'en') {
  const f = FEELINGS[feeling] ?? FEELINGS.Neutral
  const c = CLUSTERS[f.cluster]
  const script = scriptForLang(lang)
  return {
    cluster: f.cluster,
    font: script === LATIN ? latinFont(feeling in FEELINGS ? feeling : 'Neutral') : fontForScript(script, f.cluster),
    entrance: f.entrance ?? c.entrance,
    emoji: f.emoji ?? c.emoji,
    style: f.style ?? {},
    vars: {
      '--entrance-dur': `${f.dur?.entrance ?? MOTIF_DEFAULT_MS.entrance}ms`,
      '--emoji-dur': `${f.dur?.emoji ?? MOTIF_DEFAULT_MS.emoji}ms`,
      '--drift-sec': `${c.driftSec}s`,
    },
  }
}

export function topFeelings(feelingScores, feelings, selected, count = 5) {
  if (!feelingScores) return []
  const ranked = feelings
    .map((f, i) => ({ f, p: feelingScores[i] }))
    .sort((a, b) => b.p - a.p)
    .map((x) => x.f)
  const top = ranked.slice(0, count)
  if (selected && !top.includes(selected)) return [...ranked.slice(0, count - 1), selected]
  return top
}
