export const CLUSTERS = {
  anger: { font: '"Anton", system-ui, sans-serif', entrance: 'slam', emoji: 'shake', driftSec: 10 },
  joy: { font: '"Fredoka", system-ui, sans-serif', entrance: 'pop', emoji: 'hop', driftSec: 12 },
  play: { font: '"Baloo 2", system-ui, sans-serif', entrance: 'spin', emoji: 'wobble', driftSec: 11 },
  calm: { font: '"Quicksand", system-ui, sans-serif', entrance: 'settle', emoji: 'breathe', driftSec: 22 },
  sad: { font: '"Playfair Display", Georgia, serif', entrance: 'drop', emoji: 'sink', driftSec: 20 },
  anxiety: { font: '"Shantell Sans", system-ui, cursive', entrance: 'jitter', emoji: 'tremor', driftSec: 9 },
  tender: { font: '"Caveat", "Segoe Script", cursive', entrance: 'bloom', emoji: 'heartbeat', driftSec: 16 },
  drive: { font: '"Poppins", system-ui, sans-serif', entrance: 'rise', emoji: 'lift', driftSec: 14 },
  reflective: { font: '"Inter", system-ui, sans-serif', entrance: 'fadeTilt', emoji: 'tilt', driftSec: 18 },
}

export const ENTRANCE_MOTIFS = ['slam', 'pop', 'spin', 'settle', 'drop', 'jitter', 'bloom', 'rise', 'fadeTilt', 'droop', 'shrinkBack']
export const EMOJI_MOTIFS = ['shake', 'hop', 'wobble', 'breathe', 'sink', 'tremor', 'heartbeat', 'lift', 'tilt', 'droop', 'shrinkBack']
export const MOTIF_DEFAULT_MS = { entrance: 650, emoji: 2400 }

export const FEELINGS = {
  Angry: { cluster: 'anger', style: { textTransform: 'uppercase', letterSpacing: '0.06em' }, dur: { entrance: 420, emoji: 450 } },
  Annoyed: { cluster: 'anger', style: { letterSpacing: '-0.01em' }, dur: { entrance: 500, emoji: 950 } },
  Frustrated: { cluster: 'anger', style: { textTransform: 'uppercase' }, dur: { entrance: 520, emoji: 600 } },
  Happy: { cluster: 'joy', style: { fontWeight: 600 }, dur: { entrance: 560, emoji: 900 } },
  Excited: { cluster: 'joy', style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 460, emoji: 380 } },
  Amused: { cluster: 'joy', style: { fontWeight: 500 }, dur: { entrance: 620, emoji: 1400 } },
  Playful: { cluster: 'play', style: { fontWeight: 600 }, dur: { entrance: 600, emoji: 1100 } },
  Surprised: { cluster: 'play', style: { textTransform: 'uppercase', letterSpacing: '0.04em' }, dur: { entrance: 420, emoji: 2600 } },
  Calm: { cluster: 'calm', style: { fontWeight: 500 }, dur: { entrance: 900, emoji: 4200 } },
  Content: { cluster: 'calm', style: { fontWeight: 600 }, dur: { entrance: 850, emoji: 3800 } },
  Relieved: { cluster: 'calm', style: { fontWeight: 500, letterSpacing: '0.02em' }, dur: { entrance: 800, emoji: 3400 } },
  Sad: { cluster: 'sad', style: { fontStyle: 'italic' }, dur: { entrance: 1000, emoji: 3200 } },
  Disappointed: { cluster: 'sad', style: { fontStyle: 'italic', opacity: 0.92 }, dur: { entrance: 950, emoji: 3000 } },
  Lonely: { cluster: 'sad', style: { fontStyle: 'italic', letterSpacing: '0.06em', opacity: 0.88 }, dur: { entrance: 1100, emoji: 4000 } },
  Tired: { cluster: 'sad', entrance: 'droop', emoji: 'droop', style: { fontStyle: 'italic', letterSpacing: '0.04em', opacity: 0.9 }, dur: { entrance: 1150, emoji: 5200 } },
  Anxious: { cluster: 'anxiety', style: {}, dur: { entrance: 560, emoji: 220 } },
  Worried: { cluster: 'anxiety', style: { letterSpacing: '0.01em' }, dur: { entrance: 640, emoji: 420 } },
  Concerned: { cluster: 'anxiety', style: {}, dur: { entrance: 700, emoji: 600 } },
  Confused: { cluster: 'anxiety', style: { fontStyle: 'italic' }, dur: { entrance: 620, emoji: 900 } },
  Embarrassed: { cluster: 'anxiety', entrance: 'shrinkBack', emoji: 'shrinkBack', style: {}, dur: { entrance: 640, emoji: 3200 } },
  Love: { cluster: 'tender', style: { fontWeight: 700 }, dur: { entrance: 700, emoji: 1300 } },
  Caring: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 750, emoji: 1600 } },
  Grateful: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 780, emoji: 1800 } },
  Helpful: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 720, emoji: 1500 } },
  Hopeful: { cluster: 'drive', style: { fontWeight: 500 }, dur: { entrance: 780, emoji: 3000 } },
  Proud: { cluster: 'drive', style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 700, emoji: 2600 } },
  Determined: { cluster: 'drive', style: { textTransform: 'uppercase', fontWeight: 700 }, dur: { entrance: 560, emoji: 1400 } },
  Neutral: { cluster: 'reflective', style: {}, dur: { entrance: 700, emoji: 6000 } },
  Curious: { cluster: 'reflective', style: { fontWeight: 600 }, dur: { entrance: 650, emoji: 3200 } },
  Thoughtful: { cluster: 'reflective', style: { fontStyle: 'italic' }, dur: { entrance: 800, emoji: 4200 } },
}

export function resolveFeeling(feeling) {
  const f = FEELINGS[feeling] ?? FEELINGS.Neutral
  const c = CLUSTERS[f.cluster]
  return {
    cluster: f.cluster,
    font: c.font,
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

export function topFeelings(feelingScores, feelings, selected) {
  if (!feelingScores) return []
  const ranked = feelings
    .map((f, i) => ({ f, p: feelingScores[i] }))
    .sort((a, b) => b.p - a.p)
    .map((x) => x.f)
  const top = ranked.slice(0, 5)
  if (selected && !top.includes(selected)) top.push(selected)
  return top
}
