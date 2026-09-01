export const DEFAULT_COLORS = { bg1: '#a8e2f4', bg2: '#78c9f4', text_color: '#282e36' }

const SANS = 'system-ui, sans-serif'
const SERIF = 'Georgia, serif'
const HAND = '"Segoe Script", cursive'

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
  Angry: { cluster: 'anger', font: `"Anton", ${SANS}`, style: { textTransform: 'uppercase', letterSpacing: '0.06em' }, dur: { entrance: 420, emoji: 450 } },
  Annoyed: { cluster: 'anger', font: `"Oswald", ${SANS}`, style: { letterSpacing: '-0.01em' }, dur: { entrance: 500, emoji: 950 } },
  Frustrated: { cluster: 'anger', font: `"Archivo Black", ${SANS}`, style: { textTransform: 'uppercase' }, dur: { entrance: 520, emoji: 600 } },
  Disgusted: { cluster: 'anger', font: `"Griffy", ${HAND}`, style: { fontStyle: 'italic', letterSpacing: '0.03em' }, dur: { entrance: 480, emoji: 700 } },
  Happy: { cluster: 'joy', font: `"Fredoka", ${SANS}`, style: { fontWeight: 600 }, dur: { entrance: 560, emoji: 900 } },
  Excited: { cluster: 'joy', font: `"Chewy", ${SANS}`, style: { textTransform: 'uppercase', letterSpacing: '0.05em' }, dur: { entrance: 460, emoji: 380 } },
  Amused: { cluster: 'joy', font: `"Baloo 2", ${SANS}`, style: { fontWeight: 500 }, dur: { entrance: 620, emoji: 1400 } },
  Playful: { cluster: 'play', font: `"Bungee", ${SANS}`, style: {}, dur: { entrance: 600, emoji: 1100 } },
  Surprised: { cluster: 'play', font: `"Luckiest Guy", ${SANS}`, style: { letterSpacing: '0.04em' }, dur: { entrance: 420, emoji: 2600 } },
  Calm: { cluster: 'calm', font: `"Quicksand", ${SANS}`, style: { fontWeight: 500 }, dur: { entrance: 900, emoji: 4200 } },
  Content: { cluster: 'calm', font: `"Nunito", ${SANS}`, style: { fontWeight: 600 }, dur: { entrance: 850, emoji: 3800 } },
  Relieved: { cluster: 'calm', font: `"Comfortaa", ${SANS}`, style: { fontWeight: 500, letterSpacing: '0.02em' }, dur: { entrance: 800, emoji: 3400 } },
  Sad: { cluster: 'sad', font: `"Playfair Display", ${SERIF}`, style: { fontStyle: 'italic' }, dur: { entrance: 1000, emoji: 3200 } },
  Disappointed: { cluster: 'sad', font: `"Lora", ${SERIF}`, style: { fontStyle: 'italic', opacity: 0.92 }, dur: { entrance: 950, emoji: 3000 } },
  Lonely: { cluster: 'sad', font: `"EB Garamond", ${SERIF}`, style: { fontStyle: 'italic', letterSpacing: '0.06em', opacity: 0.88 }, dur: { entrance: 1100, emoji: 4000 } },
  Tired: { cluster: 'sad', font: `"Cormorant Garamond", ${SERIF}`, entrance: 'droop', emoji: 'droop', style: { fontStyle: 'italic', letterSpacing: '0.04em', opacity: 0.9 }, dur: { entrance: 1150, emoji: 5200 } },
  Longing: { cluster: 'sad', font: `"Spectral", ${SERIF}`, style: { fontStyle: 'italic', letterSpacing: '0.05em', opacity: 0.9 }, dur: { entrance: 1050, emoji: 4200 } },
  Anxious: { cluster: 'anxiety', font: `"Shantell Sans", ${SANS}`, style: {}, dur: { entrance: 560, emoji: 220 } },
  Worried: { cluster: 'anxiety', font: `"Gaegu", ${HAND}`, style: { letterSpacing: '0.01em' }, dur: { entrance: 640, emoji: 420 } },
  Concerned: { cluster: 'anxiety', font: `"Neucha", ${HAND}`, style: {}, dur: { entrance: 700, emoji: 600 } },
  Confused: { cluster: 'anxiety', font: `"Patrick Hand", ${HAND}`, style: { letterSpacing: '0.02em' }, dur: { entrance: 620, emoji: 900 } },
  Embarrassed: { cluster: 'anxiety', font: `"Schoolbell", ${HAND}`, entrance: 'shrinkBack', emoji: 'shrinkBack', style: {}, dur: { entrance: 640, emoji: 3200 } },
  Afraid: { cluster: 'anxiety', font: `"Give You Glory", ${HAND}`, style: { letterSpacing: '0.03em', opacity: 0.9 }, dur: { entrance: 600, emoji: 260 } },
  Love: { cluster: 'tender', font: `"Caveat", ${HAND}`, style: { fontWeight: 700 }, dur: { entrance: 700, emoji: 1300 } },
  Loving: { cluster: 'tender', font: `"Shadows Into Light", ${HAND}`, style: { letterSpacing: '0.01em' }, dur: { entrance: 720, emoji: 1400 } },
  Caring: { cluster: 'tender', font: `"Dancing Script", ${HAND}`, style: { fontWeight: 600 }, dur: { entrance: 750, emoji: 1600 } },
  Grateful: { cluster: 'tender', font: `"Kalam", ${HAND}`, style: { fontWeight: 400 }, dur: { entrance: 780, emoji: 1800 } },
  Helpful: { cluster: 'tender', font: `"Gochi Hand", ${HAND}`, style: {}, dur: { entrance: 720, emoji: 1500 } },
  Hopeful: { cluster: 'drive', font: `"Poppins", ${SANS}`, style: { fontWeight: 500 }, dur: { entrance: 780, emoji: 3000 } },
  Proud: { cluster: 'drive', font: `"Rubik", ${SANS}`, style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 700, emoji: 2600 } },
  Determined: { cluster: 'drive', font: `"Barlow Condensed", ${SANS}`, style: { textTransform: 'uppercase', fontWeight: 700 }, dur: { entrance: 560, emoji: 1400 } },
  Neutral: { cluster: 'reflective', font: `"Inter", ${SANS}`, style: {}, dur: { entrance: 700, emoji: 6000 } },
  Curious: { cluster: 'reflective', font: `"Work Sans", ${SANS}`, style: { fontWeight: 600 }, dur: { entrance: 650, emoji: 3200 } },
  Thoughtful: { cluster: 'reflective', font: `"Bitter", ${SERIF}`, style: { fontStyle: 'italic' }, dur: { entrance: 800, emoji: 4200 } },
}

export function resolveFeeling(feeling) {
  const f = FEELINGS[feeling] ?? FEELINGS.Neutral
  const c = CLUSTERS[f.cluster]
  return {
    cluster: f.cluster,
    font: f.font,
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
