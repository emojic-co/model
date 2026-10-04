export const EGG_PHRASE = 'emojify.ing'
export const EGG_EMOJIS = ['🎉', '✨', '😀', '🥳', '🌈', '💖', '🚀', '🎈']
export const EGG_COUNT = 40
export const EGG_MS = 4000

export function isEasterEgg(text) {
  return text.trim().toLowerCase() === EGG_PHRASE
}

// Random layout: each piece gets a column, delay, duration and glyph.
export function eggPieces(rand = Math.random) {
  return Array.from({ length: EGG_COUNT }, (_, i) => ({
    id: i,
    emoji: EGG_EMOJIS[Math.floor(rand() * EGG_EMOJIS.length)],
    left: rand() * 100,
    delay: rand() * 1.2,
    duration: 2 + rand() * 1.5,
    size: 1.5 + rand() * 1.5,
  }))
}
