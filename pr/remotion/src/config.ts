// The showcase template: everything about one video lives in a JSON file under configs/.
export type Dir = 'left' | 'right' | 'up' | 'down'
// Slides come in from a side (the old card is pushed out the opposite way); flips turn toward a side.
export type Transition = { type: 'slide'; from: Dir } | { type: 'flip'; to: Dir } | { type: 'fade' }

export type CardConfig = {
  text: string
  emoji: string // must have an original Noto animation (see validate.mjs)
  feeling: string // style name from web/src/feelings.js
  colors: { bg1: number[]; bg2: number[]; text_color: number[] } // OKLab, from the model's color GAN
  lang?: string // overrides the config's lang for this card
}

export type ShowcaseConfig = {
  name: string
  lang: string // 'en' | 'he'
  music: { file: string; credit: string; volume: number; fadeInS: number; fadeOutS: number }
  timing: { firstHoldS: number; holdS: number; transS: number; lastExtraHoldS: number; outroHoldS: number }
  transitions: Transition[] // transitions[j] brings in card j+1; one per card, the last (into the outro) is a fade
  cards: CardConfig[]
  outro: CardConfig // shown last, over the emoji rain
}

// Transition j starts at transitionStart(j); the last one waits an extra beat.
export const transitionStart = (c: ShowcaseConfig, j: number) =>
  c.timing.firstHoldS +
  j * (c.timing.holdS + c.timing.transS) +
  (j === c.transitions.length - 1 ? c.timing.lastExtraHoldS : 0)

export const durationInFrames = (c: ShowcaseConfig, fps: number) =>
  Math.ceil((transitionStart(c, c.transitions.length - 1) + c.timing.transS + c.timing.outroHoldS) * fps)
