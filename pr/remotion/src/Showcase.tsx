import { AbsoluteFill, Audio, Easing, interpolate, random, staticFile, useCurrentFrame, useVideoConfig } from 'remotion'
import { eggPieces } from '../../../web/src/easterEgg.js'
import { toHexColor } from '../../../web/src/model.js'
import { CardClip, CardData } from './CardClip'
import cards from './showcase.json'

export const FPS = 30
export const WIDTH = 1080
export const HEIGHT = 1350

// Music: "Country Cue 1" by Audionautix (CC BY 4.0, https://audionautix.com).
const MUSIC = staticFile('mp3/Country Cue 1 - Audionautix.mp3')
const FADE_IN_S = 0.5
const FADE_OUT_S = 1.5

const CARDS = cards as CardData[]
const CARD = 960 // card edge in px
const FIRST_HOLD_S = 2.2
const HOLD_S = 2.0
const TRANS_S = 0.7
const OUTRO_HOLD_S = 3.8
const TRANSITIONS = ['push', 'slide', 'flip', 'slam', 'spin'] as const
type Kind = (typeof TRANSITIONS)[number]
const SLAM = TRANSITIONS.indexOf('slam')

// Transition j starts at transitionStart(j) and brings in card j+1 (the last one is the outro).
const transitionStart = (j: number) => FIRST_HOLD_S + j * (HOLD_S + TRANS_S)
export const DURATION_FRAMES = Math.ceil(
  (transitionStart(TRANSITIONS.length - 1) + TRANS_S + OUTRO_HOLD_S) * FPS,
)

type Pose = { x: number; y: number; scale: number; rotate: number; rotateY: number; opacity: number; blur: number }
const REST: Pose = { x: 0, y: 0, scale: 1, rotate: 0, rotateY: 0, opacity: 1, blur: 0 }
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const ease = (p: number, e: (t: number) => number) => e(clamp01(p))

// Poses of the outgoing and incoming card at transition progress p (0..1).
function poses(kind: Kind, p: number): [Pose, Pose] {
  switch (kind) {
    case 'push': {
      const e = ease(p, Easing.inOut(Easing.cubic))
      const d = WIDTH + 80
      return [{ ...REST, x: -e * d }, { ...REST, x: (1 - e) * d }]
    }
    case 'slide': {
      const e = ease(p, Easing.out(Easing.back(1.6)))
      const d = HEIGHT + 60
      const inY = (1 - e) * d
      return [{ ...REST, y: inY - d, scale: 1 - 0.1 * clamp01(p), opacity: 1 - 0.4 * clamp01(p) }, { ...REST, y: inY }]
    }
    case 'flip': {
      const out = ease(p * 2, Easing.in(Easing.quad))
      const inn = ease(p * 2 - 1, Easing.out(Easing.back(1.2)))
      return [
        { ...REST, rotateY: -90 * out, scale: 1 - 0.1 * out, opacity: p < 0.5 ? 1 : 0 },
        { ...REST, rotateY: 90 * (1 - inn), scale: 0.9 + 0.1 * inn, opacity: p < 0.5 ? 0 : 1 },
      ]
    }
    case 'slam': {
      const e = ease(p, Easing.in(Easing.exp))
      return [
        { ...REST, scale: 1 - 0.45 * e, opacity: 1 - e, blur: 14 * e },
        { ...REST, scale: 3.4 - 2.4 * e, opacity: clamp01(p * 6), blur: 10 * (1 - e) },
      ]
    }
    case 'spin': {
      const e = ease(p, Easing.inOut(Easing.cubic))
      return [
        { ...REST, rotate: 360 * e, scale: 1 - e, x: -200 * e, opacity: 1 - e * e },
        { ...REST, rotate: -540 * (1 - e), scale: 0.15 + 0.85 * e, y: 120 * (1 - e) },
      ]
    }
  }
}

const poseStyle = (p: Pose): React.CSSProperties => ({
  transform: `translate(${p.x}px, ${p.y}px) rotate(${p.rotate}deg) rotateY(${p.rotateY}deg) scale(${p.scale})`,
  opacity: p.opacity,
  filter: p.blur > 0.1 ? `blur(${p.blur}px)` : undefined,
})

const hex = (c: number[]) => toHexColor(c)

// Emoji rain from the web app's easter egg (web/src/easterEgg.js), driven by frame instead of CSS animation.
// Seeded so every render is identical.
let seed = 0
const PIECES = eggPieces(() => random(`egg-${seed++}`))

const EmojiRain: React.FC<{ t: number }> = ({ t }) => (
  <AbsoluteFill style={{ overflow: 'hidden', pointerEvents: 'none' }}>
    {PIECES.map((p) => {
      const dur = p.duration * 0.9
      const prog = (t - p.delay * 0.9) / dur
      if (prog < 0 || prog > 1) return null
      return (
        <span
          key={p.id}
          style={{
            position: 'absolute',
            left: `${p.left}%`,
            top: -120,
            fontSize: p.size * 40,
            transform: `translateY(${prog * (HEIGHT + 240)}px) rotate(${prog * 360}deg)`,
          }}
        >
          {p.emoji}
        </span>
      )
    })}
  </AbsoluteFill>
)

export const Showcase: React.FC = () => {
  const frame = useCurrentFrame()
  const { fps } = useVideoConfig()
  const t = frame / fps

  // Which transition is active (or most recently finished) decides who is on screen.
  const j = TRANSITIONS.findIndex((_, k) => t < transitionStart(k) + TRANS_S)
  const lastDone = j === -1
  const cur = lastDone ? TRANSITIONS.length : j // index of the incoming/current card
  const p = lastDone ? 1 : clamp01((t - transitionStart(j)) / TRANS_S)
  const transitioning = !lastDone && p > 0

  // Card k starts playing when it starts entering (card 0 at t=0).
  const cardMs = (k: number) => (t - (k === 0 ? 0 : transitionStart(k - 1))) * 1000

  const k = transitioning ? j : cur - 1 // outgoing card index when transitioning
  const [outPose, inPose] = transitioning ? poses(TRANSITIONS[j], p) : [REST, REST]

  // Impact shake after the zoom slam lands.
  const sinceSlam = t - (transitionStart(SLAM) + TRANS_S * 0.9)
  const shake = sinceSlam > 0 && sinceSlam < 0.4 ? Math.exp(-sinceSlam * 9) * Math.sin(sinceSlam * 70) * 22 : 0
  const flash =
    j === SLAM ? interpolate(p, [0.8, 0.92, 1], [0, 0.55, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) : 0

  // Background: current card palette, darkened, crossfaded during transitions.
  const bgOf = (idx: number) => hex(CARDS[Math.min(idx, CARDS.length - 1)].bg2)
  const shown = transitioning ? j + 1 : cur
  const outro = shown === CARDS.length - 1 && (t >= transitionStart(TRANSITIONS.length - 1) + TRANS_S * 0.5)

  const rainT = t - (transitionStart(TRANSITIONS.length - 1) + TRANS_S)

  const layer = (idx: number, pose: Pose, z: number) => (
    <div
      key={idx}
      style={{
        position: 'absolute',
        left: (WIDTH - CARD) / 2,
        top: (HEIGHT - CARD) / 2,
        zIndex: z,
        ...poseStyle(pose),
      }}
    >
      <CardClip card={CARDS[idx]} timeMs={cardMs(idx)} size={CARD} />
    </div>
  )

  const bgA = bgOf(transitioning ? j : shown)
  const bgB = bgOf(shown)
  return (
    <AbsoluteFill style={{ background: '#0f0c0a', perspective: 2200 }}>
      <Audio
        src={MUSIC}
        volume={(f) =>
          interpolate(
            f,
            [0, FADE_IN_S * FPS, DURATION_FRAMES - FADE_OUT_S * FPS, DURATION_FRAMES],
            [0, 0.8, 0.8, 0],
            { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
          )
        }
      />
      <AbsoluteFill style={{ background: bgA }} />
      <AbsoluteFill style={{ background: bgB, opacity: transitioning ? p : 1 }} />
      <AbsoluteFill style={{ background: 'radial-gradient(circle at 50% 45%, rgba(0,0,0,0.05), rgba(0,0,0,0.6))' }} />
      <AbsoluteFill style={{ transform: `translate(${shake}px, ${shake * 0.6}px)`, perspective: 2200 }}>
        {transitioning ? (
          <>
            {layer(k, outPose, 1)}
            {layer(k + 1, inPose, 2)}
          </>
        ) : (
          layer(shown, REST, 1)
        )}
      </AbsoluteFill>
      {outro && rainT > 0 ? <EmojiRain t={rainT} /> : null}
      <AbsoluteFill style={{ background: '#fff', opacity: flash, pointerEvents: 'none' }} />
    </AbsoluteFill>
  )
}
