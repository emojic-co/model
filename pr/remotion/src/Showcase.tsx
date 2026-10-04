import { AbsoluteFill, Audio, Easing, interpolate, random, staticFile, useCurrentFrame, useVideoConfig } from 'remotion'
import { eggPieces } from '../../../web/src/easterEgg.js'
import { toHexColor } from '../../../web/src/model.js'
import { CardClip } from './CardClip'
import { ShowcaseConfig, Transition, transitionStart } from './config'

export const FPS = 30
export const WIDTH = 1080
export const HEIGHT = 1350
const CARD = 960 // card edge in px

type Pose = { x: number; y: number; scale: number; rotateX: number; rotateY: number; opacity: number }
const REST: Pose = { x: 0, y: 0, scale: 1, rotateX: 0, rotateY: 0, opacity: 1 }
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const ease = (p: number, e: (t: number) => number) => e(clamp01(p))

// Poses of the outgoing and incoming card at transition progress p (0..1).
function poses(tr: Transition, p: number): [Pose, Pose] {
  switch (tr.type) {
    case 'slide': {
      const e = ease(p, Easing.inOut(Easing.cubic))
      const horizontal = tr.from === 'left' || tr.from === 'right'
      const d = horizontal ? WIDTH + 80 : HEIGHT + 60
      const sign = tr.from === 'right' || tr.from === 'down' ? 1 : -1
      const inPos = (1 - e) * d * sign
      const axis = (v: number) => (horizontal ? { x: v } : { y: v })
      return [{ ...REST, ...axis(inPos - d * sign) }, { ...REST, ...axis(inPos) }]
    }
    case 'flip': {
      const out = ease(p * 2, Easing.in(Easing.quad))
      const inn = ease(p * 2 - 1, Easing.out(Easing.back(1.2)))
      const horizontal = tr.to === 'left' || tr.to === 'right'
      const sign = tr.to === 'left' || tr.to === 'down' ? -1 : 1
      const axis = (deg: number) => (horizontal ? { rotateY: deg } : { rotateX: deg })
      return [
        { ...REST, ...axis(90 * sign * out), scale: 1 - 0.1 * out, opacity: p < 0.5 ? 1 : 0 },
        { ...REST, ...axis(-90 * sign * (1 - inn)), scale: 0.9 + 0.1 * inn, opacity: p < 0.5 ? 0 : 1 },
      ]
    }
    case 'fade': {
      const e = ease(p, Easing.inOut(Easing.quad))
      return [{ ...REST, opacity: 1 - e }, { ...REST, scale: 0.96 + 0.04 * e, opacity: e }]
    }
  }
}

const poseStyle = (p: Pose): React.CSSProperties => ({
  transform: `translate(${p.x}px, ${p.y}px) rotateX(${p.rotateX}deg) rotateY(${p.rotateY}deg) scale(${p.scale})`,
  opacity: p.opacity,
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

export const Showcase: React.FC<{ config: ShowcaseConfig }> = ({ config }) => {
  const { transitions: TRANSITIONS, timing, music } = config
  const CARDS = [...config.cards, config.outro]
  const start = (j: number) => transitionStart(config, j)
  const TRANS_S = timing.transS
  const frame = useCurrentFrame()
  const { fps, durationInFrames: total } = useVideoConfig()
  const t = frame / fps

  // Which transition is active (or most recently finished) decides who is on screen.
  const j = TRANSITIONS.findIndex((_, k) => t < start(k) + TRANS_S)
  const lastDone = j === -1
  const cur = lastDone ? TRANSITIONS.length : j // index of the incoming/current card
  const p = lastDone ? 1 : clamp01((t - start(j)) / TRANS_S)
  const transitioning = !lastDone && p > 0

  // Card k starts playing when it starts entering (card 0 at t=0).
  const cardMs = (k: number) => (t - (k === 0 ? 0 : start(k - 1))) * 1000

  const k = transitioning ? j : cur - 1 // outgoing card index when transitioning
  const [outPose, inPose] = transitioning ? poses(TRANSITIONS[j], p) : [REST, REST]

  // Background: current card palette, darkened, crossfaded during transitions.
  const bgOf = (idx: number) => hex(CARDS[Math.min(idx, CARDS.length - 1)].colors.bg2)
  const shown = transitioning ? j + 1 : cur
  const isOutro = shown === CARDS.length - 1 && (t >= start(TRANSITIONS.length - 1) + TRANS_S * 0.5)

  const rainT = t - (start(TRANSITIONS.length - 1) + TRANS_S)

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
      <CardClip card={CARDS[idx]} lang={config.lang} timeMs={cardMs(idx)} size={CARD} />
    </div>
  )

  const bgA = bgOf(transitioning ? j : shown)
  const bgB = bgOf(shown)
  return (
    <AbsoluteFill style={{ background: '#0f0c0a', perspective: 2200 }}>
      <Audio
        src={staticFile(music.file)}
        volume={(f) =>
          interpolate(
            f,
            [0, music.fadeInS * FPS, total - music.fadeOutS * FPS, total],
            [0, music.volume, music.volume, 0],
            { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
          )
        }
      />
      <AbsoluteFill style={{ background: bgA }} />
      <AbsoluteFill style={{ background: bgB, opacity: transitioning ? p : 1 }} />
      <AbsoluteFill style={{ background: 'radial-gradient(circle at 50% 45%, rgba(0,0,0,0.05), rgba(0,0,0,0.6))' }} />
      <AbsoluteFill style={{ perspective: 2200 }}>
        {transitioning ? (
          <>
            {layer(k, outPose, 1)}
            {layer(k + 1, inPose, 2)}
          </>
        ) : (
          layer(shown, REST, 1)
        )}
      </AbsoluteFill>
      {isOutro && rainT > 0 ? <EmojiRain t={rainT} /> : null}
    </AbsoluteFill>
  )
}
