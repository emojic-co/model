import { useEffect, useState } from 'react'
import {
  AbsoluteFill,
  Easing,
  continueRender,
  delayRender,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion'
import { Card } from '../../../web/src/components/Card.jsx'
import '../../../web/src/styles.css'
import './scene.css'
import { FAMILIES, FONT_HREF } from './fonts'
import data from './data.json'

export const PRE_S = 0.35
export const CPS = 7
export const LOADER_S = 0.8
export const SLIDE_S = 0.55
export const HOLD_S = 2.4

const LOAD_EMOJIS = [
  '✨', '🎨', '🔮', '🎭', '🌈', '💫', '🪄', '🎲', '🧩', '🎯',
  '🌸', '🍬', '🎈', '⭐', '🌟', '🍭', '🎪', '🦋', '🌀', '🎃',
  '🐙', '🍀', '🪅', '🧸', '🎵', '💎', '🚀', '🌻', '🍉', '🎁',
]

type Entry = {
  text: string
  slug: string
  emoji: string
  feeling: string
  bg1: string
  bg2: string
  text_color: string
}
const DATA = data as Record<string, Entry>

export function sceneDurationInFrames(text: string, fps: number) {
  const typing = Math.ceil((text.length / CPS) * fps)
  return Math.ceil(
    PRE_S * fps + typing + (LOADER_S + SLIDE_S + HOLD_S) * fps,
  )
}

function useGoogleFonts() {
  const [handle] = useState(() => delayRender('google-fonts'))
  useEffect(() => {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = FONT_HREF
    link.onload = async () => {
      try {
        await Promise.all(
          FAMILIES.map((f) =>
            document.fonts.load(`400 32px "${f}"`, 'Ag☕').catch(() => undefined),
          ),
        )
        await document.fonts.ready
      } catch {
        // ignore
      }
      continueRender(handle)
    }
    link.onerror = () => continueRender(handle)
    document.head.appendChild(link)
  }, [handle])
}

export const Scene: React.FC<{ slug: string }> = ({ slug }) => {
  useGoogleFonts()
  const frame = useCurrentFrame()
  const { fps } = useVideoConfig()

  const entry = DATA[slug]
  const full = entry.text

  const framesPerChar = fps / CPS
  const typed = Math.max(
    0,
    Math.min(full.length, Math.floor((frame - PRE_S * fps) / framesPerChar)),
  )
  const shownText = full.slice(0, typed)
  const typingEnd = PRE_S * fps + full.length * framesPerChar

  const slotIn = interpolate(
    frame,
    [typingEnd, typingEnd + 0.18 * fps],
    [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  )

  const slideStart = typingEnd + LOADER_S * fps
  const slide = interpolate(
    frame,
    [slideStart, slideStart + SLIDE_S * fps],
    [0, 1],
    {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.cubic),
    },
  )

  const caretOn = Math.floor(frame / (fps * 0.5)) % 2 === 0
  const loadEmoji = LOAD_EMOJIS[Math.floor(frame / (fps * 0.1)) % LOAD_EMOJIS.length]
  const dots = '.'.repeat(1 + (Math.floor(frame / (fps * 0.26)) % 3))

  return (
    <AbsoluteFill>
      <div className="pr-root">
        <div className="pr-cam">
          <div className="pr-col">
            <div className={'pr-input' + (shownText ? '' : ' is-empty')}>
              {shownText}
              <span className="pr-caret" style={{ opacity: caretOn ? 1 : 0 }}>
                &nbsp;
              </span>
            </div>

            <div className="pr-card-slot" style={{ opacity: slotIn }}>
              <div
                className="pr-layer pr-loading"
                style={{ transform: `translateX(${-100 * slide}%)` }}
              >
                <div className="pr-loading-row">
                  <span className="pr-loading-emoji">{loadEmoji}</span>
                  <span className="pr-loading-text">
                    emojifying<span className="pr-loading-dots">{dots}</span>
                  </span>
                </div>
              </div>
              <div
                className="pr-layer"
                style={{ transform: `translateX(${100 * (1 - slide)}%)` }}
              >
                <Card
                  text={full}
                  emoji={entry.emoji}
                  feeling={entry.feeling}
                  colors={{ bg1: entry.bg1, bg2: entry.bg2, text_color: entry.text_color }}
                  onCopy={() => undefined}
                />
              </div>
            </div>

            <footer className="pr-footer">made with ❤️ by Gilad</footer>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  )
}
