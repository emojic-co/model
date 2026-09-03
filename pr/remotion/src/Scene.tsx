import { useEffect, useState } from 'react'
import {
  AbsoluteFill,
  continueRender,
  delayRender,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion'
import { Card } from '../../../web/src/components/Card.jsx'
import { DEFAULT_COLORS } from '../../../web/src/feelings.js'
import meta from '../../../web/public/meta.json'
import config from '../../../web/public/config.json'
import '../../../web/src/styles.css'
import './scene.css'
import { FAMILIES, FONT_HREF } from './fonts'
import data from './data.json'

export const START_DELAY_S = 0.6
export const CPS = 11
export const HOLD_S = 2.6

type Frame = {
  k: number
  normLen: number
  meaningful: boolean
  emoji: string
  feeling: string
  bg1: string
  bg2: string
  text_color: string
}
type Entry = { text: string; slug: string; frames: Frame[] }
const DATA = data as Record<string, Entry>

const MAX_LEN = config.max_text_len

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

function modelDate(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
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
    Math.min(full.length, Math.floor((frame - START_DELAY_S * fps) / framesPerChar)),
  )
  const shownText = full.slice(0, typed)
  const pf = typed > 0 ? entry.frames[typed - 1] : null
  const meaningful = !!pf?.meaningful

  const emoji = meaningful ? pf!.emoji : '🙂'
  const feeling = meaningful ? pf!.feeling : 'Neutral'
  const colors = meaningful
    ? { bg1: pf!.bg1, bg2: pf!.bg2, text_color: pf!.text_color }
    : DEFAULT_COLORS

  const caretOn = Math.floor(frame / (fps * 0.53)) % 2 === 0
  const showWarn = shownText.trim().length > 0 && !meaningful
  const atMax = shownText.length >= MAX_LEN

  return (
    <AbsoluteFill>
      <div className="pr-root">
        <div className="pr-col">
          <header className="masthead">
            <h1>
              emojify<span className="tld">.ing</span>
            </h1>
          </header>
          <div className="input" style={{ color: shownText ? undefined : '#6b6b6b' }}>
            {shownText || 'type at least 3 characters…'}
            <span className="pr-caret" style={{ opacity: caretOn ? 1 : 0, color: '#1a1a1a' }}>
              |
            </span>
          </div>
          <div className="input-meta">
            <p className={'warn' + (showWarn ? '' : ' is-hidden')}>
              text is too short — showing a default card
            </p>
            <div className={'counter' + (atMax ? ' full' : '')}>
              {shownText.length}
              <span>/{MAX_LEN}</span>
            </div>
          </div>
        </div>
        <Card
          text={shownText}
          emoji={emoji}
          feeling={feeling}
          colors={colors}
          onCopy={() => undefined}
        />
        <footer className="footer">
          <span>
            model updated <span>{modelDate(meta.exported_at)}</span>
          </span>
          <span>made with ❤️ by Gilad</span>
        </footer>
      </div>
    </AbsoluteFill>
  )
}
