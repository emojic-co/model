import { useEffect, useRef, useState } from 'react'
import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'

const FADE_MS = 150

export function Card({ text, emoji, feeling, palette, onCopy }) {
  const [shown, setShown] = useState({ emoji, feeling })
  const [phase, setPhase] = useState('in')
  const prev = useRef({ emoji, feeling })

  useEffect(() => {
    if (prev.current.emoji === emoji && prev.current.feeling === feeling) return
    prev.current = { emoji, feeling }
    setPhase('out')
    const t = setTimeout(() => {
      setShown({ emoji, feeling })
      setPhase('in')
    }, FADE_MS)
    return () => clearTimeout(t)
  }, [emoji, feeling])

  const textRef = useFitText(text, { min: 32, max: 104, key: shown.feeling })
  const pal = shown.feeling && palette ? palette[shown.feeling] ?? palette.Neutral : null
  const r = shown.feeling ? resolveFeeling(shown.feeling) : null
  const style =
    pal && r
      ? {
          backgroundImage: `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`,
          color: pal.text_color,
          fontFamily: r.font,
          ...r.vars,
        }
      : undefined

  return (
    <div
      className="card"
      data-feeling={shown.feeling || undefined}
      data-cluster={r?.cluster || undefined}
      data-entrance={r?.entrance || undefined}
      data-emoji={r?.emoji || undefined}
      data-phase={phase}
      style={style}
    >
      <span className="card-emoji">{shown.emoji}</span>
      <div className="card-text-box" ref={textRef}>
        <p className="card-text" style={r?.style}>
          {text}
        </p>
      </div>
      <button className="copy-btn" type="button" aria-label="Copy card as image" onClick={onCopy}>
        copy
      </button>
    </div>
  )
}
