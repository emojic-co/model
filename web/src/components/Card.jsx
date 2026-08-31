import { useEffect, useRef, useState } from 'react'
import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'

const FADE_MS = 150

export function Card({ text, emoji, feeling, colors, onCopy }) {
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

  const placeholder = !text.trim()
  const displayText = placeholder ? "What's on your mind?" : text
  const textRef = useFitText(displayText, { min: 5, max: 13, key: shown.feeling })
  const r = shown.feeling ? resolveFeeling(shown.feeling) : null
  const style =
    colors && r
      ? {
          backgroundImage: `linear-gradient(135deg, ${colors.bg1}, ${colors.bg2})`,
          color: colors.text_color,
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
        <p
          className={'card-text' + (placeholder ? ' card-text-placeholder' : '')}
          style={r?.style}
        >
          {displayText}
        </p>
      </div>
      <button className="copy-btn" type="button" aria-label="Copy card as image" onClick={onCopy}>
        copy
      </button>
    </div>
  )
}
