import { useEffect, useRef, useState } from 'react'
import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'
import { contrastRatio, patternTint } from '../model'
import { patternLayers, patternSizeCss } from '../patterns'

function watermarkInk(bg) {
  return contrastRatio('#000000', bg) >= contrastRatio('#ffffff', bg) ? '#000000' : '#ffffff'
}

const FADE_MS = 150

export function Card({ text, emoji, feeling, colors, loading, onCopy, onShare }) {
  const [shown, setShown] = useState({ emoji, feeling })
  const [phase, setPhase] = useState('in')
  const prev = useRef({ emoji, feeling })
  const wasLoading = useRef(loading)

  useEffect(() => {
    if (loading) {
      wasLoading.current = true
      prev.current = { emoji, feeling }
      setPhase('out')
      return
    }
    if (wasLoading.current) {
      wasLoading.current = false
      prev.current = { emoji, feeling }
      setShown({ emoji, feeling })
      setPhase('in')
      return
    }
    if (prev.current.emoji === emoji && prev.current.feeling === feeling) return
    prev.current = { emoji, feeling }
    setPhase('out')
    const t = setTimeout(() => {
      setShown({ emoji, feeling })
      setPhase('in')
    }, FADE_MS)
    return () => clearTimeout(t)
  }, [emoji, feeling, loading])

  const placeholder = !text.trim()
  const displayText = placeholder ? "What's on your mind?" : text
  const textRef = useFitText(displayText, { min: 5, max: 13, key: shown.feeling })
  const r = shown.feeling ? resolveFeeling(shown.feeling) : null
  const style =
    !loading && colors && r
      ? (() => {
          const layers = patternLayers(shown.feeling, patternTint(colors.bg1, colors.bg2))
          return {
            backgroundImage: [
              ...layers.map((l) => l.image),
              `linear-gradient(135deg, ${colors.bg1}, ${colors.bg2})`,
            ].join(', '),
            backgroundSize: [...patternSizeCss(layers), 'auto'].join(', '),
            color: colors.text_color,
            fontFamily: r.font,
            ...r.vars,
          }
        })()
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
      <button className="share-btn" type="button" aria-label="Share link" onClick={onShare}>
        share
      </button>
      <button className="copy-btn" type="button" aria-label="Copy card as image" onClick={onCopy}>
        copy
      </button>
      <span
        className="card-watermark"
        aria-hidden="true"
        style={colors ? { color: watermarkInk(colors.bg2) } : undefined}
      >
        emojify.ing
      </span>
    </div>
  )
}
