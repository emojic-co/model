import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'
import { FeelingBar } from './FeelingBar'

export function Card({ text, emoji, feeling, feelingOptions, palette, revision, onPickFeeling, onCopy }) {
  const textRef = useFitText(text, { min: 32, max: 104, key: feeling })
  const pal = feeling && palette ? palette[feeling] ?? palette.Neutral : null
  const r = feeling ? resolveFeeling(feeling) : null
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
      data-feeling={feeling || undefined}
      data-cluster={r?.cluster || undefined}
      data-entrance={r?.entrance || undefined}
      data-emoji={r?.emoji || undefined}
      style={style}
    >
      <span className="card-emoji">{emoji}</span>
      <div className="card-text-box" ref={textRef}>
        <p className="card-text" key={`${revision}:${feeling}`} style={r?.style}>
          {text}
        </p>
      </div>
      {feeling ? (
        <FeelingBar feelings={feelingOptions} active={feeling} onPick={onPickFeeling} />
      ) : (
        <span className="card-feeling-idle">—</span>
      )}
      <button className="copy-btn" type="button" aria-label="Copy card as image" onClick={onCopy}>
        copy
      </button>
    </div>
  )
}
