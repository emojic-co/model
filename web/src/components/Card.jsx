import { useFitText } from '../hooks/useFitText'
import { FEELING_FONTS } from '../fonts'
import { FeelingBar } from './FeelingBar'

export function Card({ text, emoji, feeling, feelings, palette, onPickFeeling, onCopy }) {
  const textRef = useFitText(text, { min: 32, max: 104, key: feeling })
  const pal = feeling && palette ? palette[feeling] ?? palette.Neutral : null
  const style = pal
    ? {
        backgroundImage: `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`,
        color: pal.text_color,
        fontFamily: FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral,
      }
    : undefined

  return (
    <div className="card" data-feeling={feeling || undefined} style={style}>
      <span className="card-emoji">{emoji}</span>
      <div className="card-text-box" ref={textRef}>
        <p className="card-text">{text}</p>
      </div>
      {feeling ? (
        <FeelingBar feelings={feelings} active={feeling} onPick={onPickFeeling} />
      ) : (
        <span className="card-feeling-idle">—</span>
      )}
      <button
        className="copy-btn"
        type="button"
        aria-label="Copy card as image"
        onClick={onCopy}
      >
        copy
      </button>
    </div>
  )
}
