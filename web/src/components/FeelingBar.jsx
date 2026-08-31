import { resolveFeeling } from '../feelings'

function FeelingButton({ feeling, active, style, onPick }) {
  return (
    <button
      type="button"
      className={active ? 'active' : undefined}
      style={style}
      onClick={() => onPick(feeling)}
    >
      {feeling}
    </button>
  )
}

export function FeelingBar({
  feelings,
  active,
  palette,
  onPick,
  ready = true,
  hidden = false,
  count = 5,
}) {
  const items = ready && feelings.length ? feelings : null
  return (
    <div className="feeling-bar-container">
      <div
        className={'feeling-bar' + (hidden ? ' is-hidden' : '')}
        aria-hidden={hidden ? 'true' : undefined}
      >
        {items
          ? items.map((f) => {
            const pal = palette ? palette[f] ?? palette.Neutral : null
            const r = resolveFeeling(f)
            const style = pal
              ? {
                backgroundImage: `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`,
                color: pal.text_color,
                fontFamily: r.font,
                ...r.style,
              }
              : undefined
            return (
              <FeelingButton
                key={f}
                feeling={f}
                active={f === active}
                style={style}
                onPick={onPick}
              />
            )
          })
          : Array.from({ length: count }, (_, i) => (
            <button
              key={i}
              type="button"
              className="placeholder"
              disabled
              aria-hidden="true"
              tabIndex={-1}
            />
          ))}
      </div>
    </div>
  )
}
