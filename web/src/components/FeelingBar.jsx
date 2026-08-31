import { resolveFeeling } from '../feelings'

const PLACEHOLDERS = 5

export function FeelingBar({ feelings, active, palette, onPick, ready = true, hidden = false }) {
  const items = ready && feelings.length ? feelings : null
  return (
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
              <button
                key={f}
                type="button"
                className={f === active ? 'active' : undefined}
                style={style}
                onClick={() => onPick(f)}
              >
                {f}
              </button>
            )
          })
        : Array.from({ length: PLACEHOLDERS }, (_, i) => (
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
  )
}
