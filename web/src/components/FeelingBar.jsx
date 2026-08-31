import { resolveFeeling } from '../feelings'

export function FeelingBar({ feelings, active, palette, onPick }) {
  return (
    <div className="feeling-bar">
      {feelings.map((f) => {
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
      })}
    </div>
  )
}
