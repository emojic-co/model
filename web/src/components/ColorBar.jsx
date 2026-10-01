import { toCssOklab } from '../model'
import { useI18n } from '../i18n'

export function ColorBar({ palettes, active, onPick, ready = true, count = 5 }) {
  const { t } = useI18n()
  const items = ready && palettes.length ? palettes.slice(0, count) : null
  return (
    <div className="color-bar-container">
      <div className="color-bar">
        {items
          ? items.map((c, i) => (
            <button
              key={i}
              type="button"
              className={i === active ? 'active' : undefined}
              style={{ backgroundImage: `linear-gradient(135deg, ${toCssOklab(c.bg1)}, ${toCssOklab(c.bg2)})` }}
              aria-label={t('color.aria', { n: i + 1 })}
              aria-pressed={i === active}
              onClick={() => onPick(i)}
            >
              <span aria-hidden="true" style={{ color: toCssOklab(c.text_color) }}>
                Aa
              </span>
            </button>
          ))
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
