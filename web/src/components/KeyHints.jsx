import { useI18n } from '../i18n'

const HINTS = [
  { keys: ['↑', '↓'], label: 'keys.emoji' },
  { keys: ['Alt', '↑', '↓'], label: 'keys.color' },
  { keys: ['Ctrl', '↑', '↓'], label: 'keys.feeling' },
  { keys: ['Enter'], label: 'keys.copy' },
  { keys: ['Esc'], label: 'keys.clear' },
]

export function KeyHints({ children }) {
  const { t } = useI18n()
  return (
    <aside className="keys" aria-label={t('keys.aria')}>
      <dl>
        {HINTS.map((h) => (
          <div key={h.label}>
            <dt>
              {h.keys.map((k) => (
                <kbd key={k}>{k}</kbd>
              ))}
            </dt>
            <dd>{t(h.label)}</dd>
          </div>
        ))}
      </dl>
      {children && <div className="keys-actions">{children}</div>}
    </aside>
  )
}
