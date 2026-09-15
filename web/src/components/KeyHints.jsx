import { useMemo } from 'react'
import { translationSupported, supportedLanguages, languageName } from '../translate'

const HINTS = [
  { keys: ['↑', '↓'], label: 'emoji' },
  { keys: ['Alt', '↑', '↓'], label: 'color' },
  { keys: ['Ctrl', '↑', '↓'], label: 'feeling' },
  { keys: ['Enter'], label: 'copy image' },
  { keys: ['Esc'], label: 'clear text' },
]

export function KeyHints({ detectedLang, langOverride, onSelectLang }) {
  const supported = useMemo(() => translationSupported(), [])
  const langs = useMemo(() => (supported ? supportedLanguages() : []), [supported])

  return (
    <aside className="keys" aria-label="Keyboard shortcuts">
      <dl>
        {HINTS.map((h) => (
          <div key={h.label}>
            <dt>
              {h.keys.map((k) => (
                <kbd key={k}>{k}</kbd>
              ))}
            </dt>
            <dd>{h.label}</dd>
          </div>
        ))}
      </dl>
      {supported ? (
        <div className="langs">
          <p className="lang-caption">source language</p>
          <ul className="lang-list">
            <li>
              <button
                type="button"
                className={langOverride === null ? 'active' : ''}
                onClick={() => onSelectLang(null)}
              >
                Auto{detectedLang ? ` (${languageName(detectedLang)})` : ''}
              </button>
            </li>
            <li>
              <button
                type="button"
                className={langOverride === 'en' ? 'active' : ''}
                onClick={() => onSelectLang('en')}
              >
                English
              </button>
            </li>
            {langs.map((l) => (
              <li key={l.code}>
                <button
                  type="button"
                  className={langOverride === l.code ? 'active' : ''}
                  onClick={() => onSelectLang(l.code)}
                >
                  {l.name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="langs">translation not supported in this browser — English only</p>
      )}
    </aside>
  )
}
