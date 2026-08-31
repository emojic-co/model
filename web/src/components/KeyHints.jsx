const HINTS = [
  { keys: ['↑', '↓'], label: 'emoji' },
  { keys: ['Ctrl', '↑', '↓'], label: 'feeling' },
  { keys: ['Enter'], label: 'copy image' },
  { keys: ['Esc'], label: 'clear text' },
]

export function KeyHints() {
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
    </aside>
  )
}
