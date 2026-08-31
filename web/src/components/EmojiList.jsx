export function EmojiList({ items, active, onPick, slots = 10 }) {
  const ready = !!items
  const top = ready ? items : []
  return (
    <ul className="emoji-list" data-ready={ready ? 'true' : undefined}>
      {Array.from({ length: slots }, (_, i) => {
        const item = top[i]
        return (
          <li key={i}>
            <button
              type="button"
              className={item && item.emoji === active ? 'active' : undefined}
              disabled={!item}
              aria-hidden={item ? undefined : 'true'}
              tabIndex={item ? undefined : -1}
              onClick={item ? () => onPick(item.emoji) : undefined}
            >
              <span className="emoji-list-glyph">{item ? item.emoji : ''}</span>
              <span className="emoji-list-weight">{item ? item.p.toFixed(2) : ''}</span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
