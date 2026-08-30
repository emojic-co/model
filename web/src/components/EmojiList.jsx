export function EmojiList({ emojiScores, emojis, onPick }) {
  if (!emojiScores) {
    return <div className="emoji-list emoji-list-empty" aria-hidden="true" />
  }
  const top = emojiScores
    .map((p, i) => ({ emoji: emojis[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 10)
  return (
    <ul className="emoji-list">
      {top.map(({ emoji, p }) => (
        <li key={emoji}>
          <button type="button" onClick={() => onPick(emoji)}>
            <span className="emoji-list-glyph">{emoji}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${(p * 100).toFixed(1)}%` }} />
            </span>
            <span className="bar-pct">{(p * 100).toFixed(1)}%</span>
          </button>
        </li>
      ))}
    </ul>
  )
}
