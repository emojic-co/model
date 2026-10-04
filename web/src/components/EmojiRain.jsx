import { useEffect, useMemo, useState } from 'react'
import { eggPieces, EGG_MS } from '../easterEgg'

export function EmojiRain({ trigger }) {
  const [active, setActive] = useState(false)
  const pieces = useMemo(() => (trigger ? eggPieces() : []), [trigger])

  useEffect(() => {
    if (!trigger) return
    setActive(true)
    const t = setTimeout(() => setActive(false), EGG_MS)
    return () => clearTimeout(t)
  }, [trigger])

  if (!active) return null
  return (
    <div className="emoji-rain" aria-hidden="true">
      {pieces.map((p) => (
        <span
          key={p.id}
          style={{
            left: `${p.left}%`,
            fontSize: `${p.size}rem`,
            animationDelay: `${p.delay}s`,
            animationDuration: `${p.duration}s`,
          }}
        >
          {p.emoji}
        </span>
      ))}
    </div>
  )
}
