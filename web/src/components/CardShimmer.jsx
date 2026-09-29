import { useEffect, useRef } from 'react'
import { TEXT_ANIMATIONS, countUnits, entranceTotalMs, splitWords } from '../textAnimation'
import { SHIMMERS, playShimmer } from '../shimmer'

// Background shimmer overlay. Starts only once the text entrance has fully finished and
// restarts whenever `replayKey` changes (the same key that replays the entrance).
export function CardShimmer({ cluster, feeling, motif, text, replayKey, anim = SHIMMERS, textAnim = TEXT_ANIMATIONS }) {
  const ref = useRef(null)
  const textRef = useRef(text)
  textRef.current = text

  useEffect(() => {
    if (!cluster) return
    const n = countUnits(splitWords(textRef.current))
    const entranceMs = motif ? entranceTotalMs(textAnim, motif, feeling, n) : 0
    return playShimmer(ref.current, { anim, cluster, feeling, entranceMs })
  }, [replayKey, cluster, feeling, motif, anim, textAnim])

  return <div className="card-shimmer" ref={ref} aria-hidden="true" />
}
