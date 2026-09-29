import { useEffect, useMemo, useRef } from 'react'
import { TEXT_ANIMATIONS, countUnits, isLongWord, playText, splitWords } from '../textAnimation'

// Renders text as per-unit spans and (re)plays the style's entrance whenever `replayKey`
// changes. Plain text edits do not replay, so typing stays calm.
export function CharText({ text, motif, feeling, replayKey, anim = TEXT_ANIMATIONS, ...rest }) {
  const ref = useRef(null)
  const words = useMemo(() => splitWords(text), [text])

  useEffect(() => {
    if (!motif) return
    return playText(ref.current, { anim, motif, feeling })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayKey, motif, feeling, anim])

  let i = 0
  return (
    <p ref={ref} {...rest}>
      {words.map((units, w) => (
        <span key={w}>
          {w > 0 && ' '}
          <span className={isLongWord(units) ? 'tw' : 'tw tw-nowrap'}>
            {units.map((u) => (
              <span key={i} className="tc" data-ci={i++}>
                {u}
              </span>
            ))}
          </span>
        </span>
      ))}
    </p>
  )
}

export { countUnits }
