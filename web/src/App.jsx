import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize } from './model'
import { topFeelings } from './feelings'
import { cycle } from './nav'
import GitHubButton from 'react-github-btn'
import { Card } from './components/Card'
import { FeelingBar } from './components/FeelingBar'
import { EmojiList } from './components/EmojiList'
import { KeyHints } from './components/KeyHints'
import { useCardImage } from './hooks/useCardImage'
import { Toast } from './components/Toast'
import { useMediaQuery } from './hooks/useMediaQuery'

const MIN_CHARS = 3
const DEBOUNCE_MS = 250

function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short',
      })
}

export function App() {
  const { meta, config, palette, ready, predict } = useOnnx()
  const mobile = useMediaQuery('(max-width: 56.25em)')
  const emojiSlots = mobile ? 9 : 10
  const feelingCount = mobile ? 4 : 5
  const [text, setText] = useState('')
  const [scores, setScores] = useState(null)
  const [override, setOverride] = useState({ emoji: null, feeling: null })
  const [toast, setToast] = useState({ msg: '', n: 0 })
  const showToast = useCallback((msg) => setToast((s) => ({ msg, n: s.n + 1 })), [])
  const seq = useRef(0)
  const inputRef = useRef(null)

  const char2idx = useMemo(
    () => (meta ? new Map([...meta.chars].map((c, i) => [c, i])) : null),
    [meta],
  )

  useEffect(() => {
    if (!ready || !char2idx) return
    if (normalize(text, char2idx).length < MIN_CHARS) {
      seq.current++
      setScores(null)
      setOverride({ emoji: null, feeling: null })
      return
    }
    const mine = ++seq.current
    const timer = setTimeout(async () => {
      const logits = await predict(text)
      if (mine !== seq.current) return
      setScores(logits)
      setOverride({ emoji: null, feeling: null })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [text, ready, char2idx, predict])

  const emojiScores = scores && scores.emoji
  const predictedEmoji = scores ? meta.emojis[argmax(scores.emoji)] : null
  const predictedFeeling = scores ? meta.feelings[argmax(scores.feeling)] : null
  const shownEmoji = override.emoji ?? predictedEmoji
  const shownFeeling = override.feeling ?? predictedFeeling
  const feelingOptions = useMemo(
    () => topFeelings(scores?.feeling, meta?.feelings ?? [], shownFeeling, feelingCount),
    [scores, meta, shownFeeling, feelingCount],
  )
  const emojiTop = useMemo(
    () =>
      emojiScores
        ? emojiScores
            .map((p, i) => ({ emoji: meta.emojis[i], p }))
            .sort((a, b) => b.p - a.p)
            .slice(0, emojiSlots)
        : null,
    [emojiScores, meta, emojiSlots],
  )
  const emojiList = useMemo(() => emojiTop?.map((x) => x.emoji) ?? [], [emojiTop])

  const cardData =
    shownEmoji && shownFeeling && palette
      ? {
          text,
          emoji: shownEmoji,
          feeling: shownFeeling,
          pal: palette[shownFeeling] ?? palette.Neutral,
        }
      : null
  const copyCard = useCardImage(cardData, showToast)

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setText('')
        inputRef.current?.focus()
        return
      }
      if (e.key === 'Enter') {
        if (e.target instanceof HTMLButtonElement) return
        e.preventDefault()
        copyCard()
        return
      }
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
      const dir = e.key === 'ArrowDown' ? 1 : -1
      if (e.ctrlKey) {
        if (!feelingOptions.length) return
        e.preventDefault()
        setOverride((o) => ({
          ...o,
          feeling: cycle(feelingOptions, o.feeling ?? predictedFeeling, dir),
        }))
        return
      }
      if (e.shiftKey || e.altKey || e.metaKey || !emojiList.length) return
      e.preventDefault()
      setOverride((o) => ({ ...o, emoji: cycle(emojiList, o.emoji ?? predictedEmoji, dir) }))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [emojiList, feelingOptions, predictedEmoji, predictedFeeling, copyCard])

  const maxLen = config?.max_text_len ?? 0
  const tooShort =
    ready && char2idx ? normalize(text, char2idx).length < MIN_CHARS : false
  const displayEmoji = shownEmoji ?? '🙂'
  const displayFeeling = shownFeeling ?? 'Neutral'

  return (
    <main>
      <div className="stage">
        <div className="head">
          <header className="masthead">
            <h1>emojic</h1>
          </header>
          <input
            className="input"
            type="text"
            autoComplete="off"
            autoFocus
            ref={inputRef}
            maxLength={maxLen || undefined}
            placeholder="type at least 3 characters…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
            {text.length}
            <span>/{maxLen}</span>
          </div>
        </div>
        <EmojiList
          items={emojiTop}
          active={shownEmoji}
          slots={emojiSlots}
          onPick={(e) => setOverride((o) => ({ ...o, emoji: e }))}
        />
        <Card
          text={text}
          emoji={displayEmoji}
          feeling={displayFeeling}
          palette={palette}
          onCopy={copyCard}
        />
        <KeyHints />
        <div className="feelings-col">
          <div className="feeling-slot">
            <FeelingBar
              feelings={feelingOptions}
              active={shownFeeling}
              palette={palette}
              count={feelingCount}
              ready={!tooShort && !!shownFeeling}
              hidden={tooShort || !shownFeeling}
              onPick={(f) => setOverride((o) => ({ ...o, feeling: f }))}
            />
            <p className={'warn' + (tooShort ? '' : ' is-hidden')}>
              text is too short — showing a default card
            </p>
          </div>
          <footer className="footer">
            <span>
              model updated <span>{formatDate(meta?.exported_at)}</span>
            </span>
            <span>made with ❤️ by Gilad</span>
            <span className="gh">
              <GitHubButton
                href="https://github.com/emojic-co/model"
                data-icon="octicon-star"
                data-show-count="true"
                aria-label="Star emojic-co/model on GitHub"
              >
                Star
              </GitHubButton>
            </span>
          </footer>
        </div>
      </div>
      <Toast toast={toast} />
    </main>
  )
}
