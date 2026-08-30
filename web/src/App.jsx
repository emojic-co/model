import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize } from './model'
import { Card } from './components/Card'
import { EmojiList } from './components/EmojiList'
import { useCardImage } from './hooks/useCardImage'
import { Toast } from './components/Toast'

const MIN_CHARS = 3
const DEBOUNCE_MS = 100

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
  const [text, setText] = useState('')
  const [scores, setScores] = useState(null)
  const [override, setOverride] = useState({ emoji: null, feeling: null })
  const [toast, setToast] = useState({ msg: '', n: 0 })
  const showToast = useCallback((msg) => setToast((s) => ({ msg, n: s.n + 1 })), [])
  const seq = useRef(0)

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

  const maxLen = config?.max_text_len ?? 0

  return (
    <main>
      <h1>emojic</h1>
      <div className="stage">
        <EmojiList
          emojiScores={emojiScores}
          emojis={meta?.emojis ?? []}
          onPick={(e) => setOverride((o) => ({ ...o, emoji: e }))}
        />
        <div className="card-col">
          <input
            className="input"
            type="text"
            autoComplete="off"
            autoFocus
            maxLength={maxLen || undefined}
            placeholder="type at least 3 characters…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                copyCard()
              }
            }}
          />
          <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
            {text.length}
            <span>/{maxLen}</span>
          </div>
          <Card
            text={text}
            emoji={shownEmoji ?? '🙂'}
            feeling={shownFeeling}
            feelings={meta?.feelings ?? []}
            palette={palette}
            onPickFeeling={(f) => setOverride((o) => ({ ...o, feeling: f }))}
            onCopy={copyCard}
          />
        </div>
      </div>
      <footer className="footer">
        model updated <span>{formatDate(meta?.exported_at)}</span>
      </footer>
      <Toast toast={toast} />
    </main>
  )
}
