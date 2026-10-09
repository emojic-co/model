import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize, fixContrast, sigmoid } from './model'
import { topFeelings, DEFAULT_COLORS } from './feelings'
import { cycle, textToHash, hashToText } from './nav'
import { ensureScriptFontsLoaded, scriptForLang, langForText } from './scriptFonts'
import GitHubButton from 'react-github-btn'
import { Card } from './components/Card'
import { FeelingBar } from './components/FeelingBar'
import { ColorBar } from './components/ColorBar'
import { EmojiList } from './components/EmojiList'
import { KeyHints } from './components/KeyHints'
import { ShareBar } from './components/ShareBar'
import { mp4Supported, useCardExport, useCardImage } from './hooks/useCardImage'
import { Toast } from './components/Toast'
import { EmojiRain } from './components/EmojiRain'
import { isEasterEgg } from './easterEgg'
import { useMediaQuery } from './hooks/useMediaQuery'
import { useI18n, LANGS, LANG_NAMES } from './i18n'

const MIN_CHARS = 3
// Idle time after the last keystroke before the card (and prediction) update.
const DEBOUNCE_MS = 600

export function pickEmojiList(scores, meta, slots) {
  if (!scores || !meta) return []
  const arr = scores.emoji
  if (!arr) return []
  return [...arr.keys()]
    .sort((a, b) => arr[b] - arr[a])
    .slice(0, slots)
    .map((idx) => ({ emoji: meta.emojis[idx], p: sigmoid([arr[idx]])[0] }))
}

function formatMs(ms) {
  if (ms == null) return '—'
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`
}

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
  const { meta, config, ready, predict } = useOnnx()
  const { lang: uiLang, setLang: setUiLang, t } = useI18n()
  const mobile = useMediaQuery('(max-width: 56.25em)')
  const emojiSlots = mobile ? 9 : 10
  const feelingCount = mobile ? 4 : 5
  const colorCount = mobile ? 4 : 5
  const [text, setText] = useState(() => hashToText(window.location.hash))
  const [modelText, setModelText] = useState('')
  const [cardText, setCardText] = useState(text)
  const [lang, setLang] = useState('en')
  const [scores, setScores] = useState(null)
  const [pending, setPending] = useState(false)
  const [override, setOverride] = useState({ emoji: null, feeling: null, color: 0 })
  const [toast, setToast] = useState({ msg: '', n: 0 })
  const showToast = useCallback((msg) => setToast((s) => ({ msg, n: s.n + 1 })), [])
  const [eggTrigger, setEggTrigger] = useState(0)
  useEffect(() => {
    if (isEasterEgg(text)) setEggTrigger((n) => n + 1)
  }, [text])
  const seq = useRef(0)
  const inputRef = useRef(null)
  const cardRef = useRef(null)

  const char2idx = useMemo(
    () => (meta ? new Map([...meta.chars].map((c, i) => [c, i])) : null),
    [meta],
  )

  useEffect(() => {
    const el = cardRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      document.documentElement.style.setProperty('--card-h', `${el.getBoundingClientRect().height}px`)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const hash = textToHash(text)
    if (window.location.hash !== hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search + hash)
    }
  }, [text])

  useEffect(() => {
    if (!text) {
      setCardText('')
      return
    }
    const timer = setTimeout(() => setCardText(text), DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [text])

  useEffect(() => {
    if (!ready || !char2idx) return
    if (text.trim().length < MIN_CHARS) {
      seq.current++
      setPending(false)
      setScores(null)
      setModelText('')
      setLang('en')
      setOverride({ emoji: null, feeling: null, color: 0 })
      return
    }
    const mine = ++seq.current
    const timer = setTimeout(async () => {
      setPending(true)
      const detected = langForText(text)
      setLang(detected)
      ensureScriptFontsLoaded(scriptForLang(detected))
      setModelText(text)
      if (normalize(text, char2idx).length < MIN_CHARS) {
        setScores(null)
        setOverride({ emoji: null, feeling: null, color: 0 })
        setPending(false)
        return
      }
      const logits = await predict(text)
      if (mine !== seq.current) return
      setScores(logits)
      setOverride({ emoji: null, feeling: null, color: 0 })
      setPending(false)
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [text, ready, char2idx, predict])

  const emojiTop = useMemo(
    () => pickEmojiList(scores, meta, emojiSlots),
    [scores, meta, emojiSlots],
  )
  const predictedEmoji = emojiTop[0]?.emoji ?? null
  const predictedFeeling = scores ? meta.styles[argmax(scores.feeling)] : null
  const shownEmoji = override.emoji ?? predictedEmoji
  const shownFeeling = override.feeling ?? predictedFeeling
  const feelingOptions = useMemo(
    () => topFeelings(scores?.feeling, meta?.styles ?? [], shownFeeling, feelingCount),
    [scores, meta, shownFeeling, feelingCount],
  )
  const emojiList = useMemo(() => emojiTop.map((x) => x.emoji), [emojiTop])

  const rawPalettes = useMemo(() => scores?.palettes ?? [DEFAULT_COLORS], [scores])
  const palettes = useMemo(() => rawPalettes.map((p) => fixContrast(p)), [rawPalettes])
  const colors = palettes[override.color] ?? palettes[0]

  const cardData =
    shownEmoji && shownFeeling
      ? {
          text,
          emoji: shownEmoji,
          feeling: shownFeeling,
          lang,
          colors,
        }
      : null
  const copyCard = useCardImage(cardData, showToast, t)
  const exporter = useCardExport(cardData, showToast, t)
  const shareProps = {
    onJpg: copyCard,
    onMp4: mp4Supported() ? exporter.saveMp4 : undefined,
    exporting: exporter.busy,
    progress: exporter.progress,
    onCancel: exporter.cancel,
  }
  const exporting = !!exporter.busy
  const exportingRef = useRef(false)
  exportingRef.current = exporting

  useEffect(() => {
    const onKey = (e) => {
      if (exportingRef.current) return
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
      if (e.altKey) {
        if (e.shiftKey || e.metaKey || palettes.length <= 1) return
        e.preventDefault()
        setOverride((o) => ({ ...o, color: (o.color + dir + colorCount) % colorCount }))
        return
      }
      if (e.shiftKey || e.metaKey || !emojiList.length) return
      e.preventDefault()
      setOverride((o) => ({ ...o, emoji: cycle(emojiList, o.emoji ?? predictedEmoji, dir) }))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    emojiList,
    feelingOptions,
    predictedEmoji,
    predictedFeeling,
    copyCard,
    palettes,
    colorCount,
  ])

  const maxLen = config?.max_text_len ?? 0
  const tooShort =
    ready && char2idx ? normalize(modelText, char2idx).length < MIN_CHARS : false
  const displayEmoji = shownEmoji ?? '🙂'
  const displayFeeling = shownFeeling ?? 'Neutral'

  return (
    <main data-exporting={exporting || undefined}>
      <div className="stage">
        <div className="head">
          <div className="head-top">
            <header className="masthead">
              <h1>
                emojify<span className="tld">.ing</span>
              </h1>
            </header>
          </div>
          <input
            className="input"
            type="text"
            dir="auto"
            autoComplete="off"
            autoFocus
            ref={inputRef}
            maxLength={maxLen || undefined}
            placeholder={t('input.placeholder')}
            value={text}
            disabled={exporting}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="input-meta">
            <p className={'warn' + (text.trim() && tooShort ? '' : ' is-hidden')}>
              {t('warn.tooShort')}
            </p>
            <span className={'timing' + (scores ? '' : ' is-hidden')}>
              {t('timing.ran', { ms: formatMs(scores?.ms) })}
            </span>
            <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
              {text.length}
              <span>/{maxLen}</span>
            </div>
          </div>
        </div>
        <EmojiList
          items={scores ? emojiTop : null}
          active={shownEmoji}
          slots={emojiSlots}
          disabled={exporting}
          onPick={(e) => setOverride((o) => ({ ...o, emoji: e }))}
        />
        <Card
          ref={cardRef}
          text={cardText}
          emoji={displayEmoji}
          feeling={displayFeeling}
          lang={lang}
          colors={colors}
          loading={pending}
          exporting={exporter.busy}
        />
        <ShareBar {...shareProps} className="share-bar--below" />
        <KeyHints>
          <ShareBar {...shareProps} />
        </KeyHints>
        <div className="feelings-col" inert={exporting || undefined}>
          <ColorBar
            palettes={palettes}
            active={override.color}
            count={colorCount}
            ready={!tooShort && !!scores}
            onPick={(i) => setOverride((o) => ({ ...o, color: i }))}
          />
          <FeelingBar
            feelings={feelingOptions}
            active={shownFeeling}
            count={feelingCount}
            ready={!tooShort && !!shownFeeling}
            onPick={(f) => setOverride((o) => ({ ...o, feeling: f }))}
          />
        </div>
        <footer className="footer">
          <div className="footer-col">
            <a
              className="play-badge"
              href="https://play.google.com/store/apps/details?id=ing.emojify"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src={`/google-play-${uiLang === 'he' ? 'he' : 'en'}.png`}
                alt={t('footer.android')}
                width="646"
                height="250"
              />
            </a>
          </div>
          <div className="footer-col">
            <span>
              {t('footer.modelUpdated', { date: formatDate(meta?.exported_at) })}
            </span>
            <span>
              <a
                href="https://github.com/emojic-co/model/blob/main/ABOUT.md"
                target="_blank"
                rel="noopener noreferrer"
              >
                {t('footer.about')}
              </a>
            </span>
            <span>
              <a href="/emoji-coverage.html" target="_blank" rel="noopener noreferrer">
                {t('footer.coverage')}
              </a>
            </span>
            <span>
              <a href="/style-preview.html" target="_blank" rel="noopener noreferrer">
                {t('footer.stylePreview')}
              </a>
            </span>
          </div>
          <div className="footer-col">
            <span className="lang-select">
              <label>
                {t('footer.language')}{' '}
                <select value={uiLang} onChange={(e) => setUiLang(e.target.value)}>
                  {LANGS.map((l) => (
                    <option key={l} value={l}>
                      {LANG_NAMES[l]}
                    </option>
                  ))}
                </select>
              </label>
            </span>
            <span>{t('footer.madeBy')}</span>
            <span>
              {t('footer.patternsBy')}{' '}
              <a href="https://heropatterns.com/" target="_blank" rel="noopener noreferrer">
                Hero Patterns
              </a>
            </span>
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
          </div>
        </footer>
      </div>
      <Toast toast={toast} />
      <EmojiRain trigger={eggTrigger} />
    </main>
  )
}
