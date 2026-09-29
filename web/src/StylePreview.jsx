import { useEffect, useState } from 'react'
import { parse } from 'yaml'
import { useFitText } from './hooks/useFitText'
import { AnimatedEmoji } from './components/AnimatedEmoji'
import { CardShimmer } from './components/CardShimmer'
import { CharText } from './components/CharText'
import { contrastRatio, fixContrast, patternTint, hexToOklab, toCssOklab, BLACK, WHITE } from './model'

function watermarkInk(bg) {
  return contrastRatio(BLACK, bg) >= contrastRatio(WHITE, bg) ? '#000000' : '#ffffff'
}

function PreviewCard({ name, lang, entry, sample, globalSettings, patterns, anim, shimmer, tick }) {
  const [own, setOwn] = useState(0)
  const colors = fixContrast({
    bg1: hexToOklab(sample.colors.bg1),
    bg2: hexToOklab(sample.colors.bg2),
    text_color: hexToOklab(sample.colors.text_color),
  })
  const displayText = sample.text.trim() ? sample.text : "What's on your mind?"
  const textRef = useFitText(displayText, {
    min: globalSettings.textMinRatio * 100,
    max: globalSettings.textMaxRatio * 100,
    key: name,
  })
  const tint = toCssOklab(patternTint(colors.bg1, colors.bg2))
  const maskUrl = `url("data:image/svg+xml,${encodeURIComponent(patterns[entry.pattern.name])}")`

  return (
    <figure className="style-preview-card">
      <div
        className="card"
        data-cluster={entry.cluster}
        data-emoji={entry.emoji}
        onClick={() => setOwn((n) => n + 1)}
        dir={lang === 'he' ? 'rtl' : 'ltr'}
        style={{
          background: `linear-gradient(135deg, ${toCssOklab(colors.bg1)}, ${toCssOklab(colors.bg2)})`,
          color: toCssOklab(colors.text_color),
          fontFamily: `"${entry.font}", sans-serif`,
          '--emoji-dur': `${entry.emojiMs}ms`,
          cursor: 'pointer',
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            backgroundColor: tint,
            WebkitMaskImage: maskUrl,
            maskImage: maskUrl,
            WebkitMaskRepeat: 'repeat',
            maskRepeat: 'repeat',
            WebkitMaskSize: `${entry.pattern.widthRatio * 100}% ${entry.pattern.heightRatio * 100}%`,
            maskSize: `${entry.pattern.widthRatio * 100}% ${entry.pattern.heightRatio * 100}%`,
            opacity: globalSettings.maxPatternOpacity,
          }}
        />
        <CardShimmer
          cluster={entry.cluster}
          feeling={name}
          motif={entry.entrance}
          text={displayText}
          replayKey={`${tick}-${own}`}
          anim={shimmer}
          textAnim={anim}
        />
        <AnimatedEmoji emoji={sample.emoji} />
        <div className="card-text-box" ref={textRef} style={{ position: 'relative' }}>
          <CharText
            className="card-text"
            dir="auto"
            style={{
              fontWeight: entry.fontWeight,
              fontStyle: entry.italic ? 'italic' : 'normal',
              textTransform: entry.uppercase ? 'uppercase' : 'none',
              letterSpacing: entry.letterSpacingEm != null ? `${entry.letterSpacingEm}em` : undefined,
              opacity: entry.opacity,
            }}
            text={displayText}
            anim={anim}
            motif={entry.entrance}
            feeling={name}
            replayKey={`${tick}-${own}`}
          />
        </div>
        <span
          className="card-watermark"
          aria-hidden="true"
          style={{ position: 'relative', color: watermarkInk(colors.bg2) }}
        >
          emojify.ing
        </span>
      </div>
      <figcaption>
        {name} <span className="style-preview-lang">{lang}</span>
        <span className="style-preview-motif"> · {entry.entrance} · {entry.cluster} shimmer</span>
      </figcaption>
    </figure>
  )
}

const BASE = import.meta.env.BASE_URL

export function StylePreview() {
  const [state, setState] = useState({ status: 'loading' })
  const [tick, setTick] = useState(0)
  const [auto, setAuto] = useState(true)

  useEffect(() => {
    if (!auto) return
    const t = setInterval(() => setTick((n) => n + 1), 9000)
    return () => clearInterval(t)
  }, [auto])

  useEffect(() => {
    Promise.all([
      fetch(BASE + 'style.yml').then((r) => r.text()),
      fetch(BASE + 'style-samples.json').then((r) => r.json()),
    ])
      .then(([yamlText, samples]) => {
        setState({ status: 'ready', style: parse(yamlText), samples })
      })
      .catch((err) => setState({ status: 'error', error: err }))
  }, [])

  if (state.status === 'loading') return <p className="style-preview-status">loading style.yml…</p>
  if (state.status === 'error') {
    return <p className="style-preview-status">failed to load style.yml: {String(state.error)}</p>
  }

  const { style, samples } = state
  const names = Object.keys(style.styles)

  return (
    <main className="style-preview">
      <header>
        <h1>Style preview</h1>
        <p>
          Every style rendered live from <code>style.yml</code> (exported{' '}
          {new Date(style.exportedAt).toLocaleString()}) — the same file the Android app syncs and renders from,
          so this page shows exactly what both apps ship.
        </p>
        <div className="style-preview-controls">
          <button type="button" onClick={() => setTick((n) => n + 1)}>
            Replay all
          </button>
          <label>
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> replay every 9s
          </label>
          <span>Click a card to replay it.</span>
        </div>
      </header>
      <div className="style-preview-grid">
        {names.flatMap((name) =>
          ['en', 'he'].map((lang) => (
            <PreviewCard
              key={`${name}-${lang}`}
              name={name}
              lang={lang}
              entry={style.styles[name]}
              sample={samples[name][lang]}
              globalSettings={style.global}
              patterns={style.patterns}
              anim={style.textAnimations}
              shimmer={style.shimmer}
              tick={tick}
            />
          )),
        )}
      </div>
    </main>
  )
}
