import { useCallback, useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { encode, decodeColorList, sigmoid } from '../model'
import { makeKeywordPredictor } from '../keywords'

const BASE = import.meta.env.BASE_URL

export function useOnnx() {
  const [meta, setMeta] = useState(null)
  const [config, setConfig] = useState(null)
  const [ready, setReady] = useState(false)
  const sessionRef = useRef(null)
  const char2idxRef = useRef(null)
  const metaRef = useRef(null)
  const kwRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [m, c, kwproj] = await Promise.all([
          fetch(BASE + 'meta.json').then((r) => r.json()),
          fetch(BASE + 'config.json').then((r) => r.json()),
          fetch(BASE + 'kwproj.json').then((r) => r.json()),
        ])
        if (cancelled) return
        setMeta(m)
        setConfig(c)
        metaRef.current = m
        char2idxRef.current = new Map([...m.chars].map((ch, i) => [ch, i]))
        kwRef.current = makeKeywordPredictor(kwproj, m.emojis.length)
        ort.env.wasm.numThreads = 1
        const session = await ort.InferenceSession.create(BASE + 'model.onnx')
        if (cancelled) return
        sessionRef.current = session
        setReady(true)
      } catch (err) {
        console.error('emojic: model load failed', err)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const predict = useCallback(async (text) => {
    const m = metaRef.current
    const ids = encode(text, m, char2idxRef.current)
    const t0 = performance.now()
    const out = await sessionRef.current.run({
      input: new ort.Tensor('int64', ids, [1, m.max_text_len]),
    })
    const ms = performance.now() - t0
    const emojiSigmoid = sigmoid(out.emoji_logits.data)
    const kwArr = kwRef.current.predict(text)
    return {
      feeling: sigmoid(out.style_logits.data),
      emoji: emojiSigmoid,
      kw: kwArr,
      palettes: decodeColorList(out.color.data),
      ms,
    }
  }, [])

  return { meta, config, ready, predict }
}
