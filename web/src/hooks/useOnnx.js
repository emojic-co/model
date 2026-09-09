import { useCallback, useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { encode, decodeColorList, sigmoid } from '../model'
import { makeFlexRanker } from '../flexrank'

const BASE = import.meta.env.BASE_URL

export function useOnnx() {
  const [meta, setMeta] = useState(null)
  const [config, setConfig] = useState(null)
  const [ready, setReady] = useState(false)
  const sessionRef = useRef(null)
  const char2idxRef = useRef(null)
  const metaRef = useRef(null)
  const flexRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [m, c, fj] = await Promise.all([
          fetch(BASE + 'meta.json').then((r) => r.json()),
          fetch(BASE + 'config.json').then((r) => r.json()),
          fetch(BASE + 'flex.json').then((r) => r.json()),
        ])
        if (cancelled) return
        setMeta(m)
        setConfig(c)
        metaRef.current = m
        char2idxRef.current = new Map([...m.chars].map((ch, i) => [ch, i]))
        flexRef.current = makeFlexRanker(fj)
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
    const fr = flexRef.current
    const V = m.emojis.length
    const ids = encode(text, m, char2idxRef.current)
    const flexRaw = fr.flexRaw(text)
    const flexQ = Float32Array.from(fr.flexQ(text))
    const t0 = performance.now()
    const out = await sessionRef.current.run({
      input: new ort.Tensor('int64', ids, [1, m.max_text_len]),
      flex: new ort.Tensor('float32', flexRaw, [1, V, 10]),
      flex_q: new ort.Tensor('float32', flexQ, [1, 5]),
    })
    const ms = performance.now() - t0
    return {
      feeling: sigmoid(out.style_logits.data),
      emoji: sigmoid(out.emoji_logits.data),
      fusion: sigmoid(out.fusion_logits.data),
      keywordRank: fr.rank(text).map((row) => row[0]),
      palettes: decodeColorList(out.color.data),
      ms,
    }
  }, [])

  return { meta, config, ready, predict }
}
