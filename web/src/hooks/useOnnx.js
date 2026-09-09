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
    const ids = encode(text, m, char2idxRef.current)
    const flexTf = Float32Array.from(fr.tfVec(text))
    const t0 = performance.now()
    const out = await sessionRef.current.run({
      input: new ort.Tensor('int64', ids, [1, m.max_text_len]),
      flex_tf: new ort.Tensor('float32', flexTf, [1, m.flex_n]),
    })
    const ms = performance.now() - t0
    return {
      feeling: sigmoid(out.style_logits.data),
      emoji: sigmoid(out.emoji_logits.data),
      kw: sigmoid(out.kw_logits.data),
      fusion: sigmoid(out.fusion_logits.data),
      palettes: decodeColorList(out.color.data),
      ms,
    }
  }, [])

  return { meta, config, ready, predict }
}
