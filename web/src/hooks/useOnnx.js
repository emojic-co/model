import { useCallback, useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { encode } from '../model'

const BASE = import.meta.env.BASE_URL

export function useOnnx() {
  const [meta, setMeta] = useState(null)
  const [config, setConfig] = useState(null)
  const [palette, setPalette] = useState(null)
  const [ready, setReady] = useState(false)
  const sessionRef = useRef(null)
  const char2idxRef = useRef(null)
  const metaRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [m, c, p] = await Promise.all([
          fetch(BASE + 'meta.json').then((r) => r.json()),
          fetch(BASE + 'config.json').then((r) => r.json()),
          fetch(BASE + 'palette.json').then((r) => r.json()),
        ])
        if (cancelled) return
        setMeta(m)
        setConfig(c)
        setPalette(p)
        metaRef.current = m
        char2idxRef.current = new Map([...m.chars].map((ch, i) => [ch, i]))
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
    const tensor = new ort.Tensor('int64', ids, [1, m.max_text_len])
    const out = await sessionRef.current.run({ input: tensor })
    return {
      feeling: Array.from(out.feeling_logits.data),
      emoji: Array.from(out.emoji_logits.data),
    }
  }, [])

  return { meta, config, palette, ready, predict }
}
