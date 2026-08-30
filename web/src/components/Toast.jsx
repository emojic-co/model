import { useEffect, useState } from 'react'

export function Toast({ toast }) {
  const [visible, setVisible] = useState(false)
  const [msg, setMsg] = useState('')

  useEffect(() => {
    if (!toast.n) return
    setMsg(toast.msg)
    setVisible(true)
    const t = setTimeout(() => setVisible(false), 1600)
    return () => clearTimeout(t)
  }, [toast])

  return (
    <div className={'toast' + (visible ? ' show' : '')} role="status" aria-live="polite">
      {msg}
    </div>
  )
}
