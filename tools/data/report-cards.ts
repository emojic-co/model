import { cardHtml, page, type CardData } from "./preview-card.ts"

const EXTRA_CSS = `
body { padding: 0.5em; }
.card { max-width: none; }
`

if (import.meta.main) {
  const cards: CardData[] = JSON.parse(await Bun.stdin.text())
  const docs = await Promise.all(
    cards.map((c) => page({ title: "sample card", extraCss: EXTRA_CSS, body: cardHtml(c) })),
  )
  console.log(JSON.stringify(docs))
}
