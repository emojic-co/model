import { LABELS_JSON } from "../../files.ts"
import { stripVS } from "../analysis/cldr-baseline.ts"
import { buildFlexRanker } from "./flexrank.ts"
import { writeFileAtomic } from "./io.ts"

const TEXTS = [
  "Cupcakes are in the break room",
  "I found a ladybug in my book",
  "pizza time with friends tonight",
  "feeling anxious about the meeting",
  "the dog is running fast in the park",
  "not good at all",
  "Woof!",
  "xyzzy qwerty",
]

const vocabOrig = JSON.parse(await Bun.file(LABELS_JSON).text()).emojis as string[]
const vocab = new Map(vocabOrig.map((e) => [stripVS(e), e]))
const { rank } = await buildFlexRanker(32)
const cases = TEXTS.map((text) => {
  const { flexsearch, flexq } = rank(text, vocab)
  return { text, flexsearch, flexq }
})
await writeFileAtomic(
  "web/src/flexrank.fixture.json",
  JSON.stringify({ vocab: vocabOrig, cases }, null, 2) + "\n",
)
console.log(`wrote web/src/flexrank.fixture.json (${cases.length} cases)`)
