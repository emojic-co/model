import { readFile } from "node:fs/promises"

import { EKMAN_FEELINGS } from "./config"
import { writeFileAtomic } from "./io.ts"

const FILES = ["./eval.jsonl", "./train.jsonl"]

const FOLD: Record<string, string[]> = {
  Angry: [
    "Angry", "Annoyed", "Frustrated", "Irritated", "Furious", "Livid",
    "Enraged", "Irate", "Aggravated", "Exasperated", "Indignant", "Outraged",
    "Resentful", "Fed-up", "Fed Up", "Fed-Up", "Grumpy", "Grouchy", "Mad",
    "Infuriated", "Incensed", "Fuming", "Seething", "Bitter", "Pissed",
    "Peeved", "Miffed", "Irritable", "Agitated", "Cross", "Snappy", "Snarky",
    "Sassy", "Sulky", "Huffy", "Impatient", "Rageful", "Raging", "Cranky",
    "Disgruntled", "Displeased", "Bothered", "Hostile", "Combative",
    "Confrontational", "Belligerent", "Vexed", "Riled",
  ],
  Disgusted: [
    "Disgusted", "Repulsed", "Revolted", "Sickened", "Appalled",
    "Grossed-out", "Grossed out", "Repelled", "Disgust", "Revulsion",
    "Squeamish", "Nauseated",
  ],
  Afraid: [
    "Afraid", "Scared", "Frightened", "Terrified", "Fearful", "Petrified",
    "Anxious", "Nervous", "Worried", "Concerned", "Uneasy", "Apprehensive",
    "Panicked", "Panicky", "Panicking", "Alarmed", "Spooked", "Jittery",
    "Dread", "Dreading", "Dreadful", "Distressed", "Frantic", "Stressed",
    "Overwhelmed", "Pressured", "Tense", "Fretful", "Unnerved", "Rattled",
    "Shaken", "Nervy", "Edgy", "Fright",
  ],
  Happy: [
    "Happy", "Joyful", "Joyous", "Overjoyed", "Delighted", "Delight",
    "Pleased", "Glad", "Cheerful", "Cheery", "Content", "Contented", "Merry",
    "Jubilant", "Gleeful", "Chipper", "Upbeat", "Buoyant", "Blissful",
    "Elated", "Elation", "Ecstatic", "Euphoric", "Thrilled", "Excited",
    "Exhilarated", "Enthusiastic", "Eager", "Energized", "Pumped", "Hyped",
    "Stoked", "Psyched", "Relieved", "Reassured", "Comforted", "Grateful",
    "Thankful", "Appreciative", "Hopeful", "Optimistic", "Encouraged",
    "Heartened", "Proud", "Accomplished", "Triumphant", "Satisfied",
    "Fulfilled", "Gratified", "Amused", "Entertained", "Tickled", "Playful",
    "Charmed", "Enchanted", "Festive", "Celebratory", "Jolly", "Giddy",
    "Lighthearted", "Carefree", "Chuffed",
  ],
  Love: [
    "Love", "Loving", "Adoring", "Adoration", "Affectionate", "Fond", "Tender",
    "Smitten", "Enamored", "Infatuated", "Loved", "Cherished", "Devoted",
    "Lovestruck", "Lovesick", "Romantic", "Caring", "Warm",
  ],
  Sad: [
    "Sad", "Unhappy", "Sorrowful", "Mournful", "Miserable", "Depressed",
    "Down", "Downcast", "Downhearted", "Blue", "Gloomy", "Glum", "Melancholy",
    "Melancholic", "Despondent", "Dejected", "Disheartened", "Discouraged",
    "Demoralized", "Crestfallen", "Forlorn", "Wretched", "Grief", "Grieving",
    "Grief-stricken", "Bereaved", "Bereft", "Heartbroken", "Heartsick",
    "Devastated", "Crushed", "Shattered", "Anguished", "Distraught",
    "Inconsolable", "Disappointed", "Let down", "Let-down", "Letdown",
    "Disillusioned", "Lonely", "Lonesome", "Isolated", "Longing", "Yearning",
    "Pining", "Homesick", "Hurt", "Wounded", "Aching", "Hollow", "Empty",
    "Numb", "Regretful", "Remorseful", "Rueful", "Regret", "Guilty",
    "Ashamed", "Embarrassed", "Humiliated", "Mortified", "Sheepish",
    "Self-conscious", "Hopeless", "Despairing", "Despair", "Defeated",
    "Somber", "Woeful", "Tearful", "Teary", "Weepy", "Bummed", "Gutted",
    "Deflated", "Dispirited", "Heavyhearted", "Sorry", "Grim", "Bleak",
  ],
  Surprised: [
    "Surprised", "Shocked", "Astonished", "Astounded", "Amazed", "Amazement",
    "Stunned", "Startled", "Dumbfounded", "Dumbstruck", "Flabbergasted",
    "Speechless", "Stupefied", "Gobsmacked", "Thunderstruck", "Taken aback",
    "Taken-aback", "Aghast", "Floored", "Blindsided", "Incredulous",
    "Disbelieving", "Disbelief", "Wide-eyed", "Awed", "Awestruck",
    "Wonderstruck",
  ],
  Neutral: [
    "Neutral", "Neutrality", "Indifferent", "Unbothered", "Nonchalant",
    "Unfazed", "Unconcerned", "Unmoved", "Blasé", "Blase", "Apathetic",
    "Uninterested", "Disinterested", "Matter-of-fact", "Matter of fact",
    "Casual", "Detached", "Impassive", "Dispassionate", "Aloof",
    "Noncommittal", "Meh", "Emotionless", "Unemotional", "Unaffected",
    "Impartial", "Even-keeled", "Blank",
  ],
}

const MAP = new Map<string, string>()
for (const [bucket, words] of Object.entries(FOLD)) {
  for (const w of words) {
    if (MAP.has(w)) throw new Error(`fold collision: "${w}"`)
    MAP.set(w, bucket)
  }
}
if (!Object.keys(FOLD).every((b) => (EKMAN_FEELINGS as readonly string[]).includes(b))) {
  throw new Error("FOLD bucket not in EKMAN_FEELINGS")
}

const ALLOWED = new Set("·abcdefghijklmnopqrstuvwxyz!?:()@$%&* ")

function normalize(text: string): string {
  let t = text.replace(/\s+/g, " ").trim().toLowerCase()
  t = t.replace(/(.)\1{2,}/g, "$1$1")
  return [...t].filter((c) => ALLOWED.has(c)).join("")
}

type Row = {
  text: string
  feeling: string
  emoji: string
  bg: unknown
  fg: unknown
}

function validRow(r: unknown): r is Row {
  if (!r || typeof r !== "object") return false
  const o = r as Record<string, unknown>
  return (
    typeof o.text === "string" &&
    typeof o.feeling === "string" &&
    typeof o.emoji === "string" &&
    Array.isArray(o.bg) &&
    o.bg.length === 2 &&
    typeof o.bg[0] === "string" &&
    typeof o.bg[1] === "string" &&
    typeof o.fg === "string"
  )
}

if (import.meta.main) {
  const seen = new Set<string>()
  const dropped = new Map<string, number>()
  const bucketCounts = new Map<string, number>()

  for (const path of FILES) {
    const src = await readFile(path, "utf8")
    const out: string[] = []
    let total = 0
    let bad = 0
    let unmapped = 0
    let dupes = 0
    let empty = 0

    for (const line of src.split("\n")) {
      if (!line.trim()) continue
      total++
      let parsed: unknown
      try {
        parsed = JSON.parse(line)
      } catch {
        bad++
        continue
      }
      if (!validRow(parsed)) {
        bad++
        continue
      }
      const row = parsed as Row
      const bucket = MAP.get(row.feeling)
      if (!bucket) {
        unmapped++
        dropped.set(row.feeling, (dropped.get(row.feeling) ?? 0) + 1)
        continue
      }
      const key = normalize(row.text)
      if (!key) {
        empty++
        continue
      }
      if (seen.has(key)) {
        dupes++
        continue
      }
      seen.add(key)
      bucketCounts.set(bucket, (bucketCounts.get(bucket) ?? 0) + 1)
      out.push(
        JSON.stringify({
          text: row.text,
          feeling: bucket,
          emoji: row.emoji,
          bg: row.bg,
          fg: row.fg,
        }),
      )
    }

    await writeFileAtomic(path, out.join("\n") + "\n")
    console.log(
      `${path}: ${total} -> ${out.length} kept  ` +
        `(dropped: ${unmapped} unmapped, ${dupes} dup-text, ${empty} empty-norm, ${bad} malformed)`,
    )
  }

  console.log("\nnew feeling distribution (train + eval combined):")
  for (const b of EKMAN_FEELINGS) {
    const c = bucketCounts.get(b) ?? 0
    console.log(`  ${b.padEnd(10)} ${String(c).padStart(7)}`)
  }

  const topDropped = [...dropped.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25)
  const totalDropped = [...dropped.values()].reduce((a, b) => a + b, 0)
  console.log(
    `\ndropped ${totalDropped} rows across ${dropped.size} unmapped feelings; top 25:`,
  )
  for (const [f, c] of topDropped) console.log(`  ${String(c).padStart(6)}  ${f}`)

  process.exit(0)
}
