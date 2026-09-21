import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import { stringify } from "yaml"

import { ANDROID_ASSETS_DIR, STYLE_YML } from "../../files.ts"
import { CLUSTERS, FEELINGS } from "../../web/src/feelings.js"
import { MAX_OPACITY, PATTERN_FNS, REFERENCE_PX } from "../../web/src/patterns.js"
import { RATIOS } from "../../web/src/hooks/useCardImage.js"

function primaryFamily(fontStack: string): string {
  return fontStack.match(/^"([^"]+)"/)?.[1] ?? fontStack.split(",")[0].trim()
}

function patternSvg(feeling: string): { svg: string; widthRatio: number; heightRatio: number } {
  const entry = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  const image = entry.pattern()
  const encoded = image.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)?.[2]
  if (!encoded) throw new Error(`could not decode pattern svg for ${feeling}`)
  const svg = decodeURIComponent(encoded)
  const tag = svg.match(/<svg[^>]*>/)?.[0]
  const w = Number(tag?.match(/width="([\d.]+)"/)?.[1] ?? REFERENCE_PX)
  const h = Number(tag?.match(/height="([\d.]+)"/)?.[1] ?? REFERENCE_PX)
  return {
    svg,
    widthRatio: (w / REFERENCE_PX) * entry.scale,
    heightRatio: (h / REFERENCE_PX) * entry.scale,
  }
}

export function buildStyleFile() {
  const styles = Object.fromEntries(
    Object.entries(FEELINGS).map(([name, f]) => {
      const cluster = CLUSTERS[f.cluster]
      const st = f.style ?? {}
      const letterSpacingEm = st.letterSpacing != null ? Number.parseFloat(st.letterSpacing) : null
      return [
        name,
        {
          cluster: f.cluster,
          font: primaryFamily(f.font),
          fontWeight: st.fontWeight ?? 400,
          italic: st.fontStyle === "italic",
          uppercase: st.textTransform === "uppercase",
          letterSpacingEm,
          opacity: st.opacity ?? 1,
          entrance: f.entrance ?? cluster.entrance,
          emoji: f.emoji ?? cluster.emoji,
          entranceMs: f.dur?.entrance ?? 650,
          emojiMs: f.dur?.emoji ?? 2400,
          pattern: patternSvg(name),
        },
      ]
    }),
  )

  return {
    exportedAt: new Date().toISOString(),
    global: {
      referencePx: REFERENCE_PX,
      maxPatternOpacity: MAX_OPACITY,
      ...RATIOS,
    },
    styles,
  }
}

function writeStyleFile(path: string, contents: string) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, contents, "utf-8")
}

function main() {
  const yaml = stringify(buildStyleFile())
  writeStyleFile(STYLE_YML, yaml)
  writeStyleFile(`${ANDROID_ASSETS_DIR}/style.yml`, yaml)
  console.log(`wrote ${STYLE_YML} and ${ANDROID_ASSETS_DIR}/style.yml`)
}

if (import.meta.main) main()
