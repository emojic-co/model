import notoIcons from "@iconify-json/noto/icons.json"
import notoChars from "@iconify-json/noto/chars.json"
import flagIcons from "@iconify-json/circle-flags/icons.json"

type IconSet = {
  icons: Record<string, { body: string; width?: number; height?: number }>
  aliases?: Record<string, { parent: string }>
  width?: number
  height?: number
}

const noto = notoIcons as unknown as IconSet
const chars = notoChars as unknown as Record<string, string>
const flags = flagIcons as unknown as IconSet

export type EmojiSvg = { body: string; width: number; height: number }

const KEYCAP_NAMES: Record<number, string> = {
  0x30: "keycap-0",
  0x31: "keycap-1",
  0x32: "keycap-2",
  0x33: "keycap-3",
  0x34: "keycap-4",
  0x35: "keycap-5",
  0x36: "keycap-6",
  0x37: "keycap-7",
  0x38: "keycap-8",
  0x39: "keycap-9",
  0x23: "keycap-pound",
  0x2a: "keycap-asterisk",
}

function codePoints(emoji: string): number[] {
  return [...emoji].map((c) => c.codePointAt(0) as number).filter((c) => c !== 0xfe0f)
}

function notoIcon(name: string): IconSet["icons"][string] | null {
  const direct = noto.icons[name]
  if (direct) return direct
  const parent = noto.aliases?.[name]?.parent
  return parent ? noto.icons[parent] ?? null : null
}

function fromNoto(emoji: string): EmojiSvg | null {
  const key = codePoints(emoji).map((c) => c.toString(16)).join("-")
  const name = chars[key]
  const icon = name ? notoIcon(name) : null
  if (!icon) return null
  return { body: icon.body, width: icon.width ?? noto.width ?? 128, height: icon.height ?? noto.height ?? 128 }
}

function fromFlag(emoji: string): EmojiSvg | null {
  const cps = codePoints(emoji)
  if (cps.length !== 2 || !cps.every((c) => c >= 0x1f1e6 && c <= 0x1f1ff)) return null
  const code = cps.map((c) => String.fromCharCode(c - 0x1f1e6 + 97)).join("")
  const icon = flags.icons[code]
  if (!icon) return null
  return { body: icon.body, width: icon.width ?? flags.width ?? 512, height: icon.height ?? flags.height ?? 512 }
}

function fromKeycap(emoji: string): EmojiSvg | null {
  const cps = codePoints(emoji)
  if (cps.length !== 2 || cps[1] !== 0x20e3) return null
  const name = KEYCAP_NAMES[cps[0]]
  const icon = name ? notoIcon(name) : null
  if (!icon) return null
  return { body: icon.body, width: icon.width ?? noto.width ?? 128, height: icon.height ?? noto.height ?? 128 }
}

export function resolveEmojiSvg(emoji: string): EmojiSvg | null {
  if (!emoji) return null
  return fromNoto(emoji) ?? fromFlag(emoji) ?? fromKeycap(emoji)
}
