export const LANGS = ["en", "he"] as const

export type Lang = (typeof LANGS)[number]

export const LANG_SET: ReadonlySet<string> = new Set(LANGS)
