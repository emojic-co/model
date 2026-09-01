export function rarest(
  keys: string[],
  counts: Map<string, number>,
  n: number,
): string[] {
  return keys
    .map((k, i) => ({ k, i, c: counts.get(k) ?? 0 }))
    .sort((a, b) => a.c - b.c || a.i - b.i)
    .slice(0, n)
    .map((x) => x.k)
}
