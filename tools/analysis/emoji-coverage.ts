import { readFile, writeFile } from "node:fs/promises"

import { parse as parseYaml } from "yaml"

import {
  EMOJI_COVERAGE_HTML,
  GOALS_YML,
  GROUP_JSON,
  LABELS_JSON,
} from "../../files.ts"

type Labels = { styles: string[]; emojis: string[] }
type Groups = Record<string, string[]>
type Targets = Record<string, number>

type GroupCoverage = {
  name: string
  members: string[]
  covered: Set<string>
  target: number | undefined
  score: number | null
  passed: boolean | null
}

const stripVariation = (e: string): string => e.replace(/️/g, "")

async function loadCoverage(): Promise<GroupCoverage[]> {
  const [groups, labels, goalsDoc] = await Promise.all([
    readFile(GROUP_JSON, "utf8").then((s) => JSON.parse(s) as Groups),
    readFile(LABELS_JSON, "utf8").then((s) => JSON.parse(s) as Labels),
    readFile(GOALS_YML, "utf8").then((s) => parseYaml(s)),
  ])

  const targets: Targets =
    goalsDoc?.goals?.vocabulary?.coverage ?? ({} as Targets)
  const vocab = new Set(labels.emojis.map(stripVariation))

  return Object.entries(groups).map(([name, members]) => {
    const covered = new Set(
      members.filter((e) => vocab.has(stripVariation(e))),
    )
    const target = targets[name]
    const score = members.length ? covered.size / members.length : null
    const passed = score === null || target === undefined ? null : score >= target
    return { name, members, covered, target, score, passed }
  })
}

function statusClass(passed: boolean | null): string {
  if (passed === null) return "na"
  return passed ? "good" : "bad"
}

function fmtPct(score: number | null): string {
  return score === null ? "—" : `${(score * 100).toFixed(0)}%`
}

function renderGroup(g: GroupCoverage): string {
  const emojis = g.members
    .map((e) => {
      const cls = g.covered.has(e) ? "in" : "out"
      return `<span class="em ${cls}" title="${cls === "in" ? "in vocab" : "not in vocab"}">${e}</span>`
    })
    .join("")
  return `<tr class="${statusClass(g.passed)}">
<td class="name">${g.name}</td>
<td class="n">${g.covered.size}/${g.members.length}</td>
<td class="n">${fmtPct(g.score)}</td>
<td class="n">${g.target === undefined ? "—" : `${(g.target * 100).toFixed(0)}%`}</td>
<td class="emojis">${emojis}</td>
</tr>`
}

function render(groups: GroupCoverage[]): string {
  const measurable = groups.filter((g) => g.passed !== null)
  const passed = measurable.filter((g) => g.passed).length
  const rows = groups.map(renderGroup).join("\n")
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Emoji vocabulary coverage</title>
<style>
:root{--ink:#1b1f24;--dim:#656b73;--line:#e2e5e9;--panel:#f5f6f8;
--good-bg:#e6f6ec;--good-bd:#b2dec1;--bad-bg:#fdeaea;--bad-bd:#f0b6b6}
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
color:var(--ink);margin:0;background:#fff}
.wrap{max-width:1100px;margin:0 auto;padding:36px 28px 100px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--dim);font-size:14px;margin:0 0 24px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.name{font-weight:600;white-space:nowrap}
tr.good{background:var(--good-bg)}
tr.bad{background:var(--bad-bg)}
tr.na{background:var(--panel);color:var(--dim)}
.emojis{font-size:16px;line-height:1.9}
.em{display:inline-block;margin:0 1px}
.em.out{opacity:.25;filter:grayscale(1)}
</style>
</head>
<body>
<div class="wrap">
<h1>Emoji vocabulary coverage</h1>
<p class="sub">${passed}/${measurable.length} groups meet their goals.yml target · grayed emoji are not in the current vocabulary (data/labels.json)</p>
<table>
<thead><tr><th>group</th><th class="n">covered</th><th class="n">score</th><th class="n">target</th><th>emojis</th></tr></thead>
<tbody>
${rows}
</tbody>
</table>
</div>
</body>
</html>
`
}

async function main(): Promise<void> {
  const groups = await loadCoverage()
  await writeFile(EMOJI_COVERAGE_HTML, render(groups), "utf8")
  console.log(EMOJI_COVERAGE_HTML)
}

main()
