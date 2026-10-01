# Unified Card Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web and Android preview and export play one shared, looping clip timeline, restarting from t=0 on any card change.

**Architecture:** A new `clip.yml` (ground truth, copied into `style.yml`) defines clip rules. Each platform has one clip module (`web/src/clip.js`, `android/.../model/Clip.kt`) that turns a card + emoji loop length into a clip length and per-time poses, verified against a shared golden fixture. Web preview becomes a canvas painted from that clip by the same painter the exports use (layout cached); Android preview feeds a looping clock into the same pose path its exports use.

**Tech Stack:** JS (React, vitest, gifenc, mp4-muxer, lottie-web), Kotlin (Compose, Lottie, JUnit), bun (`bun run export-style`).

**Spec:** `docs/superpowers/specs/2026-10-01-unified-card-animation-design.md`

## Global Constraints

- Frame rates: GIF 12 fps, MP4 20 fps, from `clip.yml` only (`gifFps`, `mp4Fps`).
- Clip length is a whole number of emoji loops, at least `minEmojiLoops` = 2 (spring-only emoji: text + shimmer length).
- `posterHoldMs` = 600: export-only first frame (finished card, no shimmer, emoji at rest); never part of the looping preview.
- Any change to emoji, feeling, language, text or colors restarts the clip at t=0.
- `web/src/clip.yml` is the single source; never hard-code clip numbers in JS/Kotlin. After editing it run `bun run export-style`; `tools/data/export-style.test.ts` fails on drift.
- `model/data.py` `normalize()` / `web/src/model.js` `normalize()` stay untouched.
- Text units/pose math stays in `textAnimation.js`/`TextAnimation.kt` and shimmer math in `shimmer*.js`/`Shimmer.kt`; do not change `textAnimations.yml` or `shimmers.yml`.
- Run `ruff` only if Python files change (none planned). Commit after each task; one logical change per commit.

## Review Focus

- Emoji with no Lottie clone (spring fallback): clip length = text + shimmer, spring plays once, no division by a 0 loop length.
- Very long Lottie loop or long text: clip stays within `maxClipMs` only by trimming the extra min-loops padding, never by cutting the entrance or shimmer.
- Rapid typing / rapid emoji or color cycling: only the latest card's painter may drive the canvas (stale async builds are discarded); no stacked rAF loops.
- `prefers-reduced-motion` (web) / animator scale 0 (Android): shows the finished frame, no loop running.
- RTL (`he`) and connected-script (Arabic/Thai/Devanagari) text keep the same unit positions in preview and export.
- Tab hidden / component unmounted: rAF loop and Lottie instances are released.

## File Structure

- Create `web/src/clip.yml` — clip rules (ground truth).
- Create `web/src/clip.js` — `CLIP`, `clipFor`, `emojiAt`/spring, `createTimeline` (web).
- Create `web/src/clip.test.js`, `web/src/clipFixture.json` — golden fixture + web tests.
- Modify `web/src/cardAnim.js` — expose `shimmerTimeline().pose(t)`.
- Modify `web/src/cardGif.js`, `web/src/cardMp4.js` — use the clip (length, fps, poster, spring).
- Modify `web/src/hooks/useCardImage.js` — cache layout in `createPainter`; poster support.
- Create `web/src/hooks/useCardPlayer.js`, `web/src/components/CardCanvas.jsx` — canvas preview.
- Modify `web/src/components/Card.jsx`, `web/src/styles.css`, `web/src/StylePreview.jsx`; delete `CharText.jsx`, `CardShimmer.jsx`, `AnimatedEmoji.jsx` once unused.
- Modify `files.py`, `files.ts`, `tools/data/export-style.ts`, `tools/data/export-style.test.ts`.
- Create `android/.../model/Clip.kt`, test `android/app/src/test/java/ing/emojify/model/ClipTest.kt`.
- Modify `android/.../model/StyleFile.kt`, `ui/components/Card.kt`, `AnimatedCharText.kt`, `CardShimmer.kt`, `AnimatedEmoji.kt`, `ui/MainScreen.kt`.

---

### Task 1: `clip.yml`, plumbing into style.yml, spec amendment

**Files:**
- Create: `web/src/clip.yml`
- Modify: `files.py` (after line 54), `files.ts` (after line 31), `tools/data/export-style.ts`, `tools/data/export-style.test.ts`, `android/app/src/main/java/ing/emojify/model/StyleFile.kt:54`
- Modify: `docs/superpowers/specs/2026-10-01-unified-card-animation-design.md` (spring + cap wording)

**Interfaces:**
- Produces: `style.yml` top-level key `clip` with fields `minEmojiLoops, maxClipMs, posterHoldMs, gifFps, mp4Fps, spring: {durationMs, from, zeta, omega}`; Kotlin `StyleFile.clip: ClipSpec?` (type defined in Task 3).

- [ ] **Step 1: Write `web/src/clip.yml`**

```yaml
# GROUND TRUTH for the shared preview/export clip. Web (web/src/clip.js) reads this directly;
# tools/data/export-style.ts copies it verbatim into style.yml under `clip` for Android.
# Edit here, then `bun run export-style`.
#
# Semantics (both platforms must follow exactly):
#  * Clip time t runs 0..L and loops. The text entrance starts at t=0; the shimmer starts at
#    entranceTotal + shimmers.startDelayMs and runs one cycle (pass + pause); the emoji Lottie loops
#    from t=0 with period loopMs (phase = (t mod loopMs) / loopMs).
#  * needed = ceil((entranceTotal + startDelayMs + shimmerCycle) / loopMs).
#    loops = max(needed, minEmojiLoops); then, while loops > needed and loops * loopMs > maxClipMs,
#    loops -= 1. L = loops * loopMs. maxClipMs never cuts the entrance or the shimmer.
#  * Emoji without a Lottie clone (loopMs = 0): L = entranceTotal + startDelayMs + shimmerCycle and the
#    emoji plays the spring once from t=0 (scale(t), see `spring`), then rests at scale 1.
#  * spring: scale(t) = 1 - (1 - from) * exp(-zeta*omega*t) * (cos(wd*t) + zeta*omega/wd * sin(wd*t)),
#    t in seconds, wd = omega * sqrt(1 - zeta^2); scale = 1 for t >= durationMs.
#  * Export only: the first frame is a poster (finished card text, no shimmer, emoji Lottie at progress 0
#    or, for the spring, at rest) held posterHoldMs, then the clip samples t = i * 1000 / fps for
#    i = 0 until ceil(L * fps / 1000), at gifFps (GIF) or mp4Fps (MP4).
minEmojiLoops: 2
maxClipMs: 12000
posterHoldMs: 600
gifFps: 12
mp4Fps: 20
spring:
  durationMs: 900
  from: 0.88
  zeta: 0.4
  omega: 14
```

- [ ] **Step 2: Register the path.** `files.py`: add `CLIP_YML = WEB_SRC_DIR / "clip.yml"`. `files.ts`: add ``export const CLIP_YML = `${WEB_SRC_DIR}/clip.yml` ``.

- [ ] **Step 3: Export it.** In `tools/data/export-style.ts` add `CLIP_YML` to the `files.ts` import, and in `buildStyleFile()` after `shimmer:` add `clip: parse(readFileSync(CLIP_YML, "utf-8")),`. In `tools/data/export-style.test.ts` add `clip: onDisk.clip` / `clip: fresh.clip` in the first `toEqual` pair (lines 12-15) and `expect(android.clip).toEqual(web.clip)` after line 26.

- [ ] **Step 4: Android model field.** In `StyleFile.kt` after `val shimmer: ShimmerSpec? = null,` add `val clip: ClipSpec? = null,` (the type is added in Task 3; until then this will not compile, so do this edit in Task 3 Step 3 instead and only run steps 1-3 and 5-6 here).

- [ ] **Step 5: Amend the spec.** In the spec section 1, replace "`maxClipMs`: safety cap; if exceeded, use the fewest loops that fit." with "`maxClipMs` only trims the extra `minEmojiLoops` padding; it never cuts the entrance or shimmer." and add a bullet "Spring parameters (`spring`) live in `clip.yml` so web and Android use one formula (they differed before: sine bounce vs damped spring)."

- [ ] **Step 6: Generate and test**

Run: `bun run export-style && bun test tools/data/export-style.test.ts`
Expected: writes `web/public/style.yml` and `android/app/src/main/assets/style.yml` (both now contain `clip:`), test PASS.

- [ ] **Step 7: Commit**

```bash
git add web/src/clip.yml files.py files.ts tools/data/export-style.ts tools/data/export-style.test.ts web/public/style.yml android/app/src/main/assets/style.yml docs/superpowers/specs
git commit -m "Add clip.yml shared timeline spec"
```

---

### Task 2: Web clip module + golden fixture

**Files:**
- Create: `web/src/clip.js`, `web/src/clip.test.js`, `web/src/clipFixture.json`
- Modify: `web/src/cardAnim.js` (add `pose` to `shimmerTimeline`)

**Interfaces:**
- Consumes: `textTimeline(motif, feeling, n)` → `{totalMs, pose(i,t)}`, `shimmerTimeline(cluster, feeling)` → `{cycleMs, startDelayMs, draw}` (existing, `cardAnim.js`).
- Produces (`web/src/clip.js`):
  - `CLIP` — parsed `clip.yml`.
  - `clipFor({ entranceMs, startDelayMs, cycleMs, loopMs }, spec = CLIP) → { durationMs, loops }`
  - `springScale(tMs, spring = CLIP.spring) → number`
  - `createTimeline({ motif, feeling, cluster, unitCount, loopMs }) → { durationMs, loops, loopMs, text: {totalMs, pose(i,t)}, shimmer: {cycleMs, startDelayMs, draw, pose(t)}, shimmerStartMs, emojiMs(t) }` where `emojiMs(t)` returns `t` (the emoji layer takes the loop modulo itself).
  - `shimmerTimeline(...).pose(t) → {c, opacity} | null`.

- [ ] **Step 1: Failing tests** in `web/src/clip.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { CLIP, clipFor, createTimeline, springScale } from './clip'

describe('clipFor', () => {
  const base = { entranceMs: 1000, startDelayMs: 250, cycleMs: 1500 } // 2750 ms of content
  it('rounds up to whole emoji loops, at least minEmojiLoops', () => {
    expect(clipFor({ ...base, loopMs: 1000 })).toEqual({ durationMs: 3000, loops: 3 })
    expect(clipFor({ ...base, loopMs: 2000 })).toEqual({ durationMs: 4000, loops: 2 })
  })
  it('trims only min-loop padding above maxClipMs', () => {
    expect(clipFor({ ...base, loopMs: 4000 }, { ...CLIP, maxClipMs: 5000 })).toEqual({ durationMs: 4000, loops: 1 })
    expect(clipFor({ ...base, loopMs: 4000 }, { ...CLIP, maxClipMs: 1000 })).toEqual({ durationMs: 4000, loops: 1 })
  })
  it('spring-only emoji uses just text + shimmer', () => {
    expect(clipFor({ ...base, loopMs: 0 })).toEqual({ durationMs: 2750, loops: 0 })
  })
})

describe('springScale', () => {
  it('starts shrunk, ends at rest', () => {
    expect(springScale(0)).toBeCloseTo(CLIP.spring.from, 5)
    expect(springScale(CLIP.spring.durationMs)).toBe(1)
  })
})

describe('createTimeline', () => {
  it('builds a loop whose shimmer starts after entrance + startDelay', () => {
    const tl = createTimeline({ motif: 'slam', feeling: 'Angry', cluster: 'anger', unitCount: 5, loopMs: 1500 })
    expect(tl.durationMs % 1500).toBe(0)
    expect(tl.shimmerStartMs).toBeCloseTo(tl.text.totalMs + tl.shimmer.startDelayMs)
    expect(tl.shimmer.pose(tl.shimmerStartMs - 1)).toBeNull()
    expect(tl.shimmer.pose(tl.shimmerStartMs + 1)).not.toBeNull()
    expect(tl.shimmerStartMs + tl.shimmer.cycleMs).toBeLessThanOrEqual(tl.durationMs + 1e-6)
  })
})
```

- [ ] **Step 2: Run, expect FAIL**

Run: `cd web && npx vitest run src/clip.test.js`
Expected: FAIL (`./clip` not found).

- [ ] **Step 3: Add `pose` to `shimmerTimeline`.** In `web/src/cardAnim.js` split the body of `draw` so the keyframe evaluation is a function returning `{c, opacity}` or `null`, and make `draw` use it:

```js
  const pose = (t) => {
    const u = (t % cycleMs) / e.durationMs
    if (t < 0 || u >= 1) return null
    let k = frames[frames.length - 1]
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1]
      const b = frames[i]
      if (u <= b.at) {
        const f = ease(b.at === a.at ? 1 : (u - a.at) / (b.at - a.at))
        k = Object.fromEntries(SHIMMER_FIELDS.map((n) => [n, lerp(a[n], b[n], f)]))
        break
      }
    }
    return k
  }
  const draw = (ctx, S, t) => {
    const k = pose(t)
    if (!k) return
    let fill
    // ...existing radial / flash / sweep fill construction and ctx.save() ... ctx.restore() unchanged
  }
  return { cycleMs, startDelayMs: anim.startDelayMs, draw, pose }
```

(Keep the existing fill-construction code verbatim inside `draw`; only the keyframe lookup moves.)

- [ ] **Step 4: Write `web/src/clip.js`**

```js
// Shared clip timeline for preview and export (rules: clip.yml, the ground truth shared with Android).
import { parse } from 'yaml'
import raw from './clip.yml?raw'
import { shimmerTimeline, textTimeline } from './cardAnim'

export const CLIP = parse(raw)

export function clipFor({ entranceMs, startDelayMs, cycleMs, loopMs }, spec = CLIP) {
  const content = entranceMs + startDelayMs + cycleMs
  if (!loopMs) return { durationMs: content, loops: 0 }
  const needed = Math.max(1, Math.ceil(content / loopMs))
  let loops = Math.max(needed, spec.minEmojiLoops)
  while (loops > needed && loops * loopMs > spec.maxClipMs) loops -= 1
  return { durationMs: loops * loopMs, loops }
}

export function springScale(tMs, s = CLIP.spring) {
  if (tMs >= s.durationMs) return 1
  const t = Math.max(0, tMs) / 1000
  const wd = s.omega * Math.sqrt(1 - s.zeta * s.zeta)
  const decay = Math.exp(-s.zeta * s.omega * t)
  const osc = Math.cos(wd * t) + ((s.zeta * s.omega) / wd) * Math.sin(wd * t)
  return 1 - (1 - s.from) * decay * osc
}

export function createTimeline({ motif, feeling, cluster, unitCount, loopMs }) {
  const text = textTimeline(motif, feeling, unitCount)
  const shimmer = shimmerTimeline(cluster, feeling)
  const shimmerStartMs = text.totalMs + shimmer.startDelayMs
  const { durationMs, loops } = clipFor({
    entranceMs: text.totalMs,
    startDelayMs: shimmer.startDelayMs,
    cycleMs: shimmer.cycleMs,
    loopMs,
  })
  return {
    durationMs,
    loops,
    loopMs,
    text,
    shimmer: { ...shimmer, pose: (t) => shimmer.pose(t - shimmerStartMs) },
    shimmerStartMs,
    emojiMs: (t) => t,
  }
}
```

- [ ] **Step 5: Run, expect PASS**

Run: `cd web && npx vitest run src/clip.test.js src/cardAnim.test.js`
Expected: PASS. (Fix the `shimmer.pose` wrapper's meaning if `cardAnim`'s `draw` still takes cycle-relative time: `draw(ctx, S, t - shimmerStartMs)` is what the painter calls, same as today.)

- [ ] **Step 6: Golden fixture test.** Append to `clip.test.js`:

```js
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const CASES = [
  { motif: 'slam', feeling: 'Angry', cluster: 'anger', unitCount: 5, loopMs: 1500 },
  { motif: 'settle', feeling: 'Serene', cluster: 'reflective', unitCount: 12, loopMs: 2400 },
  { motif: 'jitter', feeling: 'Sarcastic', cluster: 'drive', unitCount: 9, loopMs: 0 },
]
const TIMES = [0, 250, 700, 1500, 2600, 3300]
const path = fileURLToPath(new URL('./clipFixture.json', import.meta.url))

function build() {
  return CASES.map((c) => {
    const tl = createTimeline(c)
    return {
      ...c,
      durationMs: tl.durationMs,
      loops: tl.loops,
      samples: TIMES.filter((t) => t <= tl.durationMs).map((t) => ({
        t,
        unit0: tl.text.pose(0, t),
        unitLast: tl.text.pose(c.unitCount - 1, t),
        shimmer: tl.shimmer.pose(t),
        spring: c.loopMs ? null : springScale(t),
      })),
    }
  })
}

describe('golden fixture (shared with Android ClipTest)', () => {
  it('matches clipFixture.json (UPDATE_CLIP_FIXTURE=1 regenerates)', () => {
    const fresh = build()
    if (process.env.UPDATE_CLIP_FIXTURE || !existsSync(path)) writeFileSync(path, JSON.stringify(fresh, null, 2))
    expect(JSON.parse(JSON.stringify(fresh))).toEqual(JSON.parse(readFileSync(path, 'utf8')))
  })
})
```

Run: `cd web && UPDATE_CLIP_FIXTURE=1 npx vitest run src/clip.test.js && npx vitest run src/clip.test.js`
Expected: first run writes the fixture, second PASS. Open the JSON and sanity check: `durationMs` multiples of `loopMs`, shimmer null at `t=0`.

- [ ] **Step 7: Commit**

```bash
git add web/src/clip.js web/src/clip.test.js web/src/clipFixture.json web/src/cardAnim.js
git commit -m "Add web clip timeline and golden fixture"
```

---

### Task 3: Android clip module, parity test

**Files:**
- Create: `android/app/src/main/java/ing/emojify/model/Clip.kt`, `android/app/src/test/java/ing/emojify/model/ClipTest.kt`
- Modify: `android/app/src/main/java/ing/emojify/model/StyleFile.kt:54`

**Interfaces:**
- Consumes: `textScheduleFor(...)`/`TextSchedule` (`TextAnimation.kt`), `ShimmerPlayer(effect, startMs).poseAt(elapsedMs)`, `resolveShimmer` (`Shimmer.kt`); `web/src/clipFixture.json` (Task 2).
- Produces:
  - `ClipSpec(minEmojiLoops: Int, maxClipMs: Double, posterHoldMs: Int, gifFps: Int, mp4Fps: Int, spring: SpringSpec)`, `SpringSpec(durationMs, from, zeta, omega: Double)`
  - `data class ClipLength(val durationMs: Double, val loops: Int)`; `fun clipLength(entranceMs: Double, startDelayMs: Double, cycleMs: Double, loopMs: Double, spec: ClipSpec): ClipLength`
  - `fun springScale(tMs: Double, s: SpringSpec): Double`
  - `class ClipTimeline(styles: StyleFile, motif: String, feeling: String?, cluster: String, unitCount: Int, loopMs: Double)` with `durationMs`, `loops`, `loopMs`, `schedule: TextSchedule`, `shimmerPlayer: ShimmerPlayer`, `shimmerStartMs`, `shimmerPoseAt(t): ShimmerPose?`.

- [ ] **Step 1: Failing test** `ClipTest.kt`:

```kotlin
package ing.emojify.model

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ClipTest {
    private val file = parseStyleFile(File("src/main/assets/style.yml").readText())
    private val spec = file.clip!!

    @Test
    fun `clip length rounds up to whole emoji loops`() {
        assertEquals(3000.0, clipLength(1000.0, 250.0, 1500.0, 1000.0, spec).durationMs, 0.0)
        assertEquals(4000.0, clipLength(1000.0, 250.0, 1500.0, 2000.0, spec).durationMs, 0.0)
        assertEquals(2750.0, clipLength(1000.0, 250.0, 1500.0, 0.0, spec).durationMs, 0.0)
    }

    @Test
    fun `matches the web golden fixture`() {
        val cases = Json.parseToJsonElement(File("../../web/src/clipFixture.json").readText()).jsonArray
        for (c in cases) {
            val o = c.jsonObject
            val tl = ClipTimeline(
                file, o["motif"]!!.jsonPrimitive.content, o["feeling"]!!.jsonPrimitive.content,
                o["cluster"]!!.jsonPrimitive.content, o["unitCount"]!!.jsonPrimitive.int, o["loopMs"]!!.jsonPrimitive.double,
            )
            assertEquals(o["durationMs"]!!.jsonPrimitive.double, tl.durationMs, 1e-6)
            assertEquals(o["loops"]!!.jsonPrimitive.int, tl.loops)
            for (s in o["samples"]!!.jsonArray) {
                val so = s.jsonObject
                val t = so["t"]!!.jsonPrimitive.double
                val u0 = so["unit0"]!!.jsonObject
                val p0 = tl.schedule.poseAt(0, t)
                assertEquals(u0["opacity"]!!.jsonPrimitive.double, p0.opacity.toDouble(), 1e-3)
                assertEquals(u0["x"]!!.jsonPrimitive.double, p0.x.toDouble(), 1e-3)
                assertEquals(u0["scale"]!!.jsonPrimitive.double, p0.scale.toDouble(), 1e-3)
                val sh = so["shimmer"]
                val kp = tl.shimmerPoseAt(t)
                if (sh == null || sh is JsonNull) assertNull(kp)
                else {
                    assertEquals(sh.jsonObject["c"]!!.jsonPrimitive.double, kp!!.c.toDouble(), 1e-3)
                    assertEquals(sh.jsonObject["opacity"]!!.jsonPrimitive.double, kp.opacity.toDouble(), 1e-3)
                }
                val sp = so["spring"]
                if (sp != null && sp !is JsonNull) assertEquals(sp.jsonPrimitive.double, springScale(t, spec.spring), 1e-6)
            }
        }
    }
}
```

(If `kotlinx-serialization-json` is not on the test classpath, check `android/app/build.gradle.kts`; the project already uses `kotlinx.serialization` for yaml, so add the json artifact next to it if needed.)

- [ ] **Step 2: Run, expect FAIL (compile error)**

Run: `cd android && ./gradlew testDebugUnitTest --tests 'ing.emojify.model.ClipTest'`
Expected: FAIL, `Unresolved reference: ClipSpec`/`clipLength`.

- [ ] **Step 3: Implement `Clip.kt`** and the `StyleFile` field.

```kotlin
package ing.emojify.model

import kotlinx.serialization.Serializable
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

// Shared preview/export clip timeline. Rules live in web/src/clip.yml (ground truth, copied into
// style.yml as `clip`); semantics are in that file's header and mirrored by web/src/clip.js.

@Serializable
data class SpringSpec(val durationMs: Double, val from: Double, val zeta: Double, val omega: Double)

@Serializable
data class ClipSpec(
    val minEmojiLoops: Int,
    val maxClipMs: Double,
    val posterHoldMs: Int,
    val gifFps: Int,
    val mp4Fps: Int,
    val spring: SpringSpec,
)

data class ClipLength(val durationMs: Double, val loops: Int)

fun clipLength(entranceMs: Double, startDelayMs: Double, cycleMs: Double, loopMs: Double, spec: ClipSpec): ClipLength {
    val content = entranceMs + startDelayMs + cycleMs
    if (loopMs <= 0.0) return ClipLength(content, 0)
    val needed = maxOf(1, ceil(content / loopMs).toInt())
    var loops = maxOf(needed, spec.minEmojiLoops)
    while (loops > needed && loops * loopMs > spec.maxClipMs) loops -= 1
    return ClipLength(loops * loopMs, loops)
}

fun springScale(tMs: Double, s: SpringSpec): Double {
    if (tMs >= s.durationMs) return 1.0
    val t = maxOf(0.0, tMs) / 1000.0
    val wd = s.omega * sqrt(1 - s.zeta * s.zeta)
    val decay = exp(-s.zeta * s.omega * t)
    val osc = cos(wd * t) + s.zeta * s.omega / wd * sin(wd * t)
    return 1 - (1 - s.from) * decay * osc
}

class ClipTimeline(
    styles: StyleFile,
    motif: String,
    feeling: String?,
    cluster: String,
    unitCount: Int,
    val loopMs: Double,
) {
    private val spec = styles.clip!!
    val schedule: TextSchedule = textScheduleFor(styles.textAnimations!!, motif, feeling, unitCount)
    private val shimmerSpec = styles.shimmer!!
    private val effect = resolveShimmer(shimmerSpec, cluster, feeling)
    val shimmerStartMs: Double = schedule.totalMs + shimmerSpec.startDelayMs
    val shimmerPlayer = ShimmerPlayer(effect, shimmerStartMs)
    private val length = clipLength(schedule.totalMs, shimmerSpec.startDelayMs, effect.durationMs + effect.pauseMs, loopMs, spec)
    val durationMs: Double = length.durationMs
    val loops: Int = length.loops

    fun shimmerPoseAt(tMs: Double): ShimmerPose? = shimmerPlayer.poseAt(tMs)
}
```

In `StyleFile.kt` after line 54 add `val clip: ClipSpec? = null,`.

Note: the web fixture's `unit0.opacity` etc. come from `textTimeline.pose`; Android `schedule.poseAt(i, t)` has the same math (`TextAnimationTest` already cross-checks the schedule), so the only new risk is the shimmer/spring/length numbers.

- [ ] **Step 4: Run, expect PASS**

Run: `cd android && ./gradlew testDebugUnitTest --tests 'ing.emojify.model.ClipTest'`
Expected: PASS. Fix any difference by correcting Kotlin to match web (web + yml are the ground truth).

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/model/Clip.kt android/app/src/main/java/ing/emojify/model/StyleFile.kt android/app/src/test/java/ing/emojify/model/ClipTest.kt
git commit -m "Add Android clip timeline matching web fixture"
```

---

### Task 4: Web painter layout cache + export on the clip

**Files:**
- Modify: `web/src/hooks/useCardImage.js` (`createPainter`), `web/src/cardGif.js`, `web/src/cardMp4.js`, `web/src/cardAnim.test.js` (if signatures change)
- Test: `web/src/clip.test.js` (add), `web/src/cardGif.test.js` (new, small)

**Interfaces:**
- Consumes: `createTimeline`, `springScale`, `CLIP` (Task 2); existing `emojiLayer(emoji)` → `{loopMs, draw(timeMs, ctx, x, y, px), destroy}`.
- Produces:
  - `createPainter(cardData, scale)` → `paint(drawEmoji, opts)` unchanged API, plus `paint.timeline(loopMs)` → clip timeline for this laid-out card (`unitCount` from the cached layout) and `paint.frame(t, emoji)` → canvas for clip time `t`, and `paint.poster(emoji)` → canvas for the poster frame. Layout (font fit, wrapped lines, unit x offsets) is computed once.
  - `emojiLayer` spring fallback uses `springScale` (no more `1 + 0.12 sin`); `emoji.restMs` (poster time for the emoji: `0` for Lottie, `CLIP.spring.durationMs` for spring).
  - `clipDurationMs` is removed; exports use `paint.timeline(emoji.loopMs).durationMs`.

- [ ] **Step 1: Failing tests.** In a new `web/src/cardGif.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { frameCount } from './cardGif'
import { CLIP } from './clip'

describe('frameCount', () => {
  it('samples ceil(L * fps / 1000) frames', () => {
    expect(frameCount(3000, CLIP.gifFps)).toBe(36)
    expect(frameCount(3000, CLIP.mp4Fps)).toBe(60)
    expect(frameCount(1, 12)).toBe(1)
  })
})
```

Run: `cd web && npx vitest run src/cardGif.test.js` → FAIL (`frameCount` missing).

- [ ] **Step 2: Cache layout in `createPainter`.** Move the font-fit / wrap / unit-position computation out of `paintContent` into `const layout = computeLayout(octx)` evaluated lazily once (`let layout`; `layout ??= computeLayout(ctx)` at the top of `paintContent`). `computeLayout` returns `{fpx, lines, emojiCenterY, textCenterY, blockH, unitLines, lineWidths, unitCx}` where `unitCx[li][k]` is the x centre of unit `k` on line `li` (currently computed per unit per frame from `measureText`; keep the RTL branch: `rtl ? S/2 + width/2 - mid : S/2 - width/2 + mid`). `paintContent` then only loops draw calls. Verify the static (`animate` false) output is unchanged by running the existing jpg path manually (Step 7).

- [ ] **Step 3: Timeline and frame helpers.** After `paintContent`:

```js
  const unitCount = () => (layout ??= computeLayout(octx)).unitLines.reduce((k, u) => k + u.length, 0)
  const resolved = resolveFeeling(feeling, lang)
  paint.timeline = (loopMs) =>
    createTimeline({ motif: resolved.entrance, feeling, cluster: resolved.cluster, unitCount: unitCount(), loopMs })
  paint.frame = (tl, emoji, t) => paint((ctx, x, y, px) => emoji.draw(tl.emojiMs(t), ctx, x, y, px), { animate: true, timeMs: t, timeline: tl })
  paint.poster = (emoji) => paint((ctx, x, y, px) => emoji.draw(emoji.restMs, ctx, x, y, px), {})
```

and have `paintContent` use `opts.timeline` instead of building `motion` itself (delete the `motion` cache and `paint.motion`; shimmer draw becomes `opts.timeline.shimmer.draw(ctx, S, opts.timeMs - opts.timeline.shimmerStartMs)`, text pose `opts.timeline.text.pose(unitIndex++, opts.timeMs)`). Import `createTimeline` from `../clip`.

- [ ] **Step 4: Unify spring + poster time in `emojiLayer`.** In `cardGif.js` replace the fallback branch with:

```js
  return {
    loopMs: 0,
    restMs: CLIP.spring.durationMs,
    draw(timeMs, ctx, x, y, px) {
      ctx.font = `${px * springScale(timeMs)}px "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif`
      ctx.fillText(emoji, x, y)
    },
    destroy() {},
  }
```

add `restMs: 0` to the Lottie branch, delete `SPRING_MS`/`MAX_MS`, and `export const frameCount = (durationMs, fps) => Math.max(1, Math.ceil((durationMs * fps) / 1000))`. Import `CLIP, springScale` from `./clip`.

- [ ] **Step 5: GIF export.** In `encode`, replace the timeline/loop:

```js
  const paint = await createPainter(cardData, size / 512)
  const tl = paint.timeline(emoji.loopMs)
  const count = frameCount(tl.durationMs, CLIP.gifFps)
  const frames = []
  frames.push(grabPixels(paint.poster(emoji)))            // poster frame, held CLIP.posterHoldMs
  for (let i = 0; i < count; i++) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
    frames.push(grabPixels(paint.frame(tl, emoji, (i * 1000) / CLIP.gifFps)))
    onProgress?.(((i + 1) / count) * 0.8)
    await tick()
  }
```

with `const grabPixels = (c) => c.getContext('2d').getImageData(0, 0, c.width, c.height).data`. In pass 2 use `const delay = (i) => (i === 0 ? CLIP.posterHoldMs : Math.round(1000 / CLIP.gifFps))` and `{ palette, delay: delay(i) }` over `frames.length` frames; remove `const FPS`. (`gifenc` rounds to centiseconds: 83 ms stores as 8 cs. Known limit.)

- [ ] **Step 6: MP4 export.** In `cardMp4.js`: `const FPS = CLIP.mp4Fps`; `count = frameCount(tl.durationMs, FPS)` with `tl = paint.timeline(emoji.loopMs)`; encode the poster first with `timestamp: 0, duration: CLIP.posterHoldMs * 1000`, then frame `i` with `timestamp: (CLIP.posterHoldMs + (i * 1000) / FPS) * 1000` and `duration: Math.round(1e6 / FPS)`; use `paint.frame(tl, emoji, (i*1000)/FPS)` for pixels; `keyFrame: i % FPS === 0`. Remove the `clipDurationMs` import.

- [ ] **Step 7: Run tests + manual check**

Run: `cd web && npx vitest run`
Expected: PASS (update `cardAnim.test.js` only if it referenced `paint.motion`; it does not).
Manual: `cd web && npm run dev`, export a gif and an mp4: first 0.6 s is the finished card, then the entrance, shimmer after text, emoji loops whole times; open the jpg copy and confirm the card looks identical to before.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "Web exports use the shared clip: poster frame, 12/20 fps, unified spring"
```

---

### Task 5: Web canvas preview

**Files:**
- Create: `web/src/hooks/useCardPlayer.js`, `web/src/components/CardCanvas.jsx`, `web/src/hooks/useCardPlayer.test.js`
- Modify: `web/src/components/Card.jsx`, `web/src/styles.css`, `web/src/StylePreview.jsx`
- Delete (after nothing imports them): `web/src/components/CharText.jsx`, `CardShimmer.jsx`, `AnimatedEmoji.jsx`, and now-dead `playText`/`playShimmer`/`resolveShimmer` DOM players if unused (`grep -rn` first; keep `scheduleFor`, `entranceTotalMs`, `splitWords`, `shimmerTimeline` users).

**Interfaces:**
- Consumes: `createPainter`, `paint.timeline`, `paint.frame`, `paint.poster`, `emojiLayer`, `prefersReducedMotion` (`notoLottie.js`).
- Produces:
  - `useCardPlayer(cardData, canvasRef)` — builds a painter for `cardData` (emoji, feeling, lang, text, colors), runs a rAF loop `t = (now - start) % durationMs`, draws into `canvasRef.current`; restarts at t=0 whenever `cardData`'s contents change; discards stale async builds; static finished frame under reduced motion; cleans up rAF + Lottie on change/unmount.
  - `CardCanvas({ cardData })` — `<canvas className="card-canvas">` sized to its CSS box × `min(devicePixelRatio, 2)`.
  - `playerKey(cardData) → string` pure helper (`emoji|feeling|lang|text|bg1|bg2|text_color`) used as the restart trigger; unit-tested.

- [ ] **Step 1: Failing test** `useCardPlayer.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { playerKey } from './useCardPlayer'

const base = { emoji: '😀', feeling: 'Joyful', lang: 'en', text: 'hi', colors: { bg1: [0.5, 0, 0], bg2: [0.6, 0, 0], text_color: [0, 0, 0] } }

describe('playerKey', () => {
  it('changes for emoji, feeling, lang, text and colors', () => {
    const k = playerKey(base)
    expect(playerKey({ ...base, emoji: '😢' })).not.toBe(k)
    expect(playerKey({ ...base, feeling: 'Serene' })).not.toBe(k)
    expect(playerKey({ ...base, lang: 'he' })).not.toBe(k)
    expect(playerKey({ ...base, text: 'hi!' })).not.toBe(k)
    expect(playerKey({ ...base, colors: { ...base.colors, bg1: [0.4, 0, 0] } })).not.toBe(k)
  })
  it('is stable for equal content', () => {
    expect(playerKey({ ...base, colors: { ...base.colors } })).toBe(playerKey(base))
  })
})
```

Run: `cd web && npx vitest run src/hooks/useCardPlayer.test.js` → FAIL.

- [ ] **Step 2: Implement `useCardPlayer.js`**

```js
import { useEffect } from 'react'
import { createPainter } from './useCardImage'
import { emojiLayer } from '../cardGif'
import { prefersReducedMotion } from '../notoLottie'

export const playerKey = ({ emoji, feeling, lang, text, colors }) =>
  [emoji, feeling, lang, text, JSON.stringify(colors)].join('|')

const REBUILD_DEBOUNCE_MS = 150

// Preview player: draws the shared clip (same painter + timeline as GIF/MP4 export) in a loop.
export function useCardPlayer(cardData, canvasRef, scale = 1) {
  const key = cardData ? playerKey(cardData) : ''
  useEffect(() => {
    if (!cardData) return
    let cancelled = false
    let raf = 0
    let emoji
    const timer = setTimeout(async () => {
      const paint = await createPainter(cardData, scale)
      const layer = await emojiLayer(cardData.emoji)
      if (cancelled) return layer.destroy()
      emoji = layer
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      const blit = (src) => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(src, 0, 0, canvas.width, canvas.height)
      }
      if (prefersReducedMotion()) return blit(paint.poster(emoji))
      const tl = paint.timeline(emoji.loopMs)
      const start = performance.now()
      const tick = (now) => {
        if (cancelled) return
        blit(paint.frame(tl, emoji, (now - start) % tl.durationMs))
        raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
    }, REBUILD_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
      cancelAnimationFrame(raf)
      emoji?.destroy()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, scale])
}
```

(`scale` is the painter's backing-resolution factor, i.e. `canvas.width / 512`; `CardCanvas` passes it.)

- [ ] **Step 3: `CardCanvas.jsx`**

```jsx
import { useEffect, useRef, useState } from 'react'
import { useCardPlayer } from '../hooks/useCardPlayer'

export function CardCanvas({ cardData }) {
  const ref = useRef(null)
  const [scale, setScale] = useState(1)
  useEffect(() => {
    const el = ref.current
    const ro = new ResizeObserver(() => {
      const px = Math.round(el.clientWidth * Math.min(window.devicePixelRatio || 1, 2))
      if (px && el.width !== px) {
        el.width = px
        el.height = px
        setScale(px / 512)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  useCardPlayer(cardData, ref, scale)
  return <canvas ref={ref} className="card-canvas" aria-hidden="true" />
}
```

- [ ] **Step 4: Rewire `Card.jsx`.** Replace the `CardShimmer` / `AnimatedEmoji` / `.card-text-box` + `CharText` children and the inline-style background computation with `<CardCanvas cardData={cardData} />`, where `cardData = !loading && colors && shown.feeling ? { text: displayText, emoji: shown.emoji, feeling: shown.feeling, lang: shown.lang, colors } : null` (same shape `createPainter` takes today from `App.jsx` `cardData`). Keep: the outer `.card` div with `ref`, `data-phase` fade, the `.share-bar`. Remove the card watermark span (the painter draws it) and the `useFitText` call. The `shown`/`FADE_MS` fade-out logic stays so the old card fades before the new one starts at t=0 (the player restarts on `playerKey` change).

- [ ] **Step 5: CSS.** In `styles.css` add `.card-canvas { position: absolute; inset: 0; width: 100%; height: 100%; }` and delete rules that only served the removed DOM layers: `.card-emoji*`, `.card[data-emoji=...]`, `.card-emoji-lottie`, `.card-text-box`, `.card-text*`, `.tw`, `.tw-nowrap`, `.tc`, `.card-shimmer`, `.card-watermark`, and the keyframes they used (`grep -n "@keyframes"` for names only referenced by them). Keep `.card` sizing/radius/`container-type`, `.card[data-phase]` fade (retarget it to `.card[data-phase="out"] .card-canvas { opacity: 0; transition: opacity 150ms }`), `.share-bar*`, `.share-progress`.

- [ ] **Step 6: `StylePreview.jsx`** imports `CharText`: either switch it to a small `<CardCanvas>` per style (preferred, one source of preview truth) or keep a minimal local DOM text renderer; pick one so `CharText.jsx` can be deleted. Then delete the three components and dead DOM-player code, and run `cd web && npx vitest run && npm run build`.
Expected: PASS and a clean build (`grep -rn "CharText\|CardShimmer\|AnimatedEmoji" web/src` returns nothing).

- [ ] **Step 7: Manual verification + perf**

Run `cd web && npm run dev`. Check: (a) cycling emoji, feeling and colors each restarts the entrance from the first frame; (b) rapid arrow-key cycling never doubles the animation speed or leaves a stale card; (c) a text with Hebrew and one with Thai/Arabic look right; (d) OS reduced-motion shows the finished frame; (e) Chrome DevTools Performance recording on desktop shows a frame cost well under 8 ms, and on a mid-range phone via remote debugging stays at ≥ 30 fps. If (e) fails, reduce `scale` cap to 1.5 and re-measure before moving on.

- [ ] **Step 8: Commit**

```bash
git add -A web/src
git commit -m "Web preview renders the shared clip on a canvas"
```

---

### Task 6: Android export on the clip

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/ui/MainScreen.kt:346-440` (`exportCardAnimation`), `ui/components/Card.kt:300` (`ExportPose`), `AnimatedCharText.kt`, `CardShimmer.kt`, `AnimatedEmoji.kt`
- Test: `ClipTest.kt` (add frame-count test)

**Interfaces:**
- Consumes: `ClipTimeline`, `ClipSpec`, `springScale` (Task 3).
- Produces:
  - `data class ClipPose(val tMs: Float, val poster: Boolean = false)` replacing `ExportPose` (one time value drives all layers).
  - `fun clipFrameCount(durationMs: Double, fps: Int): Int` in `Clip.kt` = `max(1, ceil(durationMs * fps / 1000))`.
  - `suspend fun loadEmojiLoopMs(context: Context, emoji: String): Double` in `NotoLottie.kt` — Lottie composition duration in ms, `0.0` when the emoji has no clone.
  - Leaf composables take the pose: `AnimatedCharText(textMs: Float?)`, `CardShimmer(timeMs: Float?)` (draws `timeline.shimmerPoseAt`), `AnimatedEmoji(loopMs: Float, tMs: Float?)`.

- [ ] **Step 1: Failing test** (append to `ClipTest.kt`):

```kotlin
    @Test
    fun `frame count is ceil(L * fps / 1000)`() {
        assertEquals(36, clipFrameCount(3000.0, spec.gifFps))
        assertEquals(60, clipFrameCount(3000.0, spec.mp4Fps))
        assertEquals(1, clipFrameCount(1.0, 12))
    }
```

Run `cd android && ./gradlew testDebugUnitTest --tests 'ing.emojify.model.ClipTest'` → FAIL (`clipFrameCount`). Then add to `Clip.kt`: `fun clipFrameCount(durationMs: Double, fps: Int): Int = maxOf(1, ceil(durationMs * fps / 1000.0).toInt())` → PASS.

- [ ] **Step 2: `NotoLottie.loadEmojiLoopMs`.**

```kotlin
suspend fun loadEmojiLoopMs(context: Context, emoji: String): Double {
    val path = assetPath(context, emoji) ?: return 0.0
    return withContext(Dispatchers.IO) {
        com.airbnb.lottie.LottieCompositionFactory.fromAssetSync(context, path).value?.duration?.toDouble()
    } ?: 0.0
}
```

(add the `kotlinx.coroutines` imports if absent).

- [ ] **Step 3: Leaf composables take clip time.**
  - `AnimatedCharText`: rename `exportMs` → `textMs` (same behavior: `clock.snapTo(textMs)` when non-null) — it already renders purely from `clock`. No other change.
  - `CardShimmer`: replace `entranceMs`/`replayKey`/`exportProgress` and its own `withFrameNanos` clock with `timeline: ClipTimeline?` and `timeMs: Float?`; body: `val pose = timeline?.shimmerPoseAt(timeMs.toDouble()) ?: return@Canvas` (draw code and `shimmerBrush` unchanged). Remove the animator-scale check from here (moved to Card).
  - `AnimatedEmoji`: replace `progress` with `loopMs: Float` and `tMs: Float?`; Lottie progress `(tMs % loopMs) / loopMs` clamped to `0f..0.999f`; spring scale `springScale(tMs.toDouble(), spec).toFloat()` (pass `spec`); delete the local `springScale`/`SPRING_MS`; `static` stays for the shared still. When `tMs == null` (no clip yet / reduced motion) draw the rest pose (Lottie progress 0, scale 1).

- [ ] **Step 4: `Card` pose type.** In `Card.kt` replace `data class ExportPose(...)` with `data class ClipPose(val tMs: Float, val poster: Boolean = false)` and the `export: ExportPose?` param with `pose: ClipPose?` (null = Card runs its own preview loop, added in Task 7). Wire: `AnimatedCharText(textMs = if (pose.poster) TEXT_ANIM_DONE else pose.tMs)`, `CardShimmer(timeline, if (pose.poster) null else pose.tMs)`, `AnimatedEmoji(tMs = if (pose.poster) restMs else pose.tMs)` where poster rest = `0f` for Lottie and `spec.spring.durationMs` for the spring.

- [ ] **Step 5: Rewrite `exportCardAnimation`.**

```kotlin
private suspend fun exportCardAnimation(
    context: Context,
    format: ShareFormat,
    emoji: String,
    capture: suspend () -> android.graphics.Bitmap,
    timeline: ClipTimeline,
    onProgress: (Float) -> Unit,
    setPose: (ClipPose) -> Unit,
) {
    val spec = Styles.file.clip!!
    val fps = if (format == ShareFormat.Gif) spec.gifFps else spec.mp4Fps
    val frames = clipFrameCount(timeline.durationMs, fps)
    val frameDelayCs = 100 / fps
    val gifFile = cardGifFile(context)
    val mp4File = cardMp4File(context)
    val gif = if (format == ShareFormat.Gif) GifEncoder(gifFile, GIF_SIZE_PX, delayCs = frameDelayCs) else null
    val mp4 = if (format == ShareFormat.Mp4) Mp4Encoder(mp4File, exportSizePx(context), fps) else null

    suspend fun grab(pose: ClipPose, delayMs: Int) {
        setPose(pose)
        withFrameNanos { }
        withFrameNanos { }
        val bitmap = capture()
        withContext(Dispatchers.Default) {
            gif?.addFrame(bitmap, delayMs / 10)
            mp4?.addFrame(bitmap, delayMs)
        }
    }

    grab(ClipPose(0f, poster = true), spec.posterHoldMs)
    for (i in 0 until frames) {
        onProgress(i / frames.toFloat())
        grab(ClipPose(i * 1000f / fps), 1000 / fps)
    }
    // finish + share unchanged
}
```

(For MP4 the per-frame duration is `1000 / fps` = 50 ms; for GIF `100 / fps` cs = 8 cs at 12 fps, via `delayMs / 10` = 83 / 10 = 8.) Delete `EXPORT`/`TARGET_EMOJI_MS`/`MIN_EMOJI_LOOPS`/`MAX_EMOJI_LOOPS`/`POSTER_HOLD_CS`/`emojiLoops`, `GIF_FPS`/`MP4_FPS` (added earlier this session; the yml owns them now). Build the `ClipTimeline` at the call site (the share handler) from the card's `style.entranceMotif`, feeling, `style.cluster`, the text unit count (`splitTextUnits(displayText).sumOf { it.size }`) and `loadEmojiLoopMs(context, emoji)`, and pass it in; `MainScreen`'s `exportPose` state becomes `ClipPose?`.

- [ ] **Step 6: Build and verify**

Run: `cd android && ./gradlew testDebugUnitTest && ./gradlew installDebug`
Expected: tests PASS, installs on the Pixel. Manual: export a GIF and an MP4; first 0.6 s poster, then the entrance, then shimmer after the text, emoji loops whole times; GIF at 12 fps and MP4 at 20 fps.

- [ ] **Step 7: Commit**

```bash
git add android
git commit -m "Android exports use the shared clip timeline"
```

---

### Task 7: Android preview loops the clip

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/ui/components/Card.kt`, `ui/MainScreen.kt`

**Interfaces:**
- Consumes: `ClipTimeline`, `loadEmojiLoopMs`, `ClipPose` (Task 6).
- Produces: when `pose == null` the `Card` runs a looping clock `t in [0, durationMs)` that restarts whenever `(emoji, feeling, lang, text, colors)` change; honors animator scale 0 (shows the finished frame).

- [ ] **Step 1: Card-owned clock.** In `Card`, above `Box`:

```kotlin
val context = LocalContext.current
val animationsOff = remember {
    Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
}
var loopMs by remember(emoji) { mutableStateOf<Double?>(null) }
LaunchedEffect(emoji) { loopMs = loadEmojiLoopMs(context, emoji) }
val unitCount = remember(displayText) { splitTextUnits(displayText).sumOf { it.size } }
val timeline = remember(loopMs, style.entranceMotif, feeling, style.cluster, unitCount) {
    loopMs?.let { ClipTimeline(Styles.file, style.entranceMotif, feeling, style.cluster, unitCount, it) }
}
var previewMs by remember { mutableFloatStateOf(0f) }
val restartKey = listOf(emoji, feeling, lang, displayText, colors)
LaunchedEffect(timeline, restartKey, animationsOff) {
    val tl = timeline ?: return@LaunchedEffect
    if (animationsOff || pose != null) return@LaunchedEffect
    val start = withFrameNanos { it }
    while (true) withFrameNanos { previewMs = (((it - start) / 1_000_000f) % tl.durationMs.toFloat()) }
}
val activePose = pose ?: when {
    timeline == null -> null
    animationsOff -> ClipPose(0f, poster = true)
    else -> ClipPose(previewMs)
}
```

and feed `activePose` (instead of `pose`) to the three leaf composables from Task 6 Step 4. While `timeline == null` (Lottie still loading), show the poster frame, not a half-started clip, so t=0 begins when the clip is ready. Remove the old per-layer replay keys (`replayKey = Triple(...)`) and the `entranceMs = entranceTotalMs(...)` call in `Card`.

- [ ] **Step 2: Share-still path.** The non-export capture branch in `Card.kt` (`textClock.snapTo(TEXT_ANIM_DONE)`, `shimmerOn = false`, `staticEmoji = true`) must still produce a static finished card: set a local `forcePoster` state to `true` for the capture instead (feeding `ClipPose(0f, poster = true)`), then back to `false`; delete `shimmerOn`.

- [ ] **Step 3: Build and manual verification**

Run: `cd android && ./gradlew testDebugUnitTest && ./gradlew installDebug`
Check on device: (a) changing emoji, style (vertical swipe), and colors each restarts the entrance from the first frame; (b) the loop restarts text every cycle in sync with the emoji (compare with the exported GIF side by side); (c) with "Remove animations" enabled the card shows the finished frame; (d) jpg share still shows the finished card without shimmer; (e) typing doesn't stutter (previewMs state update every frame only recomposes the leaf canvases; if the whole `Card` recomposes, read `previewMs` inside a `derivedStateOf`/lambda to narrow recomposition and re-check).

- [ ] **Step 4: Commit**

```bash
git add android
git commit -m "Android preview loops the shared clip and restarts on any card change"
```

---

### Task 8: Cross-platform verification and cleanup

**Files:**
- Modify: `CLAUDE.md` (one index line for `clip.yml`/`clip.js`/`Clip.kt`), `todo.txt` (mark items 1-4 done)

- [ ] **Step 1: Full test run**

Run: `bun test tools/data/export-style.test.ts && cd web && npx vitest run && npm run build && cd ../android && ./gradlew testDebugUnitTest`
Expected: all PASS.

- [ ] **Step 2: Side-by-side check.** For the same text/emoji/style/colors on web and the Pixel: record the preview loop and the exported GIF/MP4 on each. Confirm: same clip length (±1 frame), same shimmer start, entrance finishes before shimmer, emoji loop count equal, poster frame first in exports only. Note remaining differences (font rendering, GIF 4% fast from centisecond rounding) in the PR description.

- [ ] **Step 3: Docs.** Add to `CLAUDE.md` (index style, no restated numbers): "`web/src/clip.yml` — shared preview/export clip rules (ground truth); mirrored by `web/src/clip.js` and `android/.../model/Clip.kt`, kept in sync by `web/src/clipFixture.json`."

- [ ] **Step 4: Commit and push to device**

```bash
git add CLAUDE.md todo.txt
git commit -m "Document shared clip timeline"
cd android && ./gradlew installDebug
```

---

## Self-review notes

- Spec coverage: clip spec (T1), poseAt/timeline per platform (T2, T3), preview web (T5) and Android (T7), exports (T4, T6), restart-on-any-change (T5 `playerKey`, T7 `restartKey`), testing/fixtures (T2, T3), perf measurement (T5 Step 7), reduced motion (T5, T7), spring unification (T1, T4, T6).
- Known soft spots to watch during execution: T4 Step 2 and T5 Step 4 touch large existing functions (`paintContent`, `Card.jsx`) — read the file first and keep unrelated behavior byte-identical; T6/T7 `Card.kt` edits depend on the exact current parameter list — read `Card.kt` fully before editing.
