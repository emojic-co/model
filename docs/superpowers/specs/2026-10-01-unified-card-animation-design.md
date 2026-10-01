# Unified card animation: preview and export from one timeline

Date: 2026-10-01. Source: `todo.txt` items 1-4 (web/android animation, preview and export parity).

## Goals

1. Any card change (emoji, style, language, text, colors) restarts the full animation from t=0.
2. Web and Android use the same animation rules and export (nearly) identical media.
3. Web and Android previews look the same.
4. Exported GIF/MP4 matches the preview.

Decisions made with the user: the preview loops the export clip (what you see is what you export);
the web preview moves to a canvas painted from the shared pose function (Approach A).

## Non-goals

- Changing text animation, shimmer, pattern or font definitions (`textAnimations.yml`, `shimmers.yml`).
- Fixing the GIF centisecond rounding (12 fps stores as 8 cs, about 4% fast on both platforms). Known limit.
- Selectable card text on web (lost with the canvas preview; accepted).

## Design

### 1. Shared clip spec

New `web/src/clip.yml` is the ground truth, copied into `style.yml` (via `tools/data/export-style.ts`) so Android
reads it too, like `textAnimations`. It holds rules only, never per-card numbers:

- Clip length `L` = text entrance total + shimmer `startDelayMs` + one shimmer cycle (pass + pause), rounded up to a
  whole number of emoji loops, with at least `minEmojiLoops` (2) loops, so the loop is seamless.
- `maxClipMs`: safety cap; if exceeded, use the fewest loops that fit.
- No Lottie clone for the emoji: the fallback spring plays once at t=0; `L` is the text + shimmer length.
- `posterHoldMs` (600): export-only first frame showing the finished card. Not part of the looping preview.
- Export frame rates: `gifFps` 12, `mp4Fps` 20.

### 2. `poseAt(t)` (one per platform, same math)

Returns the text unit poses (existing `scheduleFor` sampled at `t`), the shimmer pass progress (null before its
start time), and the emoji loop phase `(t mod loopMs) / loopMs`. Web: JS next to `cardAnim.js`. Android: Kotlin
next to `TextAnimation.kt`/`Shimmer.kt` (builds on the existing `ExportPose`).

### 3. Preview

One clock `t` loops over `[0, L)` and resets to 0 whenever any card input changes (replay is owned by this
clock, replacing the per-layer replay keys).

- Android: feed the clock into the existing `ExportPose` path; remove the independent live Lottie/shimmer
  effects in preview.
- Web: replace DOM `CharText`, `CardShimmer`, `AnimatedEmoji` with a canvas painted each frame from `poseAt(t)`.
  Layout (font fit, line wrap, per-unit x positions) is computed once per card change and cached, not per frame
  (today's `createPainter` recomputes it every paint). Share buttons stay DOM overlays.
- Reduced motion (web `prefers-reduced-motion`, Android animator scale 0): freeze on the finished frame.

### 4. Export

Samples the same `poseAt(t)` over `[0, L)` at `gifFps`/`mp4Fps`, preceded by the poster frame, on both platforms.
Web and Android share the clip structure; both previously differed (web: no poster, one emoji loop; Android: poster,
2-6 loops).

### 5. Testing

- Fixture tests: for a few cards at fixed timestamps, web and Android `poseAt` produce the same values.
- Web: the preview painter and export painter are the same code, with a test that equal `t` gives equal pixels.
- Measure web frame time on a mid-range phone before shipping; keep per-frame work to draws only.

## Risks

- Canvas preview on low-end phones: mitigated by layout caching; verify by measurement.
- Android preview change removes the live Lottie composition path; verify emoji fallbacks and the
  reduced-motion setting still behave.
- Large cross-platform change: implement web and Android behind the same fixtures so drift is caught.
