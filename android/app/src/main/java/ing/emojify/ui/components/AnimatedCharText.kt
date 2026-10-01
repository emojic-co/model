package ing.emojify.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Constraints
import ing.emojify.model.TextAnimations
import ing.emojify.model.TextSchedule
import ing.emojify.model.splitTextUnits
import ing.emojify.model.textScheduleFor

/** Elapsed-time value meaning "animation finished"; also what a frozen/static render uses. */
const val TEXT_ANIM_DONE = 1_000_000f

/**
 * Draws [text] with the style's per-character entrance. Each unit is the full text layout clipped
 * to that unit's box and transformed on its own, so shaping/kerning stay exact. [textMs] is read in
 * the draw phase only: ms into the clip's text entrance ([TEXT_ANIM_DONE] = finished).
 */
@Composable
fun AnimatedCharText(
    text: String,
    textStyle: TextStyle,
    color: Color,
    maxWidthPx: Int,
    animations: TextAnimations?,
    motif: String,
    feeling: String?,
    textMs: () -> Float,
    modifier: Modifier = Modifier,
) {
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    val layout = remember(text, textStyle, maxWidthPx) {
        measurer.measure(text, textStyle, constraints = Constraints(minWidth = maxWidthPx, maxWidth = maxWidthPx))
    }
    val units = remember(text, layout) { unitPaths(text, layout, textStyle.fontSize.value) }
    val schedule: TextSchedule? = remember(animations, motif, feeling, units.size) {
        if (animations == null || units.isEmpty()) null else textScheduleFor(animations, motif, feeling, units.size)
    }

    val widthDp = with(density) { maxWidthPx.toDp() }
    val heightDp = with(density) { layout.size.height.toDp() }
    Canvas(modifier.width(widthDp).height(heightDp).semantics { contentDescription = text }) {
        val elapsed = textMs().toDouble()
        val s = schedule
        if (s == null || elapsed >= TEXT_ANIM_DONE) {
            drawText(layout, color = color)
            return@Canvas
        }
        val em = textStyle.fontSize.toPx()
        // Finished units render in one pass; unfinished ones are excluded and drawn individually.
        val pending = Path()
        var anyPending = false
        units.forEachIndexed { i, u ->
            if (!s.isDone(i, elapsed)) {
                pending.addPath(u.path)
                anyPending = true
            }
        }
        if (anyPending) clipPath(pending, ClipOp.Difference) { drawText(layout, color = color) } else drawText(layout, color = color)
        units.forEachIndexed { i, u ->
            if (s.isDone(i, elapsed)) return@forEachIndexed
            val pose = s.poseAt(i, elapsed)
            if (pose.opacity <= 0f) return@forEachIndexed
            withTransform({
                translate(u.pivot.x + pose.x * em, u.pivot.y + pose.y * em)
                rotate(pose.rotate, androidx.compose.ui.geometry.Offset.Zero)
                scale(pose.scale, pose.scale * pose.scaleY, androidx.compose.ui.geometry.Offset.Zero)
                translate(-u.pivot.x, -u.pivot.y)
            }) {
                clipPath(u.path) { drawText(layout, color = color, alpha = pose.opacity) }
            }
        }
    }
}

private class UnitShape(val path: Path, val pivot: androidx.compose.ui.geometry.Offset)

// Each line is tiled by its units' boxes (split at the midpoints between neighbours, outer edges
// extended by an em), so glyph overhang — common in handwritten fonts — always belongs to some unit
// instead of being left drawn, un-animated, outside every clip box.
private fun unitPaths(text: String, layout: TextLayoutResult, fontSizePx: Float): List<UnitShape> {
    val pad = fontSizePx * 0.15f
    val units = splitTextUnits(text).flatten()
    // line -> (unit index, tight box), then widened per line below.
    val perLine = HashMap<Int, MutableList<Pair<Int, Rect>>>()
    val pivots = arrayOfNulls<androidx.compose.ui.geometry.Offset>(units.size)
    units.forEachIndexed { i, u ->
        val boxes = LinkedHashMap<Int, Rect>()
        for (o in u.start until u.end) {
            val box = layout.getBoundingBox(o)
            val line = layout.getLineForOffset(o)
            boxes[line] = boxes[line]?.let {
                Rect(minOf(it.left, box.left), minOf(it.top, box.top), maxOf(it.right, box.right), maxOf(it.bottom, box.bottom))
            } ?: box
        }
        pivots[i] = boxes.values.first().center
        boxes.forEach { (line, box) -> perLine.getOrPut(line) { ArrayList() }.add(i to box) }
    }
    val paths = List(units.size) { Path() }
    perLine.values.forEach { entries ->
        entries.sortBy { it.second.left }
        entries.forEachIndexed { k, (i, box) ->
            val left = if (k == 0) box.left - fontSizePx else (entries[k - 1].second.right + box.left) / 2
            val right = if (k == entries.lastIndex) box.right + fontSizePx else (box.right + entries[k + 1].second.left) / 2
            paths[i].addRect(Rect(left, box.top - pad, right, box.bottom + pad))
        }
    }
    return units.indices.map { UnitShape(paths[it], pivots[it]!!) }
}
