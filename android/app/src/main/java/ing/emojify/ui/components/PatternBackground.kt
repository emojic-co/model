package ing.emojify.ui.components

import android.graphics.BitmapShader
import android.graphics.Shader
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import coil3.ImageLoader
import coil3.request.ImageRequest
import coil3.request.SuccessResult
import coil3.toBitmap
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

const val DEFAULT_PATTERN_OPACITY = 0.25f
private const val TILE_PX = 240

private suspend fun loadTile(context: android.content.Context, source: String, widthPx: Int, heightPx: Int): android.graphics.Bitmap? {
    val loader = ImageLoader(context)
    val request = ImageRequest.Builder(context)
        .data(source)
        .size(widthPx.coerceAtLeast(1), heightPx.coerceAtLeast(1))
        .build()
    val result = loader.execute(request)
    return (result as? SuccessResult)?.image?.toBitmap()
}

private fun drawTiled(scope: androidx.compose.ui.graphics.drawscope.DrawScope, bmp: android.graphics.Bitmap, tint: androidx.compose.ui.graphics.Color, opacity: Float) {
    scope.drawIntoCanvas { canvas ->
        val shader = BitmapShader(bmp, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT)
        val paint = Paint().asFrameworkPaint().apply {
            this.shader = shader
            alpha = (opacity * 255).toInt()
            colorFilter = android.graphics.PorterDuffColorFilter(tint.toArgb(), android.graphics.PorterDuff.Mode.SRC_IN)
        }
        canvas.nativeCanvas.drawRect(0f, 0f, scope.size.width, scope.size.height, paint)
    }
}

// Decorative, cluster-level background (e.g. MainScreen's page background) — bundled per-cluster assets.
@Composable
fun PatternBackground(
    cluster: String,
    tint: androidx.compose.ui.graphics.Color,
    modifier: Modifier = Modifier,
    opacity: Float = DEFAULT_PATTERN_OPACITY,
    tilePx: Int = TILE_PX,
) {
    val context = LocalContext.current
    val bitmap by produceState<android.graphics.Bitmap?>(initialValue = null, cluster, tilePx) {
        value = loadTile(context, "file:///android_asset/patterns/$cluster.svg", tilePx, tilePx)
    }
    bitmap?.let { bmp -> Canvas(modifier = modifier.fillMaxSize()) { drawTiled(this, bmp, tint, opacity) } }
}

// Per-feeling card background, rendered from the literal SVG synced from style.yml (see
// ing.emojify.model.Styles) so the card pattern matches the web app exactly.
@Composable
fun FeelingPatternBackground(
    feeling: String,
    svg: String,
    tint: androidx.compose.ui.graphics.Color,
    modifier: Modifier = Modifier,
    opacity: Float = DEFAULT_PATTERN_OPACITY,
    tileWidthPx: Int = TILE_PX,
    tileHeightPx: Int = TILE_PX,
) {
    val context = LocalContext.current
    val bitmap by produceState<android.graphics.Bitmap?>(initialValue = null, feeling, tileWidthPx, tileHeightPx) {
        val file = withContext(Dispatchers.IO) {
            val dir = File(context.cacheDir, "patterns").apply { mkdirs() }
            val f = File(dir, "$feeling.svg")
            if (!f.exists()) f.writeText(svg)
            f
        }
        value = loadTile(context, file.toURI().toString(), tileWidthPx, tileHeightPx)
    }
    bitmap?.let { bmp -> Canvas(modifier = modifier.fillMaxSize()) { drawTiled(this, bmp, tint, opacity) } }
}
