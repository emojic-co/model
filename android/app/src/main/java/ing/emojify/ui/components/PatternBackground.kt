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

const val DEFAULT_PATTERN_OPACITY = 0.25f
private const val TILE_PX = 240

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
        val loader = ImageLoader(context)
        val request = ImageRequest.Builder(context)
            .data("file:///android_asset/patterns/$cluster.svg")
            .size(tilePx, tilePx)
            .build()
        val result = loader.execute(request)
        value = (result as? SuccessResult)?.image?.toBitmap()
    }
    bitmap?.let { bmp ->
        Canvas(modifier = modifier.fillMaxSize()) {
            drawIntoCanvas { canvas ->
                val shader = BitmapShader(bmp, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT)
                val paint = Paint().asFrameworkPaint().apply {
                    this.shader = shader
                    alpha = (opacity * 255).toInt()
                    colorFilter = android.graphics.PorterDuffColorFilter(tint.toArgb(), android.graphics.PorterDuff.Mode.SRC_IN)
                }
                canvas.nativeCanvas.drawRect(0f, 0f, size.width, size.height, paint)
            }
        }
    }
}
