package ing.emojify.model

import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File

// Offline H.264 MP4 encoder: frames are scaled to size x size, converted to YUV420 and fed with explicit
// presentation times, so per-frame durations (e.g. the held poster frame) are exact.
class Mp4Encoder(file: File, private val size: Int, fps: Int) {
    private val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val muxer = MediaMuxer(file.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private val info = MediaCodec.BufferInfo()
    private var track = -1
    private var ptsUs = 0L

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, size, size).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, size * size * 30)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    fun addFrame(bitmap: Bitmap, durationMs: Int) {
        val readable = if (bitmap.config == Bitmap.Config.HARDWARE) bitmap.copy(Bitmap.Config.ARGB_8888, false) else bitmap
        val scaled = Bitmap.createScaledBitmap(readable, size, size, true)
        val px = IntArray(size * size)
        scaled.getPixels(px, 0, size, 0, 0, size, size)
        if (scaled !== readable) scaled.recycle()
        if (readable !== bitmap) readable.recycle()
        val index = dequeueInput()
        fill(codec.getInputImage(index)!!, px)
        codec.queueInputBuffer(index, 0, size * size * 3 / 2, ptsUs, 0)
        ptsUs += durationMs * 1000L
        drain(EOS = false)
    }

    fun finish() {
        val index = dequeueInput()
        codec.queueInputBuffer(index, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        drain(EOS = true)
        codec.stop()
        codec.release()
        if (track >= 0) muxer.stop()
        muxer.release()
    }

    private fun dequeueInput(): Int {
        while (true) {
            val i = codec.dequeueInputBuffer(10_000)
            if (i >= 0) return i
            drain(EOS = false)
        }
    }

    private fun drain(EOS: Boolean) {
        while (true) {
            val i = codec.dequeueOutputBuffer(info, if (EOS) 10_000 else 0)
            when {
                i == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!EOS) return
                i == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    track = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                }
                i >= 0 -> {
                    val data = codec.getOutputBuffer(i)!!
                    if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && track >= 0) {
                        data.position(info.offset)
                        data.limit(info.offset + info.size)
                        muxer.writeSampleData(track, data, info)
                    }
                    codec.releaseOutputBuffer(i, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    // BT.601 limited range; chroma is the average of each 2x2 block.
    private fun fill(image: Image, px: IntArray) {
        val (yp, up, vp) = image.planes
        for (y in 0 until size) for (x in 0 until size) {
            val c = px[y * size + x]
            val r = c shr 16 and 0xFF; val g = c shr 8 and 0xFF; val b = c and 0xFF
            yp.buffer.put(y * yp.rowStride + x * yp.pixelStride, (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).toByte())
        }
        for (y in 0 until size / 2) for (x in 0 until size / 2) {
            var r = 0; var g = 0; var b = 0
            for (dy in 0..1) for (dx in 0..1) {
                val c = px[(2 * y + dy) * size + 2 * x + dx]
                r += c shr 16 and 0xFF; g += c shr 8 and 0xFF; b += c and 0xFF
            }
            r /= 4; g /= 4; b /= 4
            up.buffer.put(y * up.rowStride + x * up.pixelStride, (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).toByte())
            vp.buffer.put(y * vp.rowStride + x * vp.pixelStride, (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).toByte())
        }
    }
}
