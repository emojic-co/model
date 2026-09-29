package ing.emojify.model

import android.graphics.Bitmap
import java.io.File
import java.io.OutputStream

// Offline animated-GIF encoder: collect frames, then finish() builds one adaptive 256-color palette
// (median cut over all frames) and writes looping frames with ordered dithering. Ordered dither is
// stable across frames, so static areas (gradient, pattern) don't shimmer.
class GifEncoder(private val file: File, private val size: Int, private val delayCs: Int) {
    private val frames = ArrayList<IntArray>()

    fun addFrame(bitmap: Bitmap) {
        val readable = if (bitmap.config == Bitmap.Config.HARDWARE) bitmap.copy(Bitmap.Config.ARGB_8888, false) else bitmap
        val scaled = Bitmap.createScaledBitmap(readable, size, size, true)
        val px = IntArray(size * size)
        scaled.getPixels(px, 0, size, 0, 0, size, size)
        if (scaled !== readable) scaled.recycle()
        if (readable !== bitmap) readable.recycle()
        frames.add(px)
    }

    fun finish() {
        val palette = buildPalette()
        val lut = buildLut(palette)
        file.outputStream().buffered().use { out ->
            writeHeader(out, palette)
            for (px in frames) writeFrame(out, indexFrame(px, lut))
            out.write(0x3B)
        }
        frames.clear()
    }

    private fun buildPalette(): IntArray {
        val hist = IntArray(1 shl 15)
        for ((f, px) in frames.withIndex()) {
            if (f % 3 != 0) continue
            var i = 0
            while (i < px.size) { hist[key15(px[i])]++; i += 5 }
        }
        val colors = (0 until hist.size).filter { hist[it] > 0 }
        var boxes = mutableListOf(colors)
        while (boxes.size < 256) {
            val splittable = boxes.filter { it.size > 1 }
            if (splittable.isEmpty()) break
            val box = splittable.maxBy { b -> b.sumOf { hist[it].toLong() } * range(b).second }
            val (channel, _) = range(box)
            val sorted = box.sortedBy { channelOf(it, channel) }
            val half = sorted.sumOf { hist[it].toLong() } / 2
            var acc = 0L
            var cut = 1
            for ((i, c) in sorted.withIndex()) {
                acc += hist[c]
                if (acc >= half) { cut = (i + 1).coerceIn(1, sorted.size - 1); break }
            }
            boxes.remove(box)
            boxes.add(sorted.subList(0, cut))
            boxes.add(sorted.subList(cut, sorted.size))
        }
        val palette = IntArray(256)
        for ((i, b) in boxes.withIndex()) {
            var w = 0L; var r = 0L; var g = 0L; var bl = 0L
            for (c in b) {
                val n = hist[c].toLong()
                w += n; r += n * channelOf(c, 0); g += n * channelOf(c, 1); bl += n * channelOf(c, 2)
            }
            fun ch(sum: Long) = ((sum / w) * 255 / 31).toInt()
            palette[i] = (ch(r) shl 16) or (ch(g) shl 8) or ch(bl)
        }
        return palette
    }

    // channel 0=r 1=g 2=b of a 15-bit key (5 bits each)
    private fun channelOf(key: Int, channel: Int) = (key shr (10 - 5 * channel)) and 31
    private fun key15(argb: Int) = ((argb shr 19 and 31) shl 10) or ((argb shr 11 and 31) shl 5) or (argb shr 3 and 31)

    private fun range(box: List<Int>): Pair<Int, Int> {
        var best = 0; var bestRange = -1
        for (ch in 0..2) {
            var lo = 31; var hi = 0
            for (c in box) { val v = channelOf(c, ch); if (v < lo) lo = v; if (v > hi) hi = v }
            if (hi - lo > bestRange) { bestRange = hi - lo; best = ch }
        }
        return best to bestRange
    }

    private fun buildLut(palette: IntArray): ByteArray {
        val lut = ByteArray(1 shl 15)
        for (key in lut.indices) {
            val r = channelOf(key, 0) * 255 / 31
            val g = channelOf(key, 1) * 255 / 31
            val b = channelOf(key, 2) * 255 / 31
            var best = 0; var bestD = Int.MAX_VALUE
            for ((i, p) in palette.withIndex()) {
                val dr = r - (p shr 16 and 255); val dg = g - (p shr 8 and 255); val db = b - (p and 255)
                val d = dr * dr + dg * dg + db * db
                if (d < bestD) { bestD = d; best = i }
            }
            lut[key] = best.toByte()
        }
        return lut
    }

    private fun indexFrame(px: IntArray, lut: ByteArray): ByteArray {
        val out = ByteArray(px.size)
        for (y in 0 until size) for (x in 0 until size) {
            val p = px[y * size + x]
            val d = (BAYER[(y and 7) * 8 + (x and 7)] - 31.5f) * DITHER
            val r = ((p shr 16 and 255) + d).toInt().coerceIn(0, 255)
            val g = ((p shr 8 and 255) + d).toInt().coerceIn(0, 255)
            val b = ((p and 255) + d).toInt().coerceIn(0, 255)
            out[y * size + x] = lut[((r shr 3) shl 10) or ((g shr 3) shl 5) or (b shr 3)]
        }
        return out
    }

    private fun writeHeader(out: OutputStream, palette: IntArray) {
        out.write("GIF89a".toByteArray())
        out.writeShort(size); out.writeShort(size)
        out.write(0xF7); out.write(0); out.write(0) // global palette, 256 colors
        for (c in palette) { out.write(c shr 16 and 255); out.write(c shr 8 and 255); out.write(c and 255) }
        // NETSCAPE loop extension: loop forever
        out.write(byteArrayOf(0x21, 0xFF.toByte(), 0x0B) + "NETSCAPE2.0".toByteArray() + byteArrayOf(3, 1, 0, 0, 0))
    }

    private fun writeFrame(out: OutputStream, indices: ByteArray) {
        out.write(byteArrayOf(0x21, 0xF9.toByte(), 4, 0x04)) // graphic control: dispose = do not dispose
        out.writeShort(delayCs); out.write(0); out.write(0)
        out.write(0x2C)
        out.writeShort(0); out.writeShort(0); out.writeShort(size); out.writeShort(size)
        out.write(0)
        out.write(8)
        val data = lzw(indices)
        var i = 0
        while (i < data.size) {
            val n = minOf(255, data.size - i)
            out.write(n); out.write(data, i, n)
            i += n
        }
        out.write(0)
    }

    private fun OutputStream.writeShort(v: Int) { write(v and 255); write(v shr 8 and 255) }

    private fun lzw(indices: ByteArray): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        var bitBuf = 0; var bitCnt = 0
        fun emit(code: Int, width: Int) {
            bitBuf = bitBuf or (code shl bitCnt); bitCnt += width
            while (bitCnt >= 8) { out.write(bitBuf and 255); bitBuf = bitBuf ushr 8; bitCnt -= 8 }
        }
        val clear = 256; val eoi = 257
        val hashKeys = IntArray(HASH); val hashVals = IntArray(HASH)
        fun reset() { hashKeys.fill(-1) }
        reset()
        var next = 258; var width = 9
        emit(clear, width)
        var prefix = indices[0].toInt() and 255
        for (i in 1 until indices.size) {
            val c = indices[i].toInt() and 255
            val key = (prefix shl 8) or c
            var h = (key * 2654435761L ushr 8).toInt() % HASH
            if (h < 0) h += HASH
            var found = -1
            while (hashKeys[h] != -1) {
                if (hashKeys[h] == key) { found = hashVals[h]; break }
                h = (h + 1) % HASH
            }
            if (found >= 0) { prefix = found; continue }
            emit(prefix, width)
            if (next < 4096) {
                hashKeys[h] = key; hashVals[h] = next++
                if (next > (1 shl width) && width < 12) width++
            } else {
                emit(clear, width); reset(); next = 258; width = 9
            }
            prefix = c
        }
        emit(prefix, width)
        emit(eoi, width)
        if (bitCnt > 0) out.write(bitBuf and 255)
        return out.toByteArray()
    }

    private companion object {
        const val HASH = 8209
        const val DITHER = 0.125f
        val BAYER = FloatArray(64).also { m ->
            val b = intArrayOf(
                0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
                12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
                3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
                15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
            )
            for (i in 0 until 64) m[i] = b[i].toFloat()
        }
    }
}
