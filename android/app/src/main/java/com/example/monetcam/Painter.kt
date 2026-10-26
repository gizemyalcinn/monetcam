package com.example.monetcam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Fırça darbeli boyama (Hertzmann 1998, "Painterly Rendering with Curved Brush Strokes").
 *
 * Referans görüntüyü (CycleGAN'ın Monet renkli çıktısı) boş bir tuvale, büyük fırçadan
 * küçüğe katman katman, gerçek fırça darbeleriyle yeniden boyar:
 *  - Her katmanda tuvalin referanstan hâlâ belirgin şekilde farklı olduğu yerlere darbe atılır.
 *  - Darbeler görüntüdeki kenarlar boyunca kıvrılır (parlaklık gradyanına dik yönde ilerler).
 *  - Her darbenin rengine küçük bir rastgele oynama eklenir: elle boyanmış hissi.
 *
 * Hız: bir fotoğrafta ~900 bin darbe var. Her katmanda önce tüm darbelerin yolu hesaplanır,
 * sonra görüntü yatay şeritlere bölünür ve her şerit ayrı bir çekirdekte çizilir. Her şerit
 * kendisine değen darbeleri aynı sırayla çizdiği için sonuç tek çekirdekle çizilenle birebir aynıdır.
 *
 * Python prototipi: style-transfer/paint.py (aynı algoritma ve ayarlar).
 */
object Painter {
    private val RADII = intArrayOf(4, 2, 1)   // fırça yarıçapları (piksel): kalından inceye
    private const val THRESHOLD = 18f         // bu kadar farklı olan bölgeler yeniden boyanır
    private const val MAX_LEN = 14            // bir darbedeki en fazla adım
    private const val MIN_LEN = 2             // bu kadar adımdan önce darbe kesilmez
    private const val CURVE = 0.7f            // 1: kenarı sıkı takip et, 0: düz git
    private const val JITTER = 8              // renk oynaması (0-255 ölçeğinde)
    private val CANVAS_COLOR = Color.rgb(245, 240, 230)

    private val THREADS = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
    private const val BANDS = 24              // şerit sayısı; çekirdekten fazla olması yükü dengeler
    private val pool: ExecutorService by lazy {
        // Düşük öncelikli iş parçacıkları: fırça 8 çekirdeği birden kullanırken ekranı çizen
        // ana iş parçacığı yine öne geçebilsin, halka ve yazılar takılmasın
        Executors.newFixedThreadPool(THREADS) { r ->
            Thread({
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                r.run()
            }, "painter").apply { isDaemon = true }
        }
    }

    /** body(0..n-1) işlerini çekirdeklere dağıtır ve hepsi bitene kadar bekler. */
    private fun parallel(n: Int, body: (Int) -> Unit) {
        val jobs = (0 until n).map { i -> pool.submit(Runnable { body(i) }) }
        jobs.forEach { it.get() }
    }

    /** @param onLayer (katman 0..2, toplam ilerleme 0..1): ilerleme çubuğu için, her katmanın başında ve sonunda */
    fun paint(ref: Bitmap, seed: Int = 0, onLayer: (Int, Float) -> Unit = { _, _ -> }): Bitmap {
        val w = ref.width
        val h = ref.height
        val n = w * h
        val px = IntArray(n)
        ref.getPixels(px, 0, w, 0, 0, w, h)
        val r0 = FloatArray(n) { ((px[it] shr 16) and 0xFF).toFloat() }
        val g0 = FloatArray(n) { ((px[it] shr 8) and 0xFF).toFloat() }
        val b0 = FloatArray(n) { (px[it] and 0xFF).toFloat() }

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(out).drawColor(CANVAS_COLOR)
        val rnd = Random(seed)
        val cur = IntArray(n)

        val tAll = SystemClock.elapsedRealtime()
        RADII.forEachIndexed { layer, radius ->
            onLayer(layer, layer / RADII.size.toFloat())
            val R = radius.toFloat()
            val tA = SystemClock.elapsedRealtime()
            // 1) Referansı fırça boyutuna göre bulanıklaştır: kalın fırça ince detayı görmez
            val sigma = R * 0.5f
            val blurred = arrayOfNulls<FloatArray>(3)
            parallel(3) { c -> blurred[c] = gaussian(arrayOf(r0, g0, b0)[c], w, h, sigma) }
            val rr = blurred[0]!!; val gg = blurred[1]!!; val bb = blurred[2]!!
            val lum = FloatArray(n) { 0.3f * rr[it] + 0.59f * gg[it] + 0.11f * bb[it] }

            // 2) Tuvalin şu anki hâli ile referans arasındaki fark
            out.getPixels(cur, 0, w, 0, 0, w, h)
            val diff = FloatArray(n) {
                val c = cur[it]
                val dr = ((c shr 16) and 0xFF) - rr[it]
                val dg = ((c shr 8) and 0xFF) - gg[it]
                val db = (c and 0xFF) - bb[it]
                sqrt(dr * dr + dg * dg + db * db)
            }

            // 3) Izgara üzerinde darbe başlangıç noktalarını seç
            val step = max(1, (R * 0.9f).toInt())
            val starts = ArrayList<Int>()
            var y = 0
            while (y < h) {
                var x = 0
                while (x < w) {
                    var sum = 0f; var best = -1f; var bestIdx = y * w + x; var cnt = 0
                    for (yy in y until min(y + step, h)) for (xx in x until min(x + step, w)) {
                        val k = yy * w + xx
                        val d = diff[k]
                        sum += d; cnt++
                        if (d > best) { best = d; bestIdx = k }
                    }
                    if (layer == 0 || sum / cnt > THRESHOLD) starts.add(bestIdx)
                    x += step
                }
                y += step
            }
            starts.shuffle(rnd)

            val tB = SystemClock.elapsedRealtime()
            // 4) Darbelerin yollarını ve renklerini hesapla (çizmeden, sadece noktalar)
            val count = starts.size
            var pts = FloatArray(count * 8)
            var np = 0
            val off = IntArray(count + 1)
            val col = IntArray(count)
            val yMin = FloatArray(count)
            val yMax = FloatArray(count)
            for (si in 0 until count) {
                val s = starts[si]
                if (np + 2 * (MAX_LEN + 1) > pts.size) pts = pts.copyOf(pts.size * 2)
                off[si] = np
                val sx = s % w
                val sy = s / w
                val cr = rr[s]; val cg = gg[s]; val cb = bb[s]
                pts[np++] = sx + 0.5f; pts[np++] = sy + 0.5f
                var lo = sy + 0.5f; var hi = lo
                var fx = sx.toFloat(); var fy = sy.toFloat()
                var ldx = 0f; var ldy = 0f
                for (i in 0 until MAX_LEN) {
                    val xi = fx.toInt(); val yi = fy.toInt(); val k = yi * w + xi
                    if (i > MIN_LEN) {
                        // tuval burada zaten darbe renginden daha doğruysa dur
                        val c = cur[k]
                        val canvasErr = abs(rr[k] - ((c shr 16) and 0xFF)) +
                                abs(gg[k] - ((c shr 8) and 0xFF)) +
                                abs(bb[k] - (c and 0xFF))
                        val strokeErr = abs(rr[k] - cr) + abs(gg[k] - cg) + abs(bb[k] - cb)
                        if (canvasErr < strokeErr) break
                    }
                    // parlaklık gradyanı (merkezi fark)
                    val gx = (lum[yi * w + min(xi + 1, w - 1)] - lum[yi * w + max(xi - 1, 0)]) * 0.5f
                    val gy = (lum[min(yi + 1, h - 1) * w + xi] - lum[max(yi - 1, 0) * w + xi]) * 0.5f
                    if (gx * gx + gy * gy < 1e-2f) break
                    var dx = -gy; var dy = gx                 // gradyana dik = kenar boyunca
                    if (ldx * dx + ldy * dy < 0) { dx = -dx; dy = -dy }
                    dx = CURVE * dx + (1 - CURVE) * ldx
                    dy = CURVE * dy + (1 - CURVE) * ldy
                    val len = sqrt(dx * dx + dy * dy) + 1e-6f
                    dx /= len; dy /= len
                    fx += R * dx; fy += R * dy
                    ldx = dx; ldy = dy
                    if (fx < 0 || fx >= w || fy < 0 || fy >= h) break
                    pts[np++] = fx + 0.5f; pts[np++] = fy + 0.5f
                    lo = min(lo, fy + 0.5f); hi = max(hi, fy + 0.5f)
                }
                yMin[si] = lo; yMax[si] = hi
                val jr = (cr + rnd.nextInt(-JITTER, JITTER + 1)).toInt().coerceIn(0, 255)
                val jg = (cg + rnd.nextInt(-JITTER, JITTER + 1)).toInt().coerceIn(0, 255)
                val jb = (cb + rnd.nextInt(-JITTER, JITTER + 1)).toInt().coerceIn(0, 255)
                col[si] = Color.rgb(jr, jg, jb)
            }
            off[count] = np
            val tC = SystemClock.elapsedRealtime()

            // Her darbeyi değdiği şeritlere önceden dağıt (sırayı koruyarak); böylece her şerit
            // 900 bin darbenin hepsini değil, sadece kendisine değenleri dolaşır.
            val bandOf = { yy: Float -> (yy.toInt() * BANDS / h).coerceIn(0, BANDS - 1) }
            val bandCount = IntArray(BANDS)
            for (i in 0 until count) {
                val b0 = bandOf(yMin[i] - R - 1); val b1 = bandOf(yMax[i] + R + 1)
                for (b in b0..b1) bandCount[b]++
            }
            val bandStrokes = Array(BANDS) { IntArray(bandCount[it]) }
            val fill = IntArray(BANDS)
            for (i in 0 until count) {
                val b0 = bandOf(yMin[i] - R - 1); val b1 = bandOf(yMax[i] + R + 1)
                for (b in b0..b1) bandStrokes[b][fill[b]++] = i
            }

            // 5) Şeritleri paralel çiz: her şerit tuvalin kendi parçasından başlar ve
            //    kendisine değen darbeleri orijinal sırayla çizer
            val bands = arrayOfNulls<Bitmap>(BANDS)
            parallel(BANDS) { b ->
                val y0 = b * h / BANDS
                val y1 = (b + 1) * h / BANDS
                val bmp = Bitmap.createBitmap(w, y1 - y0, Bitmap.Config.ARGB_8888)
                bmp.setPixels(cur, y0 * w, w, 0, 0, w, y1 - y0)
                val canvas = Canvas(bmp)
                canvas.translate(0f, -y0.toFloat())
                val brush = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                    strokeWidth = 2f * R
                }
                val path = Path()
                for (i in bandStrokes[b]) {
                    brush.color = col[i]
                    val a = off[i]; val e = off[i + 1]
                    if (e - a == 2) {
                        brush.style = Paint.Style.FILL
                        canvas.drawCircle(pts[a], pts[a + 1], R, brush)
                    } else {
                        brush.style = Paint.Style.STROKE
                        path.rewind()
                        path.moveTo(pts[a], pts[a + 1])
                        var j = a + 2
                        while (j < e) { path.lineTo(pts[j], pts[j + 1]); j += 2 }
                        canvas.drawPath(path, brush)
                    }
                }
                bands[b] = bmp
            }
            val tD = SystemClock.elapsedRealtime()
            val outCanvas = Canvas(out)
            for (b in 0 until BANDS) {
                val bmp = bands[b]!!
                outCanvas.drawBitmap(bmp, 0f, (b * h / BANDS).toFloat(), null)
                bmp.recycle()
            }
            onLayer(layer, (layer + 1) / RADII.size.toFloat())
            Log.d("MonetCam", "fırça R=$radius: $count darbe · hazırlık ${tB - tA} ms · yol ${tC - tB} ms · çizim ${tD - tC} ms · birleştirme ${SystemClock.elapsedRealtime() - tD} ms")
        }
        Log.d("MonetCam", "fırça toplam: ${SystemClock.elapsedRealtime() - tAll} ms ($THREADS çekirdek)")
        return out
    }

    /** Ayrılabilir Gauss bulanıklaştırma (önce yatay, sonra dikey). */
    private fun gaussian(src: FloatArray, w: Int, h: Int, sigma: Float): FloatArray {
        val rad = max(1, ceil(3 * sigma).toInt())
        val kernel = FloatArray(2 * rad + 1) { exp(-((it - rad) * (it - rad)) / (2 * sigma * sigma)) }
        val ks = kernel.sum()
        for (i in kernel.indices) kernel[i] /= ks
        val tmp = FloatArray(w * h)
        val dst = FloatArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var s = 0f
                for (k in -rad..rad) s += src[row + (x + k).coerceIn(0, w - 1)] * kernel[k + rad]
                tmp[row + x] = s
            }
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (k in -rad..rad) s += tmp[(y + k).coerceIn(0, h - 1) * w + x] * kernel[k + rad]
            dst[y * w + x] = s
        }
        return dst
    }
}
