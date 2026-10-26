package com.example.monetcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

// 20 parçaya bölünmüş GPU uyumlu model (style-transfer/export_cyclegan_segments.py); girdi/çıktı NHWC: 1x896x672x3
const val MODEL_FILE = "monet_cyclegan_seg_896x672.tflite"
const val IN_W = 672
const val IN_H = 896

/**
 * Modeli çalıştıran sınıf.
 *
 * Model tek dosyada ama 20 parçaya (imza) bölünmüş durumda (style-transfer/export_cyclegan_segments.py):
 * kodlayıcı, 9 bloğun her birinin iki yarısı ve iki büyütme adımı. Tek parça hâlinde GPU'ya verilince
 * GPU ~6 saniye boyunca ekrana hiç sıra vermiyor ve ekran donuyordu; parçalar arasında kısa bir ara
 * vererek ekranın kare çizmesine fırsat tanıyoruz. Ekranda sürekli dönen bir halka yerine her parçadan
 * sonra bir adım ilerleyen bir ilerleme çubuğu gösteriliyor (onProgress): GPU meşgulken sürekli
 * animasyon akıcı olamıyor, adım adım ilerleyen bir çubuk ise doğal görünüyor.
 *
 * Donanım seçimi: GPU (tam hassasiyet) önce denenir; bilinen bir test deseniyle çalıştırılıp sonuç,
 * bilgisayarda hesaplanmış doğru cevapla karşılaştırılır. Tutmazsa CPU kullanılır.
 */
class MonetStylizer(private val context: Context) : AutoCloseable {

    /**
     * Denenecek yapılandırmalar, en hızlıdan en güvenliye.
     * GPU fp16 (yarım hassasiyet) bilerek yok: bu modelde ara değerler fp16 sınırını aşıyor ve
     * Redmi Note 13 Pro'nun GPU'sunda sonuç beyaz/siyah çıkıyordu (test hatası 0,40).
     */
    private enum class Mode(val label: String) {
        GPU_FP32("GPU fp32"), CPU("CPU")
    }

    /** Parçaların çalışma sırası. */
    private val signatures = listOf("enc") + (0 until 9).flatMap { listOf("r${it}a", "r${it}b") } + listOf("dec1", "dec2")

    /** Test deseninin 8x8 noktadaki doğru çıktısı (bilgisayarda CPU ile hesaplandı).
     *  init bloğundan önce tanımlı olmalı: Kotlin özellikleri yukarıdan aşağı sırayla başlatır. */
    private val expected = floatArrayOf(0.668f,0.612f,0.762f,0.677f,0.627f,0.734f,0.672f,0.584f,0.715f,0.657f,0.6f,0.647f,0.66f,0.593f,0.568f,0.648f,0.554f,0.531f,0.642f,0.558f,0.446f,0.663f,0.541f,0.388f,0.707f,0.477f,0.599f,0.677f,0.718f,0.652f,0.588f,0.465f,0.628f,0.717f,0.668f,0.573f,0.662f,0.566f,0.599f,0.594f,0.544f,0.54f,0.716f,0.673f,0.53f,0.645f,0.407f,0.528f,0.565f,0.425f,0.4f,0.691f,0.664f,0.386f,0.714f,0.425f,0.399f,0.583f,0.633f,0.462f,0.638f,0.557f,0.546f,0.726f,0.532f,0.598f,0.634f,0.696f,0.767f,0.654f,0.52f,0.74f,0.731f,0.639f,0.783f,0.665f,0.606f,0.861f,0.539f,0.585f,0.862f,0.73f,0.619f,0.694f,0.659f,0.578f,0.602f,0.538f,0.566f,0.537f,0.75f,0.575f,0.393f,0.704f,0.552f,0.344f,0.521f,0.556f,0.451f,0.682f,0.398f,0.401f,0.769f,0.674f,0.431f,0.495f,0.487f,0.464f,0.645f,0.618f,0.554f,0.763f,0.642f,0.577f,0.466f,0.519f,0.681f,0.67f,0.718f,0.709f,0.765f,0.679f,0.497f,0.653f,0.445f,0.518f,0.527f,0.656f,0.602f,0.762f,0.501f,0.513f,0.667f,0.602f,0.568f,0.496f,0.618f,0.652f,0.775f,0.501f,0.559f,0.703f,0.708f,0.624f,0.524f,0.614f,0.79f,0.646f,0.613f,0.799f,0.765f,0.597f,0.688f,0.458f,0.547f,0.762f,0.663f,0.603f,0.6f,0.769f,0.584f,0.527f,0.456f,0.509f,0.492f,0.657f,0.586f,0.392f,0.78f,0.495f,0.346f,0.712f,0.632f,0.339f,0.474f,0.412f,0.356f,0.779f,0.63f,0.426f,0.665f,0.558f,0.505f,0.457f,0.529f,0.659f,0.767f,0.69f,0.662f,0.658f,0.522f,0.781f)

    val accelerator: String
    private val model: CompiledModel
    private val ins: Map<String, List<TensorBuffer>>
    private val outs: Map<String, List<TensorBuffer>>

    /** true: bir parçanın çıktı tamponunu doğrudan sıradakine veriyoruz (kopyasız, hızlı). */
    private var directHandoff = true

    init {
        // Önceki sürümlerin kaydettiği GPU program önbelleğini sil (yüklenirken takılıyordu)
        context.cacheDir.listFiles()?.filter { "monet_gpu" in it.name }?.forEach { it.deleteRecursively() }

        var chosen: Triple<Mode, CompiledModel, Pair<Map<String, List<TensorBuffer>>, Map<String, List<TensorBuffer>>>>? = null
        for (mode in Mode.entries) {
            val t0 = SystemClock.elapsedRealtime()
            Log.d("MonetCam", "${mode.label}: model derleniyor…")
            val m = try {
                CompiledModel.create(context.assets, MODEL_FILE, options(mode))
            } catch (e: Exception) {
                Log.w("MonetCam", "${mode.label} açılamadı: ${e.message}")
                continue
            }
            val i = signatures.associateWith { m.createInputBuffers(it) }
            val o = signatures.associateWith { m.createOutputBuffers(it) }
            if (mode == Mode.CPU) { chosen = Triple(mode, m, i to o); break }
            Log.d("MonetCam", "${mode.label}: derlendi (${SystemClock.elapsedRealtime() - t0} ms), test çalıştırılıyor…")
            val test = {
                try {
                    selfTestError(runPipeline(m, i, o, selfTestInput()))
                } catch (e: Exception) {
                    Log.w("MonetCam", "${mode.label} test sırasında hata: ${e.message}"); 1f
                }
            }
            var err = test()
            if (err >= 0.05f && directHandoff) {
                // Tamponları doğrudan paylaşmak bu cihazda yanlış sonuç veriyorsa kopyalayarak bir kez daha dene
                Log.w("MonetCam", "doğrudan aktarımla test tutmadı (hata %.4f), kopyalayarak deneniyor".format(err))
                directHandoff = false
                err = test()
            }
            val ms = SystemClock.elapsedRealtime() - t0
            if (err < 0.05f) {
                Log.d("MonetCam", "${mode.label} test BAŞARILI (hata %.4f, %d ms)".format(err, ms))
                chosen = Triple(mode, m, i to o); break
            }
            Log.w("MonetCam", "${mode.label} test BAŞARISIZ (hata %.4f, %d ms), sıradakine geçiliyor".format(err, ms))
            i.values.flatten().forEach { it.close() }; o.values.flatten().forEach { it.close() }; m.close()
        }
        val c = chosen ?: error("Model hiçbir donanımda açılamadı")
        accelerator = c.first.label
        model = c.second
        ins = c.third.first
        outs = c.third.second
        Log.d("MonetCam", "Model hızlandırıcı: $accelerator")
    }

    private fun options(mode: Mode): CompiledModel.Options = when (mode) {
        Mode.CPU -> CompiledModel.Options(Accelerator.CPU)
        else -> CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
            // GPU + CPU birlikte: GPU'nun desteklemediği bir işlem olursa sadece o parça CPU'da çalışır
            // Normal öncelik (en hızlı). Yükleme ekranında hareketli gösterge olmadığı için
            // modelin GPU'yu birkaç saniye meşgul etmesi görünür bir takılma yaratmıyor.
            gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
    }

    /**
     * Parçaları sırayla çalıştırır. Her parçadan sonra kısa bir ara verilir: bu arada GPU boşta
     * kaldığı için ekran yeni kare çizebilir (yükleme halkası ve yazılar akmaya devam eder).
     */
    private fun runPipeline(
        m: CompiledModel,
        i: Map<String, List<TensorBuffer>>,
        o: Map<String, List<TensorBuffer>>,
        data: FloatArray,
        onStep: (Float) -> Unit = {}
    ): FloatArray {
        val total = signatures.size.toFloat()
        var done = 0
        val tick = { done++; onStep(done / total) }
        i.getValue("enc")[0].writeFloat(data)
        step(m, "enc", i.getValue("enc"), o); tick()
        var h: TensorBuffer = o.getValue("enc")[0]
        for (b in 0 until 9) {
            val y = step(m, "r${b}a", listOf(h), o, i); tick()
            h = step(m, "r${b}b", listOf(y, h), o, i); tick()
        }
        val d1 = step(m, "dec1", listOf(h), o, i); tick()
        step(m, "dec2", listOf(d1), o, i); tick()
        return o.getValue("dec2")[0].readFloat()
    }

    /** Bir parçayı çalıştırır ve çıktı tamponunu döndürür. */
    private fun step(
        m: CompiledModel, sig: String, inputs: List<TensorBuffer>,
        o: Map<String, List<TensorBuffer>>, i: Map<String, List<TensorBuffer>>? = null
    ): TensorBuffer {
        val out = o.getValue(sig)
        var done = false
        if (directHandoff || i == null) {
            try {
                m.run(inputs, out, sig)
                done = true
            } catch (e: Exception) {
                // Bu cihazda tamponlar parçalar arasında doğrudan paylaşılamıyorsa kopyalayarak devam et
                Log.w("MonetCam", "doğrudan aktarım olmadı (${e.message}), kopyalanarak devam ediliyor")
                directHandoff = false
            }
        }
        if (!done && i != null) {
            val own = i.getValue(sig)
            inputs.forEachIndexed { k, buf -> own[k].writeFloat(buf.readFloat()) }
            m.run(own, out, sig)
        }
        Thread.sleep(12)   // ekrana bir kare çizme fırsatı
        return out[0]
    }

    /** Test deseni: her renk kanalında farklı sinüs dalgaları (Python'daki referansla birebir aynı formül). */
    private fun selfTestInput(): FloatArray {
        val data = FloatArray(3 * IN_W * IN_H)
        var k = 0
        for (y in 0 until IN_H) for (x in 0 until IN_W) {
            data[k++] = (0.5 + 0.4 * sin(0.05 * x) * cos(0.03 * y)).toFloat()
            data[k++] = (0.5 + 0.4 * sin(0.031 * x + 1) * cos(0.047 * y + 2)).toFloat()
            data[k++] = (0.5 + 0.4 * sin(0.07 * x + 2) * cos(0.021 * y + 1)).toFloat()
        }
        return data
    }

    /** Modelin test çıktısı ile doğru cevap arasındaki ortalama fark (0 = birebir aynı, NaN ise 1). */
    private fun selfTestError(out: FloatArray): Float {
        var sum = 0f
        var k = 0
        for (a in 0 until 8) for (b in 0 until 8) {
            val y = 56 + 112 * a
            val x = 42 + 84 * b
            for (c in 0 until 3) {
                val v = out[(y * IN_W + x) * 3 + c]
                if (v.isNaN()) return 1f
                sum += abs(v - expected[k++])
            }
        }
        return sum / k
    }

    /**
     * Fotoğrafı Monet tarzına çevirir: önce CycleGAN renkleri verir, sonra Painter fırça darbeleriyle boyar.
     * İki görüntü döndürür: modele giren kırpılmış orijinal ve boyanmış sonuç.
     * İkisi aynı kadrajda olduğu için önce/sonra kaydırıcısında üst üste oturur.
     */
    /**
     * @param onProgress (ilerleme 0..1, aşama) — aşama: 1 = model (Monet renkleri), 2-4 = fırça katmanları
     *   (kalın, orta, ince), 5 = bitiyor. Arka plandaki iş parçacığından çağrılır.
     */
    fun stylize(photo: Bitmap, onProgress: (Float, Int) -> Unit = { _, _ -> }): Pair<Bitmap, Bitmap> {
        val landscape = photo.width > photo.height
        val upright = if (landscape) rotate(photo, 90f) else photo
        val input = cropTo34(upright)

        val t0 = SystemClock.elapsedRealtime()
        Log.d("MonetCam", "fotoğraf hazır (${photo.width}x${photo.height}), model çalışıyor ($accelerator)…")
        // 1) CycleGAN: Monet'nin renkleri ve ışığı
        onProgress(0.05f, 1)
        val colors = toBitmap(runPipeline(model, ins, outs, toFloatArray(input)) { p -> onProgress(0.05f + 0.55f * p, 1) })
        val t1 = SystemClock.elapsedRealtime()
        Log.d("MonetCam", "model bitti: ${t1 - t0} ms (doğrudan aktarım: $directHandoff), fırça başlıyor…")
        // 2) Fırça darbeleri: renkli görüntüyü tuvale gerçek darbelerle yeniden boya
        val result = Painter.paint(colors) { layer, p -> onProgress(0.6f + 0.38f * p, 2 + layer) }
        onProgress(1f, 5)
        val t2 = SystemClock.elapsedRealtime()
        Log.d("MonetCam", "model ($accelerator): ${t1 - t0} ms · fırça: ${t2 - t1} ms")

        return if (landscape) rotate(input, -90f) to rotate(result, -90f) else input to result
    }

    override fun close() {
        ins.values.flatten().forEach { it.close() }
        outs.values.flatten().forEach { it.close() }
        model.close()
    }
}

fun loadBitmap(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        // Fotoğrafı doğrudan gereken boyutta aç: model 672x896'lık bir parça kullanıyor, kısa kenarın
        // ~1000 piksel olması yeterli. 12 MP fotoğrafı tam boyut (~48 MB) açmak bellek temizliğini
        // (GC) tetikleyip tüm uygulamayı bir an durduruyordu.
        val shortSide = minOf(info.size.width, info.size.height)
        val scale = minOf(1f, 1000f / shortSide)
        decoder.setTargetSize(
            (info.size.width * scale).toInt().coerceAtLeast(1),
            (info.size.height * scale).toInt().coerceAtLeast(1)
        )
    }
}

fun rotate(bitmap: Bitmap, degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

fun cropTo34(bitmap: Bitmap): Bitmap {
    val targetRatio = IN_W.toFloat() / IN_H
    val ratio = bitmap.width.toFloat() / bitmap.height
    val cropped = if (ratio > targetRatio) {
        val w = (bitmap.height * targetRatio).toInt()
        Bitmap.createBitmap(bitmap, (bitmap.width - w) / 2, 0, w, bitmap.height)
    } else {
        val h = (bitmap.width / targetRatio).toInt()
        Bitmap.createBitmap(bitmap, 0, (bitmap.height - h) / 2, bitmap.width, h)
    }
    return Bitmap.createScaledBitmap(cropped, IN_W, IN_H, true)
}

/** Bitmap -> modelin girdisi: NHWC (her piksel için r, g, b yan yana), 0..1 aralığında. */
fun toFloatArray(bitmap: Bitmap): FloatArray {
    val plane = IN_W * IN_H
    val pixels = IntArray(plane)
    bitmap.getPixels(pixels, 0, IN_W, 0, 0, IN_W, IN_H)
    val data = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val p = pixels[i]
        data[3 * i] = ((p shr 16) and 0xFF) / 255f
        data[3 * i + 1] = ((p shr 8) and 0xFF) / 255f
        data[3 * i + 2] = (p and 0xFF) / 255f
    }
    return data
}

/** Modelin çıktısı (NHWC, 0..1) -> Bitmap. */
fun toBitmap(data: FloatArray): Bitmap {
    val plane = IN_W * IN_H
    val pixels = IntArray(plane)
    for (i in 0 until plane) {
        val r = (data[3 * i] * 255f).toInt().coerceIn(0, 255)
        val g = (data[3 * i + 1] * 255f).toInt().coerceIn(0, 255)
        val b = (data[3 * i + 2] * 255f).toInt().coerceIn(0, 255)
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    return Bitmap.createBitmap(pixels, IN_W, IN_H, Bitmap.Config.ARGB_8888)
}
