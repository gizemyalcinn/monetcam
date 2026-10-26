package com.example.monetcam

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.activity.compose.BackHandler
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.animation.Crossfade
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.ImageShader
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.Shader
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import com.example.monetcam.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MonetCamTheme { MonetRoot() }
        }
    }
}

/**
 * Uygulamanın kökü: önce karşılama ekranı, sürgü kaydırılınca işleme ekranı (yandan kayarak gelir).
 * Geri tuşu işleme ekranından karşılamaya döndürür.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@Composable
fun MonetRoot() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // Model GPU için ilk açılışta ~14 sn hazırlanıyor (derleme + doğruluk testi). Bunu karşılama ekranı
    // açılır açılmaz arka planda başlatıyoruz: kullanıcı içeri girdiğinde büyük ihtimalle hazır olur.
    val stylizerJob = remember {
        scope.async(Dispatchers.Default) { MonetStylizer(context.applicationContext) }
    }
    DisposableEffect(Unit) {
        onDispose {
            if (stylizerJob.isCompleted && !stylizerJob.isCancelled) stylizerJob.getCompleted().close()
        }
    }

    var started by rememberSaveable { mutableStateOf(false) }
    BackHandler(enabled = started) { started = false }
    AnimatedContent(
        targetState = started,
        transitionSpec = {
            // içeri: karşılama yukarı kayıp gider, işleme ekranı aşağıdan gelir; geri: tersi
            if (targetState) {
                (slideInVertically(tween(450)) { it } + fadeIn(tween(400))) togetherWith
                    (slideOutVertically(tween(450)) { -it / 2 } + fadeOut(tween(300)))
            } else {
                (slideInVertically(tween(450)) { -it / 2 } + fadeIn(tween(400))) togetherWith
                    (slideOutVertically(tween(450)) { it } + fadeOut(tween(300)))
            }
        },
        label = "ekran"
    ) { inside ->
        if (inside) MonetApp(stylizerJob) else WelcomeScreen(onStart = { started = true })
    }
}

@Composable
fun MonetApp(stylizerJob: Deferred<MonetStylizer>) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }

    var original by remember { mutableStateOf<Bitmap?>(null) }
    var painted by remember { mutableStateOf<Bitmap?>(null) }
    var busy by remember { mutableStateOf(false) }
    var phase by remember { mutableIntStateOf(0) }
    var cameraUri by remember { mutableStateOf<Uri?>(null) }

    val mainHandler = remember { android.os.Handler(android.os.Looper.getMainLooper()) }

    fun process(uri: Uri) {
        scope.launch {
            // Yeni fotoğraf işlenirken eskisi arkada görünmesin: önce kartı boşalt
            original = null
            painted = null
            phase = 0
            busy = true
            // Yükleme ekranının ilk karelerinin çizilmesine fırsat ver; ağır iş (fotoğrafı açma, GPU)
            // hemen başlarsa ekran ilk anda takılmış gibi görünüyor
            delay(350)
            try {
                val stylizer = stylizerJob.await()
                val (pair, ms) = withContext(Dispatchers.Default) {
                    val photo = loadBitmap(context, uri)
                    val t0 = SystemClock.elapsedRealtime()
                    // ilerleme arka planda geliyor; ekran durumunu ana iş parçacığında güncelle
                    val out = stylizer.stylize(photo) { _, ph ->
                        mainHandler.post { phase = ph }
                    }
                    out to (SystemClock.elapsedRealtime() - t0)
                }
                original = pair.first
                painted = pair.second
                // süre ve GPU/CPU bilgisi sadece geliştirici için Logcat'e yazılıyor, ekranda gösterilmiyor
                android.util.Log.d("MonetCam", "stylize: %.1f sn · %s".format(ms / 1000f, stylizer.accelerator))
            } catch (e: Exception) {
                snackbar.showSnackbar("Bir şeyler ters gitti: ${e.message}")
            }
            busy = false
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> if (uri != null) process(uri) }

    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success -> cameraUri?.let { if (success) process(it) } }

    Box(Modifier.fillMaxSize().background(Linen)) {
    // Arka plan: Monet'nin Nilüfer Göleti detayı, bulanıklaştırılmış; üst kısım okunurluk için açık
    Image(
        painterResource(R.drawable.monet_bg),
        contentDescription = null,
        contentScale = ContentScale.Crop,
        modifier = Modifier.fillMaxSize()
    )
    Scaffold(
        containerColor = Color.Transparent,
        snackbarHost = { SnackbarHost(snackbar) },
        modifier = Modifier.fillMaxSize()
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Header()

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                ArtFrame(original = original, painted = painted, busy = busy, phase = phase)
            }

            Spacer(Modifier.height(16.dp))

            ActionBar(
                busy = busy,
                hasResult = painted != null,
                onCamera = {
                    val uri = newCameraUri(context)
                    cameraUri = uri
                    cameraLauncher.launch(uri)
                },
                onGallery = {
                    galleryLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                onSave = {
                    painted?.let { bmp ->
                        scope.launch {
                            val ok = withContext(Dispatchers.IO) { saveToGallery(context, bmp) }
                            snackbar.showSnackbar(
                                if (ok) "Tablo galeriye kaydedildi (Pictures/MonetCam)"
                                else "Kaydedilemedi"
                            )
                        }
                    }
                },
                onShare = { painted?.let { shareImage(context, it) } }
            )
            Spacer(Modifier.height(12.dp))
        }
    }
    }
}

@Composable
fun Header() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 20.dp, bottom = 14.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        MonetWordmark()
        Spacer(Modifier.height(6.dp))
        Text(
            "ışığı yakala, gerisi tuvalde",
            style = MaterialTheme.typography.titleMedium.copy(fontSize = 22.sp),
            color = PondBlueDark
        )
    }
}

/** "MonetCam" yazısı: her harfin içi farklı bir Monet tablosuyla dolu, koyu mürekkep konturlu. */
@OptIn(ExperimentalTextApi::class)
@Composable
fun MonetWordmark() {
    // her harfin içinde ayrı bir Monet tablosu:
    // M = Izlenim, Gundogumu · o = Nilüferler · n = Gelincikler · e = Şemsiyeli Kadın · t = Parlamento, Gün Batımı
    val letters = listOf(
        'M' to R.drawable.monet_letter_1,
        'o' to R.drawable.monet_letter_2,
        'n' to R.drawable.monet_letter_3,
        'e' to R.drawable.monet_letter_4,
        't' to R.drawable.monet_letter_5,
        'C' to R.drawable.monet_letter_6,   // İzlenim, Gündoğumu (gökyüzü ve kayıklar)
        'a' to R.drawable.monet_letter_7,   // Nilüferler (üst kısım)
        'm' to R.drawable.monet_letter_8,   // Parlamento, Gün Batımı (su ve kuleler)
    )
    val base = TextStyle(fontFamily = LogoFont, fontSize = 40.sp, lineHeight = 44.sp)
    Row {
        letters.forEach { (ch, res) ->
            val img = ImageBitmap.imageResource(res)
            val brush = remember(img) { CoverImageBrush(img) }
            Box {
                // önce kalın kontur (tablonun çerçevesi gibi), üstüne tablo dokulu dolgu
                Text(ch.toString(), style = base.copy(color = PondBlueDark, drawStyle = Stroke(width = 7f, join = StrokeJoin.Round)))
                Text(ch.toString(), style = base.copy(brush = brush))
            }
        }
    }
}

/** Resmi, harfin kutusunu tamamen kaplayacak şekilde ölçekleyip ortalar (ContentScale.Crop gibi). */
class CoverImageBrush(private val img: ImageBitmap) : ShaderBrush() {
    override fun createShader(size: Size): Shader {
        val shader = ImageShader(img, TileMode.Clamp, TileMode.Clamp)
        val scale = maxOf(size.width / img.width, size.height / img.height)
        val m = android.graphics.Matrix().apply {
            setScale(scale, scale)
            postTranslate((size.width - img.width * scale) / 2f, (size.height - img.height * scale) / 2f)
        }
        shader.setLocalMatrix(m)
        return shader
    }
}

/** Tablo çerçevesi: paspartulu kart. İçinde boş durum, önce/sonra ya da yükleniyor görünümü. */
@Composable
fun ArtFrame(original: Bitmap?, painted: Bitmap?, busy: Boolean, phase: Int) {
    val ratio = painted?.let { it.width.toFloat() / it.height } ?: (3f / 4f)
    Box(
        modifier = Modifier
            .aspectRatio(ratio, matchHeightConstraintsFirst = ratio < 1f)
            .shadow(18.dp, RoundedCornerShape(20.dp), ambientColor = PondBlue, spotColor = PondBlue)
            .clip(RoundedCornerShape(20.dp))
            .background(CanvasWhite)
            .padding(10.dp)
            .clip(RoundedCornerShape(12.dp))
    ) {
        Crossfade(targetState = painted, label = "frame") { result ->
            if (result != null && original != null) {
                BeforeAfter(original = original, painted = result)
            } else {
                EmptyCanvas(showText = !busy)
            }
        }
        AnimatedVisibility(visible = busy, enter = fadeIn(), exit = fadeOut()) {
            Box(
                Modifier
                    .fillMaxSize()
                    .background(CanvasWhite),
                contentAlignment = Alignment.Center
            ) {
                // Sadece yazı: işlemin gerçek aşamasını gösterir, aşama değişince yumuşakça geçer.
                // (Sürekli hareket eden bir gösterge koymuyoruz: model GPU'da çalışırken ekran o saniyelerde
                // yeni kare çizemiyor, hareketli bir şey takılmış gibi görünüyordu.)
                Crossfade(
                    targetState = phase.coerceIn(0, loadingLines.lastIndex),
                    animationSpec = tween(400),
                    label = "asama",
                    modifier = Modifier.fillMaxWidth()
                ) { i ->
                    Text(
                        loadingLines[i],
                        style = MaterialTheme.typography.titleMedium,
                        color = PondBlueDark,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun EmptyCanvas(showText: Boolean = true) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        // Boş tuval: kenarlarında nilüfer fırça izleri olan, ortası açık bir eskiz (Higgsfield ile üretildi)
        Image(
            painterResource(R.drawable.empty_canvas),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        // Yükleniyor yazısı görünürken bu metinler gizlenir: ekranda aynı anda tek yazı olsun
        AnimatedVisibility(visible = showText, enter = fadeIn(), exit = fadeOut()) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            Icon(
                painterResource(R.drawable.ic_palette),
                contentDescription = null,
                tint = PondBlue.copy(alpha = 0.7f),
                modifier = Modifier.size(56.dp)
            )
            Spacer(Modifier.height(16.dp))
            Text(
                "Boş bir tuval",
                style = MaterialTheme.typography.titleMedium,
                color = PondBlueDark
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "Bir fotoğraf çek ya da galeriden seç;\nMonet onu yeniden boyasın.",
                style = MaterialTheme.typography.bodyMedium,
                color = InkSoft,
                textAlign = TextAlign.Center
            )
        }
        }
    }
}

/** Önce/sonra karşılaştırması: parmakla sürüklenen çizginin solu orijinal, sağı tablo. */
@Composable
fun BeforeAfter(original: Bitmap, painted: Bitmap) {
    var fraction by remember(painted) { mutableFloatStateOf(0.5f) }
    val originalImg = remember(original) { original.asImageBitmap() }
    val paintedImg = remember(painted) { painted.asImageBitmap() }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(painted) {
                detectHorizontalDragGestures { change, _ ->
                    fraction = (change.position.x / size.width).coerceIn(0f, 1f)
                }
            }
            .pointerInput(painted) {
                detectTapGestures { pos -> fraction = (pos.x / size.width).coerceIn(0f, 1f) }
            }
    ) {
        Image(paintedImg, contentDescription = "Monet tarzı", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        Image(
            originalImg,
            contentDescription = "Orijinal",
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxSize()
                .drawWithContent {
                    clipRect(right = size.width * fraction) { this@drawWithContent.drawContent() }
                }
        )

        // Ayırıcı çizgi
        Canvas(Modifier.fillMaxSize()) {
            val x = size.width * fraction
            drawLine(Color.White, Offset(x, 0f), Offset(x, size.height), strokeWidth = 3.dp.toPx())
        }

        // Tutamak
        BoxWithConstraints(Modifier.fillMaxSize()) {
            Box(
                modifier = Modifier
                    .offset(x = maxWidth * fraction - 20.dp, y = maxHeight / 2 - 20.dp)
                    .size(40.dp)
                    .shadow(6.dp, CircleShape)
                    .background(Color.White, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(painterResource(R.drawable.ic_swap), contentDescription = null, tint = PondBlue, modifier = Modifier.size(22.dp))
            }
        }

        LabelChip("Orijinal", Modifier.align(Alignment.TopStart).padding(10.dp))
        LabelChip("Monet", Modifier.align(Alignment.TopEnd).padding(10.dp))
    }
}

@Composable
fun LabelChip(text: String, modifier: Modifier = Modifier) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        color = Color.White,
        modifier = modifier
            .background(Ink.copy(alpha = 0.45f), RoundedCornerShape(50))
            .padding(horizontal = 10.dp, vertical = 4.dp)
    )
}

/** Müze etiketi havasında küçük bir bilgi satırı. */
@Composable
fun ActionBar(
    busy: Boolean,
    hasResult: Boolean,
    onCamera: () -> Unit,
    onGallery: () -> Unit,
    onSave: () -> Unit,
    onShare: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(28.dp),
        color = CanvasWhite.copy(alpha = 0.92f),
        shadowElevation = 6.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            RoundAction(R.drawable.ic_camera, "Kamera", enabled = !busy, onClick = onCamera, color = LilyRose)
            Button(
                onClick = onGallery,
                enabled = !busy,
                colors = ButtonDefaults.buttonColors(containerColor = PondBlue, disabledContainerColor = PondBlue.copy(alpha = 0.35f), disabledContentColor = Color.White.copy(alpha = 0.85f)),
                contentPadding = PaddingValues(horizontal = 22.dp, vertical = 14.dp)
            ) {
                Icon(painterResource(R.drawable.ic_gallery), contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                // düz (italik olmayan), sade yazı
                Text(
                    "Galeriden seç",
                    style = MaterialTheme.typography.labelLarge.copy(
                        fontSize = 17.sp,
                        fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold,
                        letterSpacing = 0.2.sp
                    )
                )
            }
            RoundAction(R.drawable.ic_save, "Kaydet", enabled = hasResult && !busy, onClick = onSave, color = Emerald)
            RoundAction(R.drawable.ic_share, "Paylaş", enabled = hasResult && !busy, onClick = onShare, color = SunriseOrange)
        }
    }
}

@Composable
fun RoundAction(icon: Int, label: String, enabled: Boolean, onClick: () -> Unit, color: Color = PondBlue) {
    FilledIconButton(
        onClick = onClick,
        enabled = enabled,
        colors = IconButtonDefaults.filledIconButtonColors(
            containerColor = color,
            contentColor = Color.White,
            disabledContainerColor = color.copy(alpha = 0.28f),
            disabledContentColor = Color.White.copy(alpha = 0.85f)
        ),
        modifier = Modifier.size(50.dp)
    ) {
        Icon(painterResource(icon), contentDescription = label, modifier = Modifier.size(22.dp))
    }
}

/** Yükleme ekranındaki yazılar; işlemin gerçek aşamasına göre (MonetStylizer.stylize -> phase) seçilir. */
private val loadingLines = listOf(
    "Fotoğraf inceleniyor…",                   // 0: fotoğraf açılıyor
    "Monet'nin renk paleti uygulanıyor…",      // 1: model
    "Geniş fırçayla ilk katman atılıyor…",     // 2: kalın fırça
    "Orta fırçayla şekiller belirginleşiyor…", // 3: orta fırça
    "İnce fırçayla detaylar işleniyor…",       // 4: ince fırça
    "Tablo son haline geliyor…",               // 5: bitiyor
)
