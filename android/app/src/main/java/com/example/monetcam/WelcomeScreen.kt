package com.example.monetcam

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.monetcam.ui.theme.*
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.sin

// karşılama ekranı, yukarı kaydırınca onStart çağrılır
@Composable
fun WelcomeScreen(onStart: () -> Unit) {
    val drag = remember { Animatable(0f) }      // eksi değer = yukarı kaydırılmış
    val scope = rememberCoroutineScope()
    var heightPx by remember { mutableFloatStateOf(1f) }
    var leaving by remember { mutableStateOf(false) }
    val progress = (-drag.value / heightPx).coerceIn(0f, 1f)

    Box(
        Modifier
            .fillMaxSize()
            .background(Linen)
            .onSizeChanged { heightPx = it.height.toFloat().coerceAtLeast(1f) }
            .pointerInput(Unit) {
                detectVerticalDragGestures(
                    onDragEnd = {
                        scope.launch {
                            // ekranın %18'inden fazlası kaydırıldıysa geç
                            if (-drag.value > heightPx * 0.18f && !leaving) {
                                leaving = true
                                onStart()
                            } else {
                                drag.animateTo(0f, spring(dampingRatio = 0.7f))
                            }
                        }
                    },
                    onDragCancel = { scope.launch { drag.animateTo(0f, spring()) } }
                ) { change, dy ->
                    change.consume()
                    // sadece yukarı
                    scope.launch { drag.snapTo((drag.value + dy).coerceIn(-heightPx, 0f)) }
                }
            }
    ) {
        // arka plan + üstüne açık krem katman
        Image(
            painterResource(R.drawable.monet_bg),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        Box(Modifier.fillMaxSize().background(Linen.copy(alpha = 0.62f)))

        // içerik parmakla birlikte kayıp soluyor
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxSize()
                .systemBarsPadding()
                .padding(horizontal = 32.dp)
                .graphicsLayer {
                    translationY = drag.value * 0.6f
                    alpha = 1f - progress * 2.2f
                }
        ) {
            Spacer(Modifier.weight(1f))
            // simge
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(CanvasWhite)
                    .border(BorderStroke(1.dp, PondBlue.copy(alpha = 0.15f)), CircleShape)
            ) {
                Image(
                    painterResource(R.drawable.ic_launcher_monet_fg),
                    contentDescription = null,
                    modifier = Modifier.size(132.dp)
                )
            }
            Spacer(Modifier.height(28.dp))
            MonetWordmark()
            Spacer(Modifier.height(6.dp))
            Text(
                "ışığı yakala, gerisi tuvalde",
                style = MaterialTheme.typography.titleMedium.copy(fontSize = 22.sp),
                color = PondBlueDark
            )
            Spacer(Modifier.height(20.dp))
            Box(Modifier.width(36.dp).height(1.dp).background(PondBlue.copy(alpha = 0.25f)))
            Spacer(Modifier.height(20.dp))
            Text(
                "Fotoğraflarını Monet'nin renkleriyle ve\ngerçek fırça darbeleriyle yeniden boya.",
                style = MaterialTheme.typography.bodyLarge,
                color = InkSoft,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.weight(1.3f))
            SwipeUpHint()
            Spacer(Modifier.height(28.dp))
        }
    }
}

// yukarı kaydırma ipucu: aşağıdan yukarı yanıp sönen üç ok
@Composable
fun SwipeUpHint() {
    val t = rememberInfiniteTransition(label = "ipucu")
    val phase by t.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1600, easing = LinearEasing), RepeatMode.Restart),
        label = "dalga"
    )
    val lift = with(LocalDensity.current) { 6.dp.toPx() }
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.graphicsLayer { translationY = -lift * sin(2 * PI * phase).toFloat() }
        ) {
            // üstteki ok dalgayı en son alıyor
            for (i in 2 downTo 0) {
                val a = 0.15f + 0.85f * (0.5f + 0.5f * sin(2 * PI * (phase - i * 0.18f)).toFloat())
                Icon(
                    painterResource(R.drawable.ic_chevron_up),
                    contentDescription = null,
                    tint = PondBlue.copy(alpha = a),
                    modifier = Modifier.size(26.dp).offset(y = (i * -2).dp)
                )
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Başlamak için yukarı kaydır",
            style = MaterialTheme.typography.labelLarge,
            color = InkSoft
        )
    }
}
