package com.example.monetcam.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Dinamik renk bilerek KAPALI: açık olsaydı uygulama renklerini telefonun
// duvar kağıdından alırdı ve Monet paleti kaybolurdu.
private val MonetColors = lightColorScheme(
    primary = PondBlue,
    onPrimary = Color.White,
    primaryContainer = LavenderMist,
    onPrimaryContainer = PondBlueDark,
    secondary = Emerald,
    onSecondary = Color.White,
    secondaryContainer = SageMist,
    onSecondaryContainer = Ink,
    tertiary = LilyRose,
    onTertiary = Color.White,
    background = Linen,
    onBackground = Ink,
    surface = CanvasWhite,
    onSurface = Ink,
    surfaceVariant = LavenderMist,
    onSurfaceVariant = InkSoft,
    outline = Lavender,
)

@Composable
fun MonetCamTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = MonetColors,
        typography = Typography,
        content = content
    )
}
