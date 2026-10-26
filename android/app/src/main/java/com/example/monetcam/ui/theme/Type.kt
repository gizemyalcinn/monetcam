package com.example.monetcam.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.example.monetcam.R

// yazı tipi lisansları assets/licenses içinde
val LogoFont = FontFamily(Font(R.font.abril_fatface))         // logo
val TypewriterFont = FontFamily(Font(R.font.special_elite))
val SerifItalic = FontFamily(Font(R.font.cormorant_italic))

val Typography = Typography(
    displaySmall = TextStyle(fontFamily = LogoFont, fontSize = 52.sp, lineHeight = 56.sp),
    titleMedium = TextStyle(fontFamily = SerifItalic, fontSize = 20.sp, lineHeight = 26.sp),
    bodyLarge = TextStyle(fontFamily = FontFamily.Default, fontSize = 16.sp, lineHeight = 24.sp, letterSpacing = 0.3.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.Default, fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium, fontSize = 15.sp, letterSpacing = 0.4.sp),
    labelMedium = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium, fontSize = 12.sp, letterSpacing = 0.5.sp)
)
