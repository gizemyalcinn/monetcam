package com.example.monetcam

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import java.io.File

private fun authority(context: Context) = "${context.packageName}.fileprovider"

private fun imagesDir(context: Context): File =
    File(context.cacheDir, "images").apply { mkdirs() }

// kameranın fotoğrafı yazacağı geçici dosya
fun newCameraUri(context: Context): Uri {
    val file = File(imagesDir(context), "camera_${System.currentTimeMillis()}.jpg")
    return FileProvider.getUriForFile(context, authority(context), file)
}

// galeriye kaydet (Pictures/MonetCam)
fun saveToGallery(context: Context, bitmap: Bitmap): Boolean {
    val name = "Monet_${System.currentTimeMillis()}.jpg"
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, name)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MonetCam")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
    }
    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
    val ok = resolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) } ?: false
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
    }
    return ok
}

// geçici dosyaya yazıp paylaşım menüsünü açar
fun shareImage(context: Context, bitmap: Bitmap) {
    val file = File(imagesDir(context), "monet_share.jpg")
    file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    val uri = FileProvider.getUriForFile(context, authority(context), file)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "image/jpeg"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Tabloyu paylaş"))
}
