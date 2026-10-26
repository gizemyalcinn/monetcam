# MonetCam

Telefonla çektiğin ya da galeriden seçtiğin fotoğrafı Claude Monet tablosuna çeviren bir Android uygulaması. Her şey telefonun içinde çalışıyor, internete ihtiyaç yok.

Proje iki parçadan oluşuyor: modelin eğitildiği Python tarafı (bu klasör) ve uygulamanın kendisi (`android/`).

## Nasıl çalışıyor

1. **CycleGAN:** Fotoğraf önce Monet tablolarıyla eğitilmiş bir CycleGAN'dan geçiyor. Bu adım Monet'nin renklerini ve ışığını veriyor.
2. **Fırça darbeleri:** Çıkan görüntü boş bir tuvale gerçek fırça darbeleriyle yeniden boyanıyor (Hertzmann 1998). Önce kalın, sonra orta, en son ince fırça kullanılıyor. Darbeler görüntüdeki kenarlar boyunca kıvrılıyor.

İlk denemelerde fast style transfer (Johnson vd.) kullandım. Sonuçlarda leke leke bozulmalar oluyordu ve tablo hissi vermiyordu. Bu yüzden CycleGAN'a geçtim.

## Klasörler

| Dosya | Ne işe yarıyor |
|---|---|
| `cyclegan.py` | CycleGAN üreticisi (resmi koddaki ResnetGenerator) |
| `train_cyclegan.py` | monet2photo verisiyle ince ayar (Kaggle'da çalıştırıldı) |
| `try_cyclegan.py` | Bilgisayarda tek bir fotoğrafla deneme |
| `paint.py` | Fırça darbeli boyamanın Python hali |
| `export_cyclegan_gpu.py` | Modeli telefon GPU'sunda çalışacak şekilde tflite'a çevirme |
| `export_cyclegan_segments.py` | Modeli 20 parçaya bölerek çevirme (uygulamada bu kullanılıyor) |
| `model.py`, `losses.py`, `train.py`, `stylize.py` | Eski fast style transfer denemeleri |
| `android/` | Android uygulaması (Kotlin, Jetpack Compose, LiteRT) |

## Kurulum (Python)

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Bir fotoğrafı bilgisayarda denemek için:

```bash
python try_cyclegan.py images/foto.jpg
```

Ağırlık dosyaları (`checkpoints/`) ve veri setleri büyük olduğu için repoda yok.

## Eğitim

- **Veri:** Kaggle'daki monet2photo veri seti.
- **Başlangıç ağırlıkları:** CycleGAN yazarlarının yayımladığı `style_monet` ağırlıkları.
- **İnce ayar:** `train_cyclegan.py` bu ağırlıklardan başlayarak ince ayar yapıyor.

Kaggle'da notebook'u **Save & Run All** ile çalıştırmak gerekiyor. Etkileşimli oturumda çıktılar kayboluyor.

## Telefona aktarma

```bash
python export_cyclegan_segments.py
```

Çıkan `monet_cyclegan_seg_896x672.tflite` dosyası `android/app/src/main/assets/` klasörüne gidiyor. Bu dönüşümde neyi neden değiştirdiğim:

- **Desteklenmeyen işlemler:** ReflectionPad, InstanceNorm ve ConvTranspose katmanları, GPU'nun desteklediği işlemlerle yeniden yazıldı.
- **Hassasiyet:** GPU'nun yarım hassasiyetinde (fp16) değerler taşıyor ve resim bozuk çıkıyordu. Uygulama bu yüzden tam hassasiyetle (fp32) çalışıyor.
- **Parçalara bölme:** Model tek parça halindeyken GPU'yu birkaç saniye meşgul edip ekranı donduruyordu. 20 parçaya bölünce bu düzeldi.

## Android uygulaması

`android/` klasörünü Android Studio'da açıp çalıştırman yeterli (minSdk 28).

- **Giriş:** Karşılama ekranında yukarı kaydırarak uygulamaya giriliyor.
- **Fotoğraf:** Kamerayla çekiliyor ya da galeriden seçiliyor.
- **Sonuç:** Önce/sonra kaydırıcısıyla orijinalle karşılaştırılıyor, galeriye kaydedilebiliyor ve paylaşılabiliyor.
- **Model hazırlığı:** Açılışta GPU, bilinen bir test deseniyle kontrol ediliyor. Test tutmazsa model CPU'da çalışıyor.

Redmi Note 13 Pro 5G'de bir fotoğraf yaklaşık 15 saniyede işleniyor: model ~6 sn, fırça darbeleri ~8 sn.

## Kaynaklar ve lisanslar

- **CycleGAN:** Zhu vd., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (2017).
- **Fırça darbeleri:** Hertzmann, *Painterly Rendering with Curved Brush Strokes of Multiple Sizes* (1998).
- **Logodaki tablolar:** Monet'nin tabloları kamu malı, görseller Wikimedia Commons'tan alındı.
- **Görseller:** Arka plan, boş tuval ve uygulama simgesi Higgsfield (Nano Banana Pro) ile üretildi.
- **Yazı tipleri:** Abril Fatface ve Cormorant Garamond OFL, Special Elite Apache 2.0 lisanslı. Lisans metinleri `android/app/src/main/assets/licenses/` klasöründe.
