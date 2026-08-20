# blink

Yatan/hareketsiz bir hastayı tek kamerayla izleyip **göz kırpma, bakış yönü, kol/bacak hareketi, parmak hareketi, kafa hareketi ve ses** sinyallerini otomatik sayan bir izleme aracı. MediaPipe (yüz/el/vücut iskeleti) ve Intel OpenVINO (`gaze-estimation-adas-0002`, bakış yönü) kullanır; hem ticari kullanıma uygundur (Apache 2.0 / MIT, araştırma-amaçlı kısıtlama yok).

## Özellikler

- **Göz kırpma** ve **bakış yönü** (sol/sağ/yukarı/aşağı) sayacı
- **Kol** ve **bacak** hareketi sayacı (yön-bağımsız, EMA tabanlı algılama)
- **Parmak hareketi** sayacı (sol/sağ el, ayrı ayrı)
- **Kafa hareketi** sayacı (dönüş VEYA konum değişikliği — örn. başı çevirme ya da yerden kaldırma)
- **Ses**: yatan kişinin ses çıkarmasını algılama + mikrofon→hoparlör canlı geçiş (yerel interkom)
- **GCS motor tepki testi** (M2-M5, sezgisel — kesin tıbbi ölçüm değildir, bkz. aşağıdaki uyarı)
- Olay anında öncesi/sonrasıyla otomatik video klip kaydı (her sayaç için ayrı)
- İki kamera senaryosu:
  - **Uzak/sabit kamera** (`gaze-birlesik-uzak.py`) — kamera hastadan uzakta/sabit (örn. tavana/köşeye montajlı); `nokta_sec.py` ile işaretlenen sabit bölgeler dijital olarak yakınlaştırılıp o bölgede tespit çalıştırılır.
  - **Ana kamera modu** (aynı dosyada `z` tuşu) — bölge işaretlemeye gerek kalmadan doğrudan geniş açıdan.

## Kurulum

```bash
pip install -r requirements.txt
```

Model dosyaları (MediaPipe `.task`, OpenVINO `.xml`/`.bin`) ilk çalıştırmada otomatik indirilir (internet gerekir, sadece ilk sefer).

`openvino`, `onnxruntime` ve `uniface`, `requirements.txt`'te sürüm sabitlenmeden (unpinned) listelenmiştir — kendi ortamınızdaki uyumlu sürümleri kurup ardından `pip freeze` ile sabitlemeniz önerilir. `uniface` için GPU'lu kurulum isterseniz: `pip install "uniface[gpu]"`.

## Kullanım

1. **Kamera indeksini bul** (birden fazla kamera varsa):
   ```bash
   python kamerabul.py
   ```
2. **Uzak kamera kullanıyorsanız**, sabit bölgeleri (yüz / sol el / sağ el) işaretle:
   ```bash
   python nokta_sec.py
   ```
   1/2/3 ile bölge seç, sol tıkla ile yerleştir, +/- ile yakınlaştırma oranı, `s` ile kaydet.
3. **Ana uygulamayı çalıştır:**
   ```bash
   python gaze-birlesik-uzak.py
   ```

### Kontroller

| Tuş | İşlev |
|---|---|
| `c` | Kalibre et (bakış sapmasını sıfırla) |
| `r` | Kesit al (JPEG) |
| `v` | Video kaydı aç/kapat |
| `h` | Sayaçları aç/kapat (başlangıçta kapalı) |
| `m` | Mikrofon→hoparlör canlı geçişi aç/kapat |
| `z` | Ana kamera modu aç/kapat |
| `g` | GCS motor tepki testi başlat |
| `q` | Çıkış |

## Proje yapısı

| Dosya | Açıklama |
|---|---|
| `ayarlar.py` | Tüm eşik/sabit/yol ayarları — davranışı değiştirmek için önce buraya bakın |
| `gaze-birlesik-uzak.py` | Ana uygulama (uzak/sabit kamera + ana kamera modu) |
| `modeller.py` | MediaPipe/OpenVINO model yükleme |
| `gorsellik.py` | Çizim ve geometri/hareket-algılama yardımcıları |
| `kayit.py` | Kesit (JPEG) ve video (MP4) kaydı |
| `bolgeler.py` | Sabit bölge noktalarının `zoom_noktalari.json`'a kaydı/okunması |
| `nokta_sec.py` | Sabit bölgeleri fare ile işaretleme aracı |
| `kamerabul.py` | Kamera indeksini bulma aracı |
| `ses.py` | Mikrofon/hoparlör: ses algılama + canlı geçiş |
| `l2csbirlesik.py` | Alternatif bakış motoru (L2CS-Net / Gaze360) |
| `uniface-birlesik.py` | Alternatif bakış motoru (UniFace / MobileGaze) |

## Önemli uyarılar

- **GCS motor tepki testi kesin bir tıbbi ölçüm aracı değildir.** Tek kameradan 2D landmark konumuna dayanan sezgisel bir tahmindir; sonuç bir klinisyenin kendi gözlemiyle doğrulaması gereken bir ön-öneri olarak değerlendirilmelidir.
- Eşik değerleri (parmak, kafa, ses vb.) gerçek kullanım ortamınıza göre kalibrasyon gerektirebilir — `ayarlar.py` içindeki ilgili sabitlerin yanındaki açıklamalara bakın.
