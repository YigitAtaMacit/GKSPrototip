"""MediaPipe + Intel OpenVINO (gaze-estimation-adas-0002) birlesik uygulamasi
icin TUM ayarlar/sabitler/yollar.

Davranisi degistirmek istedigin zaman (esik degerleri, hangi cizimlerin
gorunecegi, hangi modellerin aktif olacagi vb.) SADECE bu dosyayi duzenlemen
yeterli - diger dosyalar (modeller.py, gorsellik.py, kayit.py,
gaze_birlesik.py) buradaki degerleri kullanir.

LISANS NOTU: Bu surum, L2CS-Net'in Gaze360 ile egitilmis agirliklari yerine
Intel OpenVINO'nun gaze-estimation-adas-0002 modelini kullanir. Hem bu model
hem MediaPipe Apache 2.0 lisanslidir ve Gaze360/MPIIGaze'deki "sadece
arastirma amacli" kisitlamasini TASIMAZ - ticari kullanima uygundur.
"""
from pathlib import Path

BURASI = Path(__file__).resolve().parent

# --- Acik/kapali ozellikler --------------------------------------------------
# Performans icin acik/kapali yapilabilir modeller - yavas gelirse False yap.
# AKTIF_GAZE=False: OpenVINO gaze modeli HIC YUKLENMEZ/CALISMAZ - bakis yonu
# sayaclari (SOL/SAG/YUKARI/ASAGI) ve goz oku bu durumda calismaz.
AKTIF_GAZE = True
AKTIF_POSE = True
AKTIF_EL = True

# --- Kimlik kilidi (tek kisiye odaklanma) --------------------------------
# Ekrana ilk giren kisi "kilitlenir" - sonradan giren baska biri ne kadar
# yakinda/onde olursa olsun sayaclar/bakis SADECE o kilitli kisi icin
# calismaya devam eder, digerleri TAMAMEN yok sayilir. GORUNTUYU KIRPMAZ -
# sadece MediaPipe'in dondurdugu adaylar arasindan (asagidaki *_ADAY_SAYISI
# kadar) hangisinin "kilitli kisi" oldugunu her karede yeniden bulur (en
# yakin komsu + boyut esigi ile). Pahali olan OpenVINO bakis modeli YINE
# SADECE kilitli kisi icin calisir (adaylarin geri kalani icin calismaz),
# bu yuzden ikinci bir kisi girdiginde performans/donma sorunu YARATMAZ.
KIMLIK_KILIDI_AKTIF = True
YUZ_ADAY_SAYISI = 3    # FaceLandmarker'in dondurecegi maks aday sayisi
GOVDE_ADAY_SAYISI = 3  # PoseLandmarker'in dondurecegi maks aday sayisi

# Bir karede kilitli kisiye "en yakin" aday, kilitli kisinin SON pozisyonuna
# gore kendi buyuklugunun (yuz genisligi / omuz genisligi) bu ORANINDAN
# fazla uzaklastiysa REDDEDILIR (baska biri sayilir). Kucultursen kilit
# daha "siki" olur (kolayca baska birine gecmez) ama hizli hareketlerde
# kilidi kaybetme ihtimali artar.
KIMLIK_KILIDI_MAKS_SICRAMA_ORANI = 0.9

# Kilitli kisi bu kadar KARE boyunca hic eslesmezse (kafasini tam cevirmis,
# kadraj disina cikmis vb.) kilit tamamen birakilir - bir sonraki karede en
# BUYUK/on plandaki kisiye yeniden kilitlenir.
KIMLIK_KILIDI_KAYIP_KARE_LIMITI = 45  # ~1.5 saniye, ~30fps'te

# Otomatik OLAY KESITI (once/sonra video klibi) kategori basina ac/kapa -
# simdilik UCU DE kapali. SAYACLAR (KIRPMA/SOL KOL/SAG KOL/SOL/SAG/YUKARI/
# ASAGI) bundan ETKILENMEZ, hepsi normal calismaya devam eder - sadece
# otomatik VIDEO KLIBI alinmiyor. Tekrar acmak istersen ilgiliyi True yap.
KOL_OLAY_KESITI_AKTIF = False
KIRPMA_OLAY_KESITI_AKTIF = False
BAKIS_OLAY_KESITI_AKTIF = False

# False yaparsan ilgili gorsel EKRANDA CIZILMEZ ama tespit (ve dolayisiyla
# ilgili sayaclar/kalibrasyon) YINE DE CALISIR - sadece gorsel gizlenir.
GOVDE_CIZIMI_GOSTER = True
EL_CIZIMI_GOSTER = True
YUZ_CIZIMI_GOSTER = True

# FaceLandmarker'in yuzu "yuz" olarak kabul etmesi icin gereken minimum
# guven skoru (0-1). Varsayilan MediaPipe degeri 0.5'tir, biz zaten 0.3'e
# indirmistik. TEPEDEN/DIREK KAMERA gibi asiri acili kurulumlarda (yuz
# normalden COK farkli/sikismis gorunuyor) bunu daha da dusurmek tespiti
# kolaylastirir - ama COK dusurursen yuz OLMAYAN seyleri de "yuz" sanma
# riski artar (yanlis pozitif). 0.1-0.15 arasi genelde makul bir sinir.
YUZ_TESPIT_ESIK = 0.15

# DIJITAL YAKINLASTIRMA: kare, TAM ORTASINDAN kirpilip eski boyutuna geri
# buyutulur - yuz kucuk/uzakta kaliyorsa (orn. direk/tepeden kamera) bunu
# artirmak MediaPipe'e daha buyuk/net bir yuz verir, tespiti kolaylastirir.
# 1.0 = yakinlastirma YOK (tam goruntu). 2.0 = ortadaki YARISI (genislik VE
# yukseklikte) alinip tam kareye buyutulur - yani goruntu alani (FOV) 4'te
# bire duser. DIKKAT: kisi/yuz bu kirpilan alanin DISINA cikarsa TAMAMEN
# kadraj disi kalir - hareket alani genisse (orn. egzersiz) dusuk tut,
# kisi sabit bir yerde (orn. yatak) kaliyorsa yuksek tutmak sorun olmaz.
DIJITAL_YAKINLASTIRMA = 3.0

# --- Kamera --------------------------------------------------------------
# USB kamera (A4Tech FHD 1080p) takinca isletim sistemi ona 0'dan farkli bir
# indeks verebilir (ozellikle laptop'ta dahili kamera zaten 0'i kullaniyorsa
# USB kamera genelde 1 olur). Uygulamayi calistirinca acilan pencerede YANLIS
# kamera goruntusu geliyorsa bu degeri 1, 2... diye degistirip tekrar dene.
KAMERA_INDEKSI = 1

# DENENDI: cv2.set() ile 1920x1080@30 + MJPG'yi ACIKCA istemek bu kamerada
# Windows Kamera uygulamasindan DAHA KOTU goruntu verdi (muhtemelen surucu
# bu explicit istegi otomatik/varsayilan modundan daha kotu bir moda
# dusuruyor). Bu yuzden KAMERA_COZUNURLUK_ZORLA=False - kamera artik HICBIR
# .set() cagrisi olmadan, sadece KAMERA_INDEKSI ile aciliyor (eskiden iyi
# calisan hal). True yaparsan asagidaki degerler zorlanir - farkli bir
# kamerada/surucude ise yarayabilir, dene istersen.
KAMERA_COZUNURLUK_ZORLA = False
KAMERA_GENISLIK = 1920
KAMERA_YUKSEKLIK = 1080
KAMERA_FPS = 30
KAMERA_FOURCC = "MJPG"

# --- Dosya yollari -----------------------------------------------------------
# gaze-estimation-adas-0002.xml VE .bin - ilk calistirmada BURAYA otomatik
# indirilir (internet gerekir, sadece ilk sefer; MediaPipe .task dosyalari
# gibi). Indirme basarisiz olursa elle bu iki URL'den indirip BURASI'na
# koyabilirsin:
GAZE_MODEL_XML_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
GAZE_MODEL_BIN_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.bin"
GAZE_MODEL_XML = BURASI / "gaze-estimation-adas-0002.xml"
GAZE_MODEL_BIN = BURASI / "gaze-estimation-adas-0002.bin"
GAZE_CIHAZ = "CPU"  # "GPU" da denenebilir ama SADECE Intel GPU'larda calisir (NVIDIA'da degil)

FACE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_TASK_YOLU = BURASI / "face_landmarker.task"

POSE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_TASK_YOLU = BURASI / "pose_landmarker_lite.task"

HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_TASK_YOLU = BURASI / "hand_landmarker.task"

KESIT_KLASORU = BURASI / "kesitler"
VIDEO_KLASORU = BURASI / "videolar"

# Olay kesitleri (otomatik once/sonra klipleri) kategoriye gore AYRI alt
# klasorlere yazilir - hepsi videolar/ altinda.
KOL_SOL_KLASORU = VIDEO_KLASORU / "sol_kol"
KOL_SAG_KLASORU = VIDEO_KLASORU / "sag_kol"
GOZ_KIRPMA_KLASORU = VIDEO_KLASORU / "goz_kirpma"
GOZ_BAKISI_KLASORU = VIDEO_KLASORU / "goz_bakisi"

# Kayit dosya adlarindaki zaman damgasinin bicimi: gun_ay_yil_saat_dakika_saniye
ZAMAN_DAMGASI_FORMATI = "%d_%m_%Y_%H_%M_%S"

# --- Kenar/kol/kirpma esikleri ------------------------------------------------
# Kenar "kutusu" YUZE GORE (merkez_x, merkez_y, uzunluk - yuz genisligi)
# hesaplanir, ekrana sabit degildir - hareket ettikce kutu da SENINLE
# birlikte kayar. Degerler "yuz genisligi carpi bu oran" olarak merkezden
# mesafedir.
KENAR_MESAFE_YATAY = 0.30  # SOL/SAG icin merkezden mesafe
KENAR_MESAFE_UST = 0.20    # YUKARI icin merkezden mesafe (kucuk = daha kolay tetiklenir)
# ASAGI icin merkezden mesafe - BAKIS_YANAL_SIZINTI_K duzeltmesi asagi
# bakisin kendi gy sinyalini de biraz zayiflatiyor (ham ~-0.38 iken
# duzeltmeden sonra ~-0.23'e dusuyor) - 0.40'ta hemen hic tetiklenmiyordu,
# bu yuzden UST ile ayni seviyeye (0.20) cektik.
KENAR_MESAFE_ALT = 0.20

# OpenVINO modeli normalize edilmis 3D gaze VEKTORU (x,y,z) donduruyor -
# L2CS'teki gibi pitch/yaw radyan degil. "Duz bakis" olu bolgesi bu yuzden
# vektorun x/y bilesenleri uzerinden (boyutsuz, -1..1 araligi) tanimlaniyor.
ESIK_BAKIS_XY = 0.10

# CAPRAZ EKSEN SIZINTISI: 5 yonde (duz/yukari/asagi/sol/sag) ham gx/gy
# olculdu, sizinti TEK bir dogrusal sabitle (gx = k*gy gibi) ACIKLANAMADI -
# iki AYRI, ASIMETRIK olay var:
#   1) SADECE asagi bakinca (gy<0) gx'e POZITIF sizinti biniyor - yukari
#      bakinca (gy>0) bu sizinti YOK (gx~0 kaldi). Olculen: asagida
#      gx=+0.33..0.35 iken gy=-0.38 -> oran ~0.87.
#   2) YANLARA (sol VEYA sag, |gx| buyudukce) bakinca gy'e NEGATIF (asagi
#      yonlu) bir sizinti biniyor - bu YONDEN BAGIMSIZ, |gx| ile orantili.
#      Olculen: sagda gx=-0.43/gy=-0.20, solda gx=+0.52/gy=-0.24 -> ikisinde
#      de oran ~0.46.
# Bu muhtemelen modelin kendi zaafi (goz kapaklari asagi/yana bakista
# kirpintinin gorunumunu degistiriyor) - matematiksel bir hata degil, bu
# yuzden tek sabitle tam duzelmiyor, sadece BUYUK kismi giderilebiliyor.
BAKIS_ASAGI_SIZINTI_K = 0.87  # sadece gy<0 iken gx'ten cikarilir
BAKIS_YANAL_SIZINTI_K = 0.46  # her zaman |gx| ile orantili, gy'e eklenir

KOL_Y_ESIK = 0.08       # bilek-omuz y farki bu degerden KUCUKSE "ayni hizada" sayilir
DIRSEK_ACI_ESIK = 90.0  # derece - dirsek acisi bu degerin ALTINDAYSA kol "kivrik" sayilir

# HISTEREZIS: kol "kalkik" siniri civarinda tutulunca landmark titremesi
# esigi ileri-geri gecip sayaci ART ARDA artiriyordu. Bunu onlemek icin GIRIS
# ve CIKIS esikleri farkli (Schmitt trigger mantigi).
KOL_HISTEREZIS_Y = 0.04
KOL_HISTEREZIS_ACI = 15.0

# BlazePose (PoseLandmarker), kol/govde kadraj DISINDA kalsa bile 33 noktanin
# HEPSI icin bir tahmin uretir - sadece "visibility" skoru dusuk olur. Bu
# yuzden kol sayilirken ilgili noktalarin yeterince "gorunur" olmasi sart
# kosuluyor (bkz. gorsellik.gorunur_mu).
GORUNURLUK_ESIK = 0.35

ESIK_BLINK = 0.5

# --- Goz kirpintisi (OpenVINO gaze modeli icin) --------------------------------
# Sol/sag goz landmark noktalarindan cikarilan kutunun kac kat genisletilecegi
# - model, goz kapaklarinin biraz otesini de (kas/yanak baglami) gormek uzere
# egitilmis, bu yuzden tam goz aralıgindan biraz daha genis kesiyoruz.
GOZ_KIRPINTI_MARJI = 2.2

# --- OLAY KESITI (event clip) -------------------------------------------------
# SOL KOL / SAG KOL / KIRPMA / BAKIS sayaclarindan biri her artinca, o andan
# ONCEKI ve SONRAKI birkac saniyeyi birlikte tek bir MP4 olarak otomatik
# kaydeder (guvenlik kamerasi mantigi).
OLAY_ONCE_SANIYE = 2.0   # tetiklenmeden ONCEKI kac saniye dahil edilsin
OLAY_SONRA_SANIYE = 2.0  # tetiklendikten SONRAKI kac saniye dahil edilsin

# --- Yumusatma (EMA) ----------------------------------------------------------
# Yuz uzaklastikca/kuculdukce hem landmark hem gaze vektoru tahmini daha
# gurultulu oluyor, bu da oku titrek gosteriyordu. YUMUSATMA_ORANI
# KUCULTULURSE daha az titrer ama tepki gecikir; BUYUTULURSE daha hizli tepki
# verir ama daha titrek olur. 1.0 = yumusatma YOK.
YUMUSATMA_ORANI = 0.35

# SICRAMA (outlier) SINIRLAMASI: yuz kucukken/uzaktayken gaze vektorunun
# x/y bilesenlerinde bazen tek bir karede COK sapan bir tahmin ("sicrama")
# olusabiliyor - EMA yumusatma bunu yavaslatir ama tek basina yeterli degil.
# Bu yuzden EMA'dan ONCE, bir karede vektor bileseninin en fazla bu kadar
# degismesine izin veriyoruz - daha buyuk bir fark gelirse kirpip sadece bu
# kadarini kabul ediyoruz.
MAKS_BAKIS_SICRAMA = 0.35  # boyutsuz, vektor bileseni (-1..1) icin karede izin verilen maks degisim

VIDEO_FPS_VARSAYILAN = 15.0