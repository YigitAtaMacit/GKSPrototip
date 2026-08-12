"""L2CS-Net + MediaPipe birlesik uygulamasi icin TUM ayarlar/sabitler/yollar.

Davranisi degistirmek istedigin zaman (esik degerleri, hangi cizimlerin
gorunecegi, hangi modellerin aktif olacagi vb.) SADECE bu dosyayi duzenlemen
yeterli - diger dosyalar (modeller.py, gorsellik.py, kayit.py,
l2cs_birlesik.py) buradaki degerleri kullanir.
"""
from pathlib import Path

BURASI = Path(__file__).resolve().parent

# --- Acik/kapali ozellikler --------------------------------------------------
# Performans icin acik/kapali yapilabilir modeller - yavas gelirse False yap.
AKTIF_POSE = True
AKTIF_EL = True

# Otomatik OLAY KESITI (once/sonra video klibi) kategori basina ac/kapa -
# simdilik UCU DE kapali. SAYACLAR (KIRPMA/SOL KOL/SAG KOL/SOL/SAG/YUKARI/
# ASAGI) bundan ETKILENMEZ, hepsi normal calismaya devam eder - sadece
# otomatik VIDEO KLIBI alinmiyor. Tekrar acmak istersen ilgiliyi True yap.
KOL_OLAY_KESITI_AKTIF = False
KIRPMA_OLAY_KESITI_AKTIF = False
BAKIS_OLAY_KESITI_AKTIF = False

# False yaparsan ilgili gorsel EKRANDA CIZILMEZ ama tespit (ve dolayisiyla
# ilgili sayaclar/kalibrasyon) YINE DE CALISIR - sadece gorsel gizlenir.
GOVDE_CIZIMI_GOSTER = False
EL_CIZIMI_GOSTER = False
YUZ_CIZIMI_GOSTER = True

# --- Dosya yollari -----------------------------------------------------------
AGIRLIK = BURASI / "L2CSNet_gaze360.pkl"

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
# (orn. "12_08_2026_14_05_33.mp4"). Saniye kismi ayni saat icinde birden fazla
# kayit olduysa dosya adlarinin CAKISMAMASI icin var - istersen "%d_%m_%Y_%H"
# yaparak sadece saate kadar kisaltabilirsin.
ZAMAN_DAMGASI_FORMATI = "%d_%m_%Y_%H_%M_%S"

# --- Kenar/kol/kirpma esikleri ------------------------------------------------
# Kenar "kutusu" YUZE GORE (merkez_x, merkez_y, uzunluk - yuz kutusunun
# genisligi) hesaplanir, ekrana sabit degildir - hareket ettikce kutu da
# SENINLE birlikte kayar. Degerler "yuz genisligi carpi bu oran" olarak
# merkezden mesafedir.
KENAR_MESAFE_YATAY = 0.50  # SOL/SAG icin merkezden mesafe
KENAR_MESAFE_UST = 0.20    # YUKARI icin merkezden mesafe (kucuk = daha kolay tetiklenir)
KENAR_MESAFE_ALT = 0.40    # ASAGI icin merkezden mesafe
ESIK_ACI = 0.08             # ~4.5 derece - duz bakis "olu bolgesi"

KOL_Y_ESIK = 0.08       # bilek-omuz y farki bu degerden KUCUKSE "ayni hizada" sayilir
DIRSEK_ACI_ESIK = 90.0  # derece - dirsek acisi bu degerin ALTINDAYSA kol "kivrik" sayilir

# HISTEREZIS: kol "kalkik" siniri civarinda tutulunca (orn. tam omuz
# hizasinda) landmark titremesi esigi ileri-geri gecip sayaci ART ARDA
# artiriyordu. Bunu onlemek icin GIRIS ve CIKIS esikleri farkli: kol AKTIF
# OLMAK icin normal esigi gecmeli, ama AKTIF KALIRKEN cikmak icin daha
# BELIRGIN sekilde indirilmesi/duzlesmesi gerekiyor (Schmitt trigger mantigi).
KOL_HISTEREZIS_Y = 0.04        # kalkik/degil siniri icin olu bolge (normalize y)
KOL_HISTEREZIS_ACI = 15.0      # derece - kivrik/degil siniri icin olu bolge

# BlazePose (PoseLandmarker), kol/govde kadraj DISINDA kalsa bile 33 noktanin
# HEPSI icin bir tahmin uretir - sadece "visibility" skoru dusuk olur. Bu
# yuzden kol sayilirken ilgili noktalarin yeterince "gorunur" olmasi sart
# kosuluyor (bkz. gorsellik.gorunur_mu).
GORUNURLUK_ESIK = 0.35

ESIK_BLINK = 0.5

# --- OLAY KESITI (event clip) -------------------------------------------------
# SOL KOL / SAG KOL / KIRPMA / BAKIS sayaclarindan biri her artinca, o andan
# ONCEKI ve SONRAKI birkac saniyeyi birlikte tek bir MP4 olarak otomatik
# kaydeder (guvenlik kamerasi mantigi). Dort kategori de AYNI sureleri
# kullanir - istersen her biri icin ayri sure de tanimlanabilir.
OLAY_ONCE_SANIYE = 2.0   # tetiklenmeden ONCEKI kac saniye dahil edilsin
OLAY_SONRA_SANIYE = 2.0  # tetiklendikten SONRAKI kac saniye dahil edilsin

# --- Yumusatma (EMA) ----------------------------------------------------------
# Yuz uzaklastikca/kuculdukce L2CS'in pitch/yaw ve RetinaFace'in bbox tahmini
# daha gurultulu oluyor, bu da oku titrek gosteriyordu. YUMUSATMA_ORANI
# KUCULTULURSE daha az titrer ama tepki gecikir; BUYUTULURSE daha hizli tepki
# verir ama daha titrek olur. 1.0 = yumusatma YOK.
YUMUSATMA_ORANI = 0.35

# SICRAMA (outlier) SINIRLAMASI: yuz kucukken/uzaktayken L2CS bazen tek bir
# karede COK sapan bir pitch/yaw tahmini (ani "sicrama") uretebiliyor - EMA
# yumusatma bunu yavaslatir ama tek basina yeterli degil, cunku yumusatilmis
# deger de o sicramaya dogru surukleniyor. Bu yuzden EMA'dan ONCE, bir
# karede acinin en fazla bu kadar (radyan) DEGISMESINE izin veriyoruz - daha
# buyuk bir fark gelirse kirpip sadece bu kadarini kabul ediyoruz. Boylece
# tek karelik "cilginca" sicramalar ok'u anlik olarak firlatamaz.
MAKS_ACI_SICRAMA = 0.35  # radyan, ~20 derece - karede izin verilen maks pitch/yaw degisimi

VIDEO_FPS_VARSAYILAN = 15.0