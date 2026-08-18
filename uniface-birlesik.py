"""MobileGaze (UniFace) + MediaPipe: TEK webcam'de bakis yonu + goz kirpma +
el/kol/vucut iskeleti - HEPSI AYNI ANDA, AYNI PENCEREDE.

BU DOSYA, ana OpenVINO surumune (gaze_birlesik.py) ALTERNATIF bir bakis
motoru - L2CS-Net'in (l2cs_birlesik.py) yerini alan, UniFace kutuphanesinin
MobileGaze (bakis regresyonu) modelini kullanir. Kurulum:
    pip install "uniface[cpu]" opencv-python
    (NVIDIA GPU icin "uniface[cpu]" yerine "uniface[gpu]")
Agirliklar (secilen MobileGaze modeli) ilk calistirmada UniFace tarafindan
KENDI onbellegine otomatik indirilir - elle bir seye gerek yok.

PERFORMANS NOTU (ONEMLI - versiyon gecmisi): Bu dosya ONCEDEN yuz tespiti
icin RetinaFace de kullaniyordu (UniFace'in KENDI dedektoru), MediaPipe'in
FaceLandmarker'inin YANINDA, AYRI bir tam-kare CNN olarak. Bu, her karede
IKI ayri tam-kare yuz dedektoru + Pose + Hand + MobileGaze = 5 agin ayni
anda calismasi demekti ve gozle gorulur kasmaya (dusuk FPS) yol aciyordu.
RetinaFace TAMAMEN KALDIRILDI: artik gaze_birlesik.py'deki (OpenVINO)
mimariyle AYNI mantik kullaniliyor - zaten HER KARE calismak zorunda olan
MediaPipe FaceLandmarker'in urettigi 478 noktalik yuz mesh'inden bir
sinirlayici kutu (bbox) cikarilip DOGRUDAN MobileGaze'e besleniyor, ayri
bir dedektor calistirilmiyor. Yan fayda: WIDER FACE veri setinin (RetinaFace
agirliklarinin egitildigi, ticari kullanima kapali CC-BY-NC-ND lisansli
veri seti) lisans kisitlamasi bu dosya icin artik gecerli DEGIL - sadece
asagidaki MobileGaze/Gaze360 kisitlamasi kaldi (L2CS ile ayni durum).

LISANS NOTU (ONEMLI): UniFace'in KODU (MobileGaze sarmalayicisi) MIT
lisanslidir - bu kisim tamamen serbest. AMA MobileGaze'in ONCEDEN EGITILMIS
AGIRLIKLARI Gaze360 veri setiyle egitilmis (resmi belgelerindeki MAE tablosu
"Gaze360 test set" uzerinden veriliyor) - yani L2CS'teki AYNI kisitlama
burada da gecerli: Gaze360 lisansi "veri setinde egitilmis modeller dahil
hicbir turevin ticari kullanilamayacagini" acikca soyluyor. Ticari kullanim
icin hala gaze_birlesik.py'deki OpenVINO modeli (Apache 2.0) tercih edilmeli.

KENDINE YETERLI (self-contained) dosya: UniFace'e OZEL ayarlar (mimari
secimi, esikler, kenar mesafeleri) asagida BU DOSYANIN ICINDE tanimli -
ayarlar.py'deki OpenVINO'ya ozel degerlerden BAGIMSIZDIR. Kamera, kol/kirpma
esikleri, olay kesiti sureleri, kimlik kilidi genel ayarlari, klasor
yollari gibi BAKISTAN BAGIMSIZ ayarlar yine ORTAK ayarlar.py'den (A.) gelir.

KIMLIK KILIDI: iki AYRI kilit var - yuz (MediaPipe FaceLandmarker uzerinden,
HEM bakis HEM kirpma icin ORTAK kullanilir - artik RetinaFace'in kendi
kilidi yok, cunku RetinaFace yok) ve govde (kol icin). Kadraja ikinci bir
kisi girse bile ilk bulunan kisi takip edilmeye devam eder, pahali olan
MobileGaze cikarimi SADECE kilitli yuz icin cagrilir (bkz. gorsellik.
kilitli_aday_sec).

Kontroller:
  c = kalibre et (kameraya duz bakarken bas, bakis sapmasini sifirlar)
  r = kesit al (o anki kare, TUM overlay'lerle, "kesitler/" klasorune JPEG)
  v = video kaydi ac/kapat ("videolar/" klasorune MP4, gercek FPS ile)
  h = sayaclari ac/kapat (BASLANGICTA KAPALI; DURAKLATILDIYSA tespit/cizim
      NORMAL calismaya devam eder, sadece sayaclarin ARTMASI ve onlara bagli
      otomatik olay kesitleri durur)
  z = yakinlastirmayi ac/kapat (BASLANGICTA KAPALI; KAPALIYKEN ayarlar.
      DIJITAL_YAKINLASTIRMA ne olursa olsun goruntu HER ZAMAN 1x/tam kalir)
  q = cikis
"""
import time
import types

import cv2
import mediapipe as mp
import numpy as np

import ayarlar as A
import gorsellik as G
import kayit as K
import modeller as M

# --- Bu dosyaya OZEL UniFace/MobileGaze ayarlari (ortak ayarlar.py'den
# BAGIMSIZ) ----------------------------------------------------------------
AKTIF_GAZE_UNIFACE = True  # False yaparsan MobileGaze hic yuklenmez

# MobileGaze mimarisi - uniface.constants.GazeWeights uyesi adi (string).
# Secenekler: RESNET18, RESNET34 (varsayilan/onerilen), RESNET50,
# MOBILENETV2, MOBILEONE_S0 (en hafif). Daha kucuk = daha hizli ama daha
# az dogru (bkz. dosya basindaki lisans notu - hepsi Gaze360 ile egitilmis).
UNIFACE_GAZE_MIMARISI = "RESNET34"

# MobileGaze ONNX Runtime uzerinde calisir ve 'providers' parametresiyle
# hangi donanimda calisacagini SECEBILIR (bkz. UniFace belgeleri: Execution
# Providers). Listedeki ILK calisan saglayici kullanilir, geri kalanlar
# YEDEK/fallback'tir - CUDA kurulu/uyumlu degilse otomatik CPU'ya duser,
# hata VERMEZ. GPU icin gereken: NVIDIA GPU + surucu + CUDA 11.x/12.x +
# cuDNN 8.x + "pip install uniface[gpu]" (onnxruntime yerine
# onnxruntime-gpu kurar). Sadece CPU istersen tek elemanli
# ["CPUExecutionProvider"] yap.
# NOT: CUDA denendi ama bu makinede CUDA 13.x/cuDNN 9.x eksik oldugu icin
# calismiyordu, otomatik CPU'ya duessu (performansi ETKILEMEDI, sadece
# baslangicta kirmizi hata satirlari yaziyordu). RetinaFace'in kaldirilmasi
# zaten asil kasma sorununu cozdugu ve CPU'da performans yeterli oldugu icin
# CUDA denemesi tamamen KAPATILDI - baslangic artik sessiz. CUDA kurup
# denemek istersen: ["CUDAExecutionProvider", "CPUExecutionProvider"] yap.
UNIFACE_ONNX_SAGLAYICILAR = ["CPUExecutionProvider"]

UNIFACE_ESIK_ACI = 0.08              # radyan, ~4.5 derece - duz bakis "olu bolgesi"
UNIFACE_MAKS_ACI_SICRAMA = 0.35      # radyan, ~20 derece - karede izin verilen maks pitch/yaw degisimi
UNIFACE_KENAR_MESAFE_YATAY = 0.50    # SOL/SAG icin merkezden mesafe (yuz genisligi carpani)
UNIFACE_KENAR_MESAFE_UST = 0.20      # YUKARI icin merkezden mesafe
UNIFACE_KENAR_MESAFE_ALT = 0.40      # ASAGI icin merkezden mesafe

# MediaPipe'in 478 noktalik mesh'inden cikan bbox, RetinaFace'in urettigi
# kutudan biraz daha DAR olabilir (RetinaFace tipik olarak biraz pay
# birakir) - MobileGaze'in egitim sirasinda gordugu kirpintiya daha yakin
# olmasi icin kutuyu bu oranda buyutuyoruz (1.0 = pay yok).
UNIFACE_YUZ_KIRPINTI_MARJI = 1.2


def _mobilegaze_yukle():
    """MobileGaze (bakis regresyonu) yukler. Yuz tespiti ARTIK RetinaFace ile
    degil, zaten her karede calisan MediaPipe FaceLandmarker'in bbox'iyla
    yapiliyor (bkz. dosya basindaki PERFORMANS NOTU) - bu yuzden burada
    SADECE MobileGaze var, ayri bir dedektor yuklenmiyor.

    NOT: uniface import'u BILEREK burada, fonksiyon icinde (lazy) -
    AKTIF_GAZE_UNIFACE=False iken bu fonksiyon hic cagrilmiyor, yani bu
    import'un (ve ilk calistirmada agirlik indirmesinin) suresi de HIC
    harcanmiyor. Donus: gaze_estimator.
    """
    from uniface.constants import GazeWeights
    from uniface.gaze import MobileGaze

    _t1 = time.time()
    model_adi = getattr(GazeWeights, UNIFACE_GAZE_MIMARISI)
    gaze_estimator = MobileGaze(model_name=model_adi, providers=UNIFACE_ONNX_SAGLAYICILAR)
    print(f"[zaman] MobileGaze ({UNIFACE_GAZE_MIMARISI}) yuklendi: {time.time() - _t1:.1f}s "
          f"- istenen saglayicilar: {UNIFACE_ONNX_SAGLAYICILAR}")

    try:
        import onnxruntime as ort
        print(f"[bilgi] ONNX Runtime'da GERCEKTEN kurulu/gorunen saglayicilar: {ort.get_available_providers()}")
    except Exception:
        pass

    return gaze_estimator


def _bbox_marjli(x1, y1, x2, y2, marj, w, h):
    """Bir bbox'u kendi merkezi etrafinda 'marj' orani kadar buyutur, goruntu
    sinirlarina kirpar."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    yeni_genislik = (x2 - x1) * marj
    yeni_yukseklik = (y2 - y1) * marj
    nx1 = max(0, int(cx - yeni_genislik / 2))
    ny1 = max(0, int(cy - yeni_yukseklik / 2))
    nx2 = min(w, int(cx + yeni_genislik / 2))
    ny2 = min(h, int(cy + yeni_yukseklik / 2))
    return nx1, ny1, nx2, ny2


# --- Modelleri yukle ----------------------------------------------------
if AKTIF_GAZE_UNIFACE:
    gaze_estimator = _mobilegaze_yukle()
else:
    gaze_estimator = None
    print("[bilgi] AKTIF_GAZE_UNIFACE=False - MobileGaze YUKLENMEDI. Bakis yonu (SOL/SAG/YUKARI/ASAGI) sayaclari ve yuz oku calismayacak.")

_t2 = time.time()
cap = cv2.VideoCapture(A.KAMERA_INDEKSI)
if not cap.isOpened():
    raise SystemExit(
        f"Webcam acilamadi (indeks {A.KAMERA_INDEKSI}). Baska uygulama "
        "kullaniyor olabilir ya da ayarlar.py'deki KAMERA_INDEKSI yanlis "
        "kamerayi gosteriyor olabilir (0, 1, 2... dene)."
    )
if A.KAMERA_COZUNURLUK_ZORLA:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*A.KAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, A.KAMERA_GENISLIK)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, A.KAMERA_YUKSEKLIK)
    cap.set(cv2.CAP_PROP_FPS, A.KAMERA_FPS)
gercek_genislik = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
gercek_yukseklik = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
gercek_fps = cap.get(cv2.CAP_PROP_FPS)
print(
    f"[zaman] webcam acildi (indeks {A.KAMERA_INDEKSI}): {time.time() - _t2:.1f}s "
    f"- kameranin kendi/varsayilan modu: {gercek_genislik}x{gercek_yukseklik}@{gercek_fps:.0f}fps"
)

face_landmarker, pose_landmarker, hand_landmarker = M.mediapipe_landmarker_lari_yukle()

PoseLandmark = mp.tasks.vision.PoseLandmark  # LEFT_WRIST, LEFT_SHOULDER, vb.

# --- Sayaclar / durum -----------------------------------------------------
sayaclar = {
    "sag": 0, "sol": 0, "yukari": 0, "asagi": 0, "kirpma": 0, "kesit": 0,
    "sol_kol": 0, "sag_kol": 0, "sol_bacak": 0, "sag_bacak": 0,
}
# 'h' tusuyla ac/kapa - False iken TESPIT/CIZIM NORMAL calismaya devam eder,
# sadece sayaclarin ARTMASI (ve onlara bagli otomatik olay kesitleri) durur.
# BASLANGICTA KAPALI - yanlislikla sayim baslamasin diye 'h' ile ACMAN gerekir.
sayaclar_aktif = False
# 'z' tusuyla ac/kapa - False iken yakinlastirma ORANI NE OLURSA OLSUN (bkz.
# ayarlar.DIJITAL_YAKINLASTIRMA) her kare TAM GORUNTUDE (1x) kalir.
# BASLANGICTA KAPALI - 'z' ile ACMAN gerekir.
yakinlastirma_aktif = False
onceki_yatay = "merkez"
onceki_dikey = "merkez"
onceki_sol_kol_aktif = False
onceki_sag_kol_aktif = False
gcs_test_aktif = False  # 'g' tusuyla GCS motor tepki testi (M2-M5, SEZGISEL)
gcs_test_baslangic_zaman = None
gcs_sol_omuz_baslangic_y = None
gcs_sag_omuz_baslangic_y = None
gcs_sol_dirsek_baslangic_acisi = None
gcs_sag_dirsek_baslangic_acisi = None
gcs_sol_ornekler = []
gcs_sag_ornekler = []
gcs_son_sonuc = None            # (sol_etiket, sag_etiket) - son test sonucu
gcs_sonuc_gosterim_bitis = 0.0  # time.time() bu degere kadar sonuc ekranda kalir
sol_kol_cikis_sayaci = 0  # debounce - bkz. gorsellik.hareket_algila
sag_kol_cikis_sayaci = 0
# Kol hareketi ARTIK bacaktaki ile ayni yontemle (hareket_algila, yone
# bagimsiz) algilaniyor - bilek icin HIZLI/YAVAS EMA durumu.
sol_kol_hizli_x = sol_kol_hizli_y = None
sol_kol_yavas_x = sol_kol_yavas_y = None
sag_kol_hizli_x = sag_kol_hizli_y = None
sag_kol_yavas_x = sag_kol_yavas_y = None

# Bacak/ayak hareketi (bkz. gorsellik.hareket_algila) - her ayak bilegi icin
# AYRI "durgun referans" konumu (None = henuz yok) ve AYRI hareketli/durgun
# durumu (histerezis icin).
sol_bacak_hizli_x = sol_bacak_hizli_y = None
sol_bacak_yavas_x = sol_bacak_yavas_y = None
sag_bacak_hizli_x = sag_bacak_hizli_y = None
sag_bacak_yavas_x = sag_bacak_yavas_y = None
onceki_sol_bacak_hareketli = False
onceki_sag_bacak_hareketli = False
sol_bacak_cikis_sayaci = 0  # debounce - bkz. gorsellik.hareket_algila
sag_bacak_cikis_sayaci = 0

# Kimlik kilidi durumu - IKI kilit: yuz (bakis + kirpma ORTAK - artik ikisi
# de AYNI MediaPipe FaceLandmarker sonucunu kullaniyor) ve govde (kol).
kilitli_yuz_merkez = None
yuz_kayip_kare = 0
kilitli_govde_merkez = None
govde_kayip_kare = 0

# TAKIP EDEN dijital yakinlastirma icin durum - kilitli yuzun SON bilinen
# konumu, HAM (kirpilmamis) kamera karesinin koordinatlarinda (bkz.
# gorsellik.takip_yakinlastir / raw_konuma_cevir, ayarlar.
# DIJITAL_YAKINLASTIRMA). None = henuz kilit yok / kilit tamamen birakildi
# -> o kare TAM ORTADAN (genis/arama modunda) kirpilir.
takip_merkezi = None

# Yuz henuz bulunmadan once (ilk kareler) cizilmesin diye baslangic degerleri.
cizgi_sol_x = cizgi_sag_x = cizgi_ust_y = cizgi_alt_y = None

BIAS_PITCH = 0.0
BIAS_YAW = 0.0
son_pitch_ham = 0.0
son_yaw_ham = 0.0
yuz_bulundu_bu_kare = False

yumusak_pitch = None
yumusak_yaw = None
yumusak_merkez_x = None
yumusak_merkez_y = None
yumusak_uzunluk = None

goz_kapali_onceki = False
kare_zaman_damgasi_ms = 0  # face/pose/hand landmarker'larin UCU icin ortak, artan sahte zaman damgasi

kaydedici = K.VideoKaydedici()

# Kategori basina AYRI olay kesiti yoneticisi - her biri kendi klasorune,
# kendi (bagimsiz) once/sonra penceresiyle yazar - manuel 'v' kaydindan
# BAGIMSIZ, her zaman arka planda calisir.
sol_kol_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.KOL_SOL_KLASORU,
    dosya_on_eki="sol_kol",
)
sag_kol_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.KOL_SAG_KLASORU,
    dosya_on_eki="sag_kol",
)
sol_bacak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.BACAK_SOL_KLASORU,
    dosya_on_eki="sol_bacak",
)
sag_bacak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.BACAK_SAG_KLASORU,
    dosya_on_eki="sag_bacak",
)
kirpma_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_KIRPMA_KLASORU,
    dosya_on_eki="kirpma",
)
bakis_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE,
    sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_BAKISI_KLASORU,
    dosya_on_eki="bakis",
)

print("Kameraya DUZ bakip 'c' ile kalibre et. 'r' = kesit al. 'v' = video kaydi ac/kapat. Cikis: 'q'.")

ilk_kare_mi = True

while True:
    ok, kare = cap.read()
    if not ok:
        break

    # Yakinlastirma en basta uygulanir - sonrasindaki HER SEY (tespit,
    # cizim, kayit) zaten yakinlastirilmis kare uzerinden calisir. TAKIP
    # EDEN: kirpma alani takip_merkezi'ni (bir onceki karede bulunan yuz
    # konumu) izler, sabit merkez DEGIL - bkz. ayarlar.DIJITAL_YAKINLASTIRMA.
    # Kilitli kimse YOKSA (henuz bulunamadi / kaybedildi) 1x'e (TAM GORUNTU)
    # duser - dar/yakinlastirilmis bir alanda "arama" yapip kisiyi kadrajin
    # geri kalaninda gormeme riskini onler.
    _etkin_yakinlastirma = A.DIJITAL_YAKINLASTIRMA if (yakinlastirma_aktif and takip_merkezi is not None) else 1.0
    kare, _kirpma_dikdortgeni = G.takip_yakinlastir(kare, takip_merkezi, _etkin_yakinlastirma)

    if ilk_kare_mi:
        _t6 = time.time()

    h, w = kare.shape[:2]
    yuz_bulundu_bu_kare = False

    # --- MediaPipe: bakis + kirpma + govde + eller (AYNI kare, AYNI mp_image,
    # TEK yuz dedektoru - RetinaFace KALDIRILDI, bkz. dosya basindaki
    # PERFORMANS NOTU) --------------------------------------------------
    try:
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        kare_zaman_damgasi_ms += 33

        landmarker_sonuc = face_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

        # --- Kimlik kilidi: adaylar arasindan "kilitli kisiyi" sec. Bu
        # secim ARTIK HEM bakis HEM kirpma icin ORTAK kullaniliyor.
        secilen_yuz_i = None
        if landmarker_sonuc.face_landmarks:
            if A.KIMLIK_KILIDI_AKTIF:
                yuz_merkezleri = []
                yuz_buyuklukleri = []
                for aday in landmarker_sonuc.face_landmarks:
                    ax1, ay1, ax2, ay2 = G.yuz_bbox_hesapla(aday, w, h)
                    yuz_merkezleri.append(((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0))
                    yuz_buyuklukleri.append(ax2 - ax1)
                secilen_yuz_i, kilitli_yuz_merkez, yuz_kayip_kare = G.kilitli_aday_sec(
                    kilitli_yuz_merkez, yuz_kayip_kare, yuz_merkezleri, yuz_buyuklukleri,
                    A.KIMLIK_KILIDI_MAKS_SICRAMA_ORANI, A.KIMLIK_KILIDI_KAYIP_KARE_LIMITI,
                )
            else:
                secilen_yuz_i = 0

        # TAKIP EDEN yakinlastirma icin: bu karede GUVENLE bir yuz
        # bulunduysa (secilen_yuz_i) konumunu HAM kamera koordinatlarina
        # cevirip bir sonraki karenin kirpma merkezi olarak sakla.
        # BULUNAMADIYSA (kimse yok / kimlik kilidi henuz eslesme bekliyor)
        # yakinlastirmayi HEMEN 1x'e dusur - dar/yakinlastirilmis bir alanda
        # "kor" sekilde aramaya devam ETMESIN, bir sonraki kare TAM
        # GORUNTUYE doner (bkz. yukaridaki _etkin_yakinlastirma).
        if secilen_yuz_i is not None:
            _ax1, _ay1, _ax2, _ay2 = G.yuz_bbox_hesapla(landmarker_sonuc.face_landmarks[secilen_yuz_i], w, h)
            takip_merkezi = G.raw_konuma_cevir(
                (_ax1 + _ax2) / 2.0, (_ay1 + _ay2) / 2.0, _kirpma_dikdortgeni, w, h,
            )
        else:
            takip_merkezi = None

        # --- Bakis (MobileGaze) - secilen yuzun landmark bbox'indan kirpilir,
        # ayri bir dedektor CALISTIRILMIYOR. ---------------------------------
        if AKTIF_GAZE_UNIFACE and gaze_estimator is not None and secilen_yuz_i is not None:
            aday = landmarker_sonuc.face_landmarks[secilen_yuz_i]
            fx1, fy1, fx2, fy2 = G.yuz_bbox_hesapla(aday, w, h)
            x1, y1, x2, y2 = _bbox_marjli(fx1, fy1, fx2, fy2, UNIFACE_YUZ_KIRPINTI_MARJI, w, h)
            yuz_kirpintisi = kare[y1:y2, x1:x2]

            if yuz_kirpintisi.size > 0:
                yuz_bulundu_bu_kare = True

                sonuc = gaze_estimator.estimate(yuz_kirpintisi)
                son_pitch_ham = float(sonuc.pitch)
                son_yaw_ham = float(sonuc.yaw)

                pitch = son_pitch_ham - BIAS_PITCH
                yaw = son_yaw_ham - BIAS_YAW

                if A.YUZ_CIZIMI_GOSTER:
                    cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 1)

                merkez_x = (x1 + x2) / 2.0
                merkez_y = (y1 + y2) / 2.0
                uzunluk = x2 - x1

                # Yumusatma: bkz. gorsellik.yumusat - ani "sicrama" (outlier)
                # degerlerin oku bir anda yanlis yone firlatmasini engeller.
                pitch = yumusak_pitch = G.yumusat(yumusak_pitch, pitch, UNIFACE_MAKS_ACI_SICRAMA)
                yaw = yumusak_yaw = G.yumusat(yumusak_yaw, yaw, UNIFACE_MAKS_ACI_SICRAMA)
                merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
                merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
                uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

                cizgi_sol_x = int(merkez_x - uzunluk * UNIFACE_KENAR_MESAFE_YATAY)
                cizgi_sag_x = int(merkez_x + uzunluk * UNIFACE_KENAR_MESAFE_YATAY)
                cizgi_ust_y = int(merkez_y - uzunluk * UNIFACE_KENAR_MESAFE_UST)
                cizgi_alt_y = int(merkez_y + uzunluk * UNIFACE_KENAR_MESAFE_ALT)

                # UniFace'in KENDI belgelenen formulu (pitch + = yukari,
                # yaw + = saga - L2CS'inkinden FARKLI/duz bir eksen
                # kuralı, ekseni "ters atma" gerekmiyor).
                dx = -uzunluk * np.sin(yaw) * np.cos(pitch)
                dy = -uzunluk * np.sin(pitch)
                ucur_x = merkez_x + dx
                ucur_y = merkez_y + dy

                duz_bakiyor = abs(pitch) < UNIFACE_ESIK_ACI and abs(yaw) < UNIFACE_ESIK_ACI

                if A.YUZ_CIZIMI_GOSTER and not duz_bakiyor:
                    cv2.arrowedLine(
                        kare, (int(merkez_x), int(merkez_y)), (int(ucur_x), int(ucur_y)),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18,
                    )

                yatay = "sol" if ucur_x > cizgi_sag_x else "sag" if ucur_x < cizgi_sol_x else "merkez"
                dikey = "asagi" if ucur_y > cizgi_alt_y else "yukari" if ucur_y < cizgi_ust_y else "merkez"

                if yatay != "merkez" and yatay != onceki_yatay:
                    if sayaclar_aktif:
                        sayaclar[yatay] += 1
                        if A.BAKIS_OLAY_KESITI_AKTIF:
                            bakis_olay_kaydedici.olay_tetikle(yatay)
                onceki_yatay = yatay

                if dikey != "merkez" and dikey != onceki_dikey:
                    if sayaclar_aktif:
                        sayaclar[dikey] += 1
                        if A.BAKIS_OLAY_KESITI_AKTIF:
                            bakis_olay_kaydedici.olay_tetikle(dikey)
                onceki_dikey = dikey

        # --- Kirpma - AYNI secilen_yuz_i, blendshape skorlarindan. ----------
        if secilen_yuz_i is not None and landmarker_sonuc.face_blendshapes:
            skorlar = {b.category_name: b.score for b in landmarker_sonuc.face_blendshapes[secilen_yuz_i]}
            sol_kirpma = skorlar.get("eyeBlinkLeft", 0.0)
            sag_kirpma = skorlar.get("eyeBlinkRight", 0.0)
            kirpma_skoru = (sol_kirpma + sag_kirpma) / 2.0

            goz_kapali_simdi = kirpma_skoru > A.ESIK_BLINK
            if goz_kapali_simdi and not goz_kapali_onceki:
                if sayaclar_aktif:
                    sayaclar["kirpma"] += 1
                    if A.KIRPMA_OLAY_KESITI_AKTIF:
                        kirpma_olay_kaydedici.olay_tetikle("kirpma")
            goz_kapali_onceki = goz_kapali_simdi

        if A.AKTIF_POSE and pose_landmarker is not None:
            pose_sonuc = pose_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

            secilen_govde_i = None
            if pose_sonuc.pose_landmarks:
                if A.KIMLIK_KILIDI_AKTIF:
                    govde_merkezleri = []
                    govde_buyuklukleri = []
                    for aday in pose_sonuc.pose_landmarks:
                        aday_sol_omuz = aday[PoseLandmark.LEFT_SHOULDER]
                        aday_sag_omuz = aday[PoseLandmark.RIGHT_SHOULDER]
                        govde_merkezleri.append((
                            (aday_sol_omuz.x + aday_sag_omuz.x) / 2.0 * w,
                            (aday_sol_omuz.y + aday_sag_omuz.y) / 2.0 * h,
                        ))
                        omuz_genisligi = G.omuz_genisligi_piksel(aday_sol_omuz, aday_sag_omuz, w, h)
                        govde_buyuklukleri.append(max(omuz_genisligi, 1.0))
                    secilen_govde_i, kilitli_govde_merkez, govde_kayip_kare = G.kilitli_aday_sec(
                        kilitli_govde_merkez, govde_kayip_kare, govde_merkezleri, govde_buyuklukleri,
                        A.KIMLIK_KILIDI_MAKS_SICRAMA_ORANI, A.KIMLIK_KILIDI_KAYIP_KARE_LIMITI,
                    )
                else:
                    secilen_govde_i = 0

            if A.GOVDE_CIZIMI_GOSTER:
                _cizilecek = [pose_sonuc.pose_landmarks[secilen_govde_i]] if secilen_govde_i is not None else []
                G.govde_ciz(kare, types.SimpleNamespace(pose_landmarks=_cizilecek))

            # --- Kol sayaci: IKI tetikleyiciden HERHANGI BIRI olusunca (asagidan-
            # yukari GECIS anini yakalayip) sayaci artirir - bkz. dosya basindaki
            # docstring. Normalize y kuculdukce ekranda yukari demektir.
            if secilen_govde_i is not None:
                lm = pose_sonuc.pose_landmarks[secilen_govde_i]

                sol_omuz = lm[PoseLandmark.LEFT_SHOULDER]
                sol_dirsek = lm[PoseLandmark.LEFT_ELBOW]
                sol_bilek = lm[PoseLandmark.LEFT_WRIST]
                sag_bilek_ham = lm[PoseLandmark.RIGHT_WRIST]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_bilek, sag_bilek_ham = G.ekran_sol_sag_ayikla(sol_bilek, sag_bilek_ham)
                if G.gorunur_mu(sol_bilek):
                    (sol_kol_aktif, sol_kol_hizli_x, sol_kol_hizli_y,
                     sol_kol_yavas_x, sol_kol_yavas_y, sol_kol_cikis_sayaci) = G.hareket_algila(
                        sol_kol_hizli_x, sol_kol_hizli_y, sol_kol_yavas_x, sol_kol_yavas_y,
                        sol_bilek.x, sol_bilek.y,
                        onceki_sol_kol_aktif, A.KOL_HAREKET_ESIK,
                        A.KOL_HAREKET_HISTEREZIS_ORANI, A.KOL_HAREKET_HIZLI_ORAN, A.KOL_HAREKET_YAVAS_ORAN,
                        sol_kol_cikis_sayaci, A.KOL_HAREKET_MIN_CIKIS_KARE,
                    )
                    if sol_kol_aktif and not onceki_sol_kol_aktif:
                        if sayaclar_aktif:
                            sayaclar["sol_kol"] += 1
                            if A.KOL_OLAY_KESITI_AKTIF:
                                sol_kol_olay_kaydedici.olay_tetikle("sol_kol")
                    onceki_sol_kol_aktif = sol_kol_aktif

                sag_omuz = lm[PoseLandmark.RIGHT_SHOULDER]
                sag_dirsek = lm[PoseLandmark.RIGHT_ELBOW]
                sag_bilek = sag_bilek_ham
                if G.gorunur_mu(sag_bilek):
                    (sag_kol_aktif, sag_kol_hizli_x, sag_kol_hizli_y,
                     sag_kol_yavas_x, sag_kol_yavas_y, sag_kol_cikis_sayaci) = G.hareket_algila(
                        sag_kol_hizli_x, sag_kol_hizli_y, sag_kol_yavas_x, sag_kol_yavas_y,
                        sag_bilek.x, sag_bilek.y,
                        onceki_sag_kol_aktif, A.KOL_HAREKET_ESIK,
                        A.KOL_HAREKET_HISTEREZIS_ORANI, A.KOL_HAREKET_HIZLI_ORAN, A.KOL_HAREKET_YAVAS_ORAN,
                        sag_kol_cikis_sayaci, A.KOL_HAREKET_MIN_CIKIS_KARE,
                    )
                    if sag_kol_aktif and not onceki_sag_kol_aktif:
                        if sayaclar_aktif:
                            sayaclar["sag_kol"] += 1
                            if A.KOL_OLAY_KESITI_AKTIF:
                                sag_kol_olay_kaydedici.olay_tetikle("sag_kol")
                    onceki_sag_kol_aktif = sag_kol_aktif

                # --- GCS Motor Tepkisi Testi (M2-M5, SEZGISEL - bkz.
                # gorsellik.gcs_kol_tepkisini_sinifla docstring'i, KESIN
                # tibbi olcum DEGIL). 'g' ile baslatilan pencere boyunca
                # her iki kolun dirsek acisi + bilek konumu ornekleniyor.
                if gcs_test_aktif:
                    if G.gorunur_mu(sol_bilek) and G.gorunur_mu(sol_dirsek):
                        _gcs_sol_aci = G.dirsek_acisi_derece(sol_omuz, sol_dirsek, sol_bilek)
                        gcs_sol_ornekler.append((_gcs_sol_aci, sol_bilek.x, sol_bilek.y))
                        if gcs_sol_omuz_baslangic_y is None:
                            gcs_sol_omuz_baslangic_y = sol_omuz.y
                            gcs_sol_dirsek_baslangic_acisi = _gcs_sol_aci
                    if G.gorunur_mu(sag_bilek) and G.gorunur_mu(sag_dirsek):
                        _gcs_sag_aci = G.dirsek_acisi_derece(sag_omuz, sag_dirsek, sag_bilek)
                        gcs_sag_ornekler.append((_gcs_sag_aci, sag_bilek.x, sag_bilek.y))
                        if gcs_sag_omuz_baslangic_y is None:
                            gcs_sag_omuz_baslangic_y = sag_omuz.y
                            gcs_sag_dirsek_baslangic_acisi = _gcs_sag_aci

                    if time.time() - gcs_test_baslangic_zaman >= A.GCS_PENCERE_SANIYE:
                        _gcs_sol_etiket, _gcs_sol_detay = G.gcs_kol_tepkisini_sinifla(
                            gcs_sol_ornekler, gcs_sol_dirsek_baslangic_acisi, gcs_sol_omuz_baslangic_y)
                        _gcs_sag_etiket, _gcs_sag_detay = G.gcs_kol_tepkisini_sinifla(
                            gcs_sag_ornekler, gcs_sag_dirsek_baslangic_acisi, gcs_sag_omuz_baslangic_y)
                        gcs_son_sonuc = (_gcs_sol_etiket, _gcs_sag_etiket)
                        print(f"[GCS] SOL KOL: {_gcs_sol_etiket or ('IZLENEMEDI' if _gcs_sol_detay.get('sebep')=='izlenemedi' else 'TEPKI YOK (M1 olabilir - kamera/gorunurluk sorunu da olabilir, kontrol et)')}  detay={_gcs_sol_detay}")
                        print(f"[GCS] SAG KOL: {_gcs_sag_etiket or ('IZLENEMEDI' if _gcs_sag_detay.get('sebep')=='izlenemedi' else 'TEPKI YOK (M1 olabilir - kamera/gorunurluk sorunu da olabilir, kontrol et)')}  detay={_gcs_sag_detay}")
                        print("[GCS] UYARI: Bu SEZGISEL bir ON-ONERI, kesin tibbi olcum degil - klinisyen kendi gozlemiyle dogrulamali.")
                        gcs_sonuc_gosterim_bitis = time.time() + A.GCS_SONUC_GOSTERIM_SANIYE
                        gcs_test_aktif = False

                # --- Bacak/ayak hareketi: ayak bilegi konumundaki ANI
                # degisim (herhangi bir YONE) - bkz. gorsellik.hareket_algila
                # ve ayarlar.py'deki BACAK_HAREKET_* aciklamasi. KOL'un
                # aksine "yukari kalkti mi" gibi bir yon sarti YOK, bu yuzden
                # yatan hastada da calisir.
                sol_ayak_bilegi = lm[PoseLandmark.LEFT_ANKLE]
                sag_ayak_bilegi_ham = lm[PoseLandmark.RIGHT_ANKLE]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_ayak_bilegi, sag_ayak_bilegi_ham = G.ekran_sol_sag_ayikla(sol_ayak_bilegi, sag_ayak_bilegi_ham)
                if G.gorunur_mu(sol_ayak_bilegi):
                    (sol_bacak_hareketli, sol_bacak_hizli_x, sol_bacak_hizli_y,
                     sol_bacak_yavas_x, sol_bacak_yavas_y, sol_bacak_cikis_sayaci) = G.hareket_algila(
                        sol_bacak_hizli_x, sol_bacak_hizli_y, sol_bacak_yavas_x, sol_bacak_yavas_y,
                        sol_ayak_bilegi.x, sol_ayak_bilegi.y,
                        onceki_sol_bacak_hareketli, A.BACAK_HAREKET_ESIK,
                        A.BACAK_HAREKET_HISTEREZIS_ORANI, A.BACAK_HAREKET_HIZLI_ORAN, A.BACAK_HAREKET_YAVAS_ORAN,
                        sol_bacak_cikis_sayaci, A.BACAK_HAREKET_MIN_CIKIS_KARE,
                    )
                    if sol_bacak_hareketli and not onceki_sol_bacak_hareketli:
                        if sayaclar_aktif:
                            sayaclar["sol_bacak"] += 1
                            if A.BACAK_OLAY_KESITI_AKTIF:
                                sol_bacak_olay_kaydedici.olay_tetikle("sol_bacak")
                    onceki_sol_bacak_hareketli = sol_bacak_hareketli

                sag_ayak_bilegi = sag_ayak_bilegi_ham
                if G.gorunur_mu(sag_ayak_bilegi):
                    (sag_bacak_hareketli, sag_bacak_hizli_x, sag_bacak_hizli_y,
                     sag_bacak_yavas_x, sag_bacak_yavas_y, sag_bacak_cikis_sayaci) = G.hareket_algila(
                        sag_bacak_hizli_x, sag_bacak_hizli_y, sag_bacak_yavas_x, sag_bacak_yavas_y,
                        sag_ayak_bilegi.x, sag_ayak_bilegi.y,
                        onceki_sag_bacak_hareketli, A.BACAK_HAREKET_ESIK,
                        A.BACAK_HAREKET_HISTEREZIS_ORANI, A.BACAK_HAREKET_HIZLI_ORAN, A.BACAK_HAREKET_YAVAS_ORAN,
                        sag_bacak_cikis_sayaci, A.BACAK_HAREKET_MIN_CIKIS_KARE,
                    )
                    if sag_bacak_hareketli and not onceki_sag_bacak_hareketli:
                        if sayaclar_aktif:
                            sayaclar["sag_bacak"] += 1
                            if A.BACAK_OLAY_KESITI_AKTIF:
                                sag_bacak_olay_kaydedici.olay_tetikle("sag_bacak")
                    onceki_sag_bacak_hareketli = sag_bacak_hareketli

        if A.AKTIF_EL and hand_landmarker is not None:
            hand_sonuc = hand_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
            if A.EL_CIZIMI_GOSTER:
                G.eller_ciz(kare, hand_sonuc)
    except Exception:
        pass

    # --- Yuze bagli kenar kutusu + sayaclar (en ustte, en sonda ciziliyor ki
    # okunsun). Kutu SADECE bu karede yuz bulunduysa cizilir.
    renk = (255, 255, 0)
    if yuz_bulundu_bu_kare and cizgi_sol_x is not None:
        cv2.rectangle(kare, (cizgi_sol_x, cizgi_ust_y), (cizgi_sag_x, cizgi_alt_y), renk, 1)

    cv2.putText(kare, f"KIRPMA: {sayaclar['kirpma']}   KESIT: {sayaclar['kesit']}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(kare, f"SOL: {sayaclar['sol']}  SAG: {sayaclar['sag']}  YUKARI: {sayaclar['yukari']}  ASAGI: {sayaclar['asagi']}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
    cv2.putText(kare, f"SOL KOL: {sayaclar['sol_kol']}   SAG KOL: {sayaclar['sag_kol']}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(kare, f"SOL BACAK: {sayaclar['sol_bacak']}   SAG BACAK: {sayaclar['sag_bacak']}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(
        kare, f"SAYAÇLAR: {'AKTIF' if sayaclar_aktif else 'DURAKLATILDI'} (h)",
        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0) if sayaclar_aktif else (0, 0, 255), 2,
    )
    cv2.putText(
        kare, f"YAKINLAŞTIRMA: {'AKTIF' if yakinlastirma_aktif else 'KAPALI'} (z)",
        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0) if yakinlastirma_aktif else (0, 0, 255), 2,
    )
    if gcs_test_aktif:
        _gcs_kalan = max(0.0, A.GCS_PENCERE_SANIYE - (time.time() - gcs_test_baslangic_zaman))
        cv2.putText(kare, f"GCS TESTI: OLCULUYOR ({_gcs_kalan:.1f}sn) - uyaran uygula",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    elif gcs_son_sonuc is not None and time.time() < gcs_sonuc_gosterim_bitis:
        _gcs_sol_e, _gcs_sag_e = gcs_son_sonuc
        cv2.putText(kare, f"GCS SONUC (sezgisel): SOL={_gcs_sol_e or '?'}  SAG={_gcs_sag_e or '?'}",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(kare, "GCS TESTI: HAZIR (g)", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    cv2.putText(kare, "c: kalibre et  |  r: kesit al  |  v: video kaydi  |  h: sayac ac/kapat  |  z: yakinlastirma ac/kapat  |  g: GCS testi  |  q: cikis",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if kaydedici.kayit_yapiliyor:
        cv2.circle(kare, (w - 30, 30), 8, (0, 0, 255), -1)
        cv2.putText(kare, "REC", (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Video kaydi - TUM overlay'ler cizildikten SONRA bellege ekleniyor.
    kaydedici.kare_ekle(kare)
    # Olay kesitleri - HER karede, SADECE aktif olan kategoriler icin cagrilir.
    if A.KOL_OLAY_KESITI_AKTIF:
        sol_kol_olay_kaydedici.kare_ekle(kare)
        sag_kol_olay_kaydedici.kare_ekle(kare)
    if A.BACAK_OLAY_KESITI_AKTIF:
        sol_bacak_olay_kaydedici.kare_ekle(kare)
        sag_bacak_olay_kaydedici.kare_ekle(kare)
    if A.KIRPMA_OLAY_KESITI_AKTIF:
        kirpma_olay_kaydedici.kare_ekle(kare)
    if A.BAKIS_OLAY_KESITI_AKTIF:
        bakis_olay_kaydedici.kare_ekle(kare)

    if ilk_kare_mi:
        print(f"[zaman] ilk kare islendi (modellerin ilk 'isinmasi' dahil): {time.time() - _t6:.1f}s")
        ilk_kare_mi = False

    cv2.imshow("UniFace (MobileGaze) + MediaPipe: bakis + kirpma + govde + eller (q = cik)", kare)
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        if kaydedici.kayit_yapiliyor:
            kaydedici.bitir()
        sol_kol_olay_kaydedici.bitir()
        sag_kol_olay_kaydedici.bitir()
        sol_bacak_olay_kaydedici.bitir()
        sag_bacak_olay_kaydedici.bitir()
        kirpma_olay_kaydedici.bitir()
        bakis_olay_kaydedici.bitir()
        break
    if tus == ord("c") and yuz_bulundu_bu_kare:
        BIAS_PITCH = son_pitch_ham
        BIAS_YAW = son_yaw_ham
        print(f"Kalibre edildi. BIAS_PITCH={BIAS_PITCH:.3f} BIAS_YAW={BIAS_YAW:.3f}")
    if tus == ord("r"):
        K.kesit_al(kare, sayaclar)
    if tus == ord("h"):
        sayaclar_aktif = not sayaclar_aktif
        print(f"Sayaçlar {'AKTIF' if sayaclar_aktif else 'DURAKLATILDI'}.")
    if tus == ord("z"):
        yakinlastirma_aktif = not yakinlastirma_aktif
        print(f"Yakınlaştırma {'AKTIF' if yakinlastirma_aktif else 'KAPALI'}.")
    if tus == ord("g"):
        if not gcs_test_aktif:
            gcs_test_aktif = True
            gcs_test_baslangic_zaman = time.time()
            gcs_sol_omuz_baslangic_y = None
            gcs_sag_omuz_baslangic_y = None
            gcs_sol_dirsek_baslangic_acisi = None
            gcs_sag_dirsek_baslangic_acisi = None
            gcs_sol_ornekler = []
            gcs_sag_ornekler = []
            print(f"[GCS] Test basladi - {A.GCS_PENCERE_SANIYE:.0f} saniye boyunca kollari "
                  "gozlemliyorum. SIMDI merkezi agrili uyarani (orn. sternal ovma) uygula.")
    if tus == ord("v"):
        if not kaydedici.kayit_yapiliyor:
            kaydedici.baslat()
        else:
            kaydedici.bitir()

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
if pose_landmarker is not None:
    pose_landmarker.close()
if hand_landmarker is not None:
    hand_landmarker.close()
if kaydedici.kayit_yapiliyor:
    kaydedici.bitir()
sol_kol_olay_kaydedici.bitir()
sag_kol_olay_kaydedici.bitir()
sol_bacak_olay_kaydedici.bitir()
sag_bacak_olay_kaydedici.bitir()
kirpma_olay_kaydedici.bitir()
bakis_olay_kaydedici.bitir()
print("Sayaclar:", sayaclar)