"""MediaPipe + Intel OpenVINO (gaze-estimation-adas-0002): TEK webcam'de
bakis yonu + goz kirpma + el/kol/vucut iskeleti - HEPSI AYNI ANDA, AYNI
PENCEREDE.

BU DOSYA, calistirdigin TEK dosya (`python gaze_birlesik.py`). Kod
okunabilirlik icin AYNI klasordeki kucuk modullere bolundu, ama hepsi bu
dosya uzerinden TEK PARCA halinde calisir:
  ayarlar.py       - TUM sabitler/esikler/acik-kapali bayraklar (davranisi
                      degistirmek istersen SADECE burayi duzenle)
  modeller.py       - OpenVINO gaze modeli + MediaPipe landmarker yukleme/indirme
  gorsellik.py       - iskelet cizimi + geometri yardimcilari (dirsek acisi,
                      gorunurluk kontrolu, EMA yumusatma, goz kirpintisi/head-pose)
  kayit.py           - kesit (JPEG) / video (MP4) kaydi (VideoKaydedici sinifi)
  gaze_birlesik.py (BU DOSYA) - hepsini birbirine baglayan ana webcam dongusu

TEK bir cv2.VideoCapture ve TEK bir MediaPipe FaceLandmarker cagrisiyla hem
bakis (OpenVINO gaze-estimation-adas-0002) hem kirpma (blendshape) hem
head-pose (transformation matrix) bilgisi cikarilir; ayrica MediaPipe
PoseLandmarker (govde) ve HandLandmarker (eller) calisir, hepsinin sonucu
AYNI karenin uzerine cizilir.

L2CS-Net surumunden farki: orada yuz tespiti IKI KEZ calisiyordu (L2CS'in
kendi RetinaFace'i + kirpma icin ayri FaceLandmarker). Burada TEK
FaceLandmarker cagrisi hem kirpma hem bakis (landmark + head-pose) icin
yeterli - hem daha hafif hem lisans acisindan temiz (Gaze360 yerine Apache
2.0 lisansli OpenVINO modeli).

ONCE model dosyalari gerekli:
  gaze-estimation-adas-0002.xml + .bin - ELLE indirilip BU klasore konmali
  (bkz. ayarlar.py docstring'i).
MediaPipe model dosyalari (face_landmarker.task, pose_landmarker_lite.task,
hand_landmarker.task) ilk calistirmada otomatik indirilir.

PERFORMANS UYARISI: Tek karede UC model (FaceLandmarker+OpenVINO gaze,
PoseLandmarker, HandLandmarker) calisiyor. Yavas gelirse ayarlar.py'deki
AKTIF_POSE / AKTIF_EL bayraklarini False yap.

SOL KOL / SAG KOL sayaclari - IKI AYRI tetikleyiciden HERHANGI BIRI olusunca
artar (transitleri, yani "az once yoktu simdi var" anini yakalar):
  1) Kol kalkik: bilek, omuz hizasinin USTUNE cikinca.
  2) Dirsek kivrik + el omuz hizasinda: bilek ile omuz YAKLASIK ayni
     yukseklikteyken dirsek acisi kucukse ("biceps curl" hareketi).
AKTIF_POSE=False ise bu sayaclar calismaz (PoseLandmarker gerekli).

OLAY KESITI: SOL KOL / SAG KOL / KIRPMA / BAKIS (SOL-SAG-YUKARI-ASAGI) sayaclarindan
biri her artinca (ilgili olayin ANI), o andan ONCEKI ayarlar.OLAY_ONCE_SANIYE
(varsayilan 2sn) VE SONRAKI ayarlar.OLAY_SONRA_SANIYE (varsayilan 2sn) -
TOPLAM ~4 saniyelik bir MP4 otomatik olarak kaydedilir (kayit.
OlayKlibiYoneticisi). Her kategorinin KENDI klasoru ve KENDI (bagimsiz)
yoneticisi var, "videolar/" altinda:
  sol_kol/     - SOL KOL sayaci artinca
  sag_kol/     - SAG KOL sayaci artinca
  goz_kirpma/  - KIRPMA sayaci artinca
  goz_bakisi/  - SOL/SAG/YUKARI/ASAGI bakis yonu sayaclarindan biri artinca
Bu, 'v' tusuyla acilan MANUEL kayittan BAGIMSIZ, her zaman arka planda calisir.

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
import math
import time
import types

import cv2
import mediapipe as mp
import numpy as np

import ayarlar as A
import gorsellik as G
import kayit as K
import modeller as M

# --- Modelleri yukle ----------------------------------------------------
if A.AKTIF_GAZE:
    gaze = M.gaze_pipeline_yukle()
else:
    gaze = None
    print("[bilgi] AKTIF_GAZE=False - OpenVINO gaze modeli YUKLENMEDI. Bakis yonu (SOL/SAG/YUKARI/ASAGI) sayaclari ve goz oku calismayacak.")

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
    "sol_parmak": 0, "sag_parmak": 0,
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

# Parmak hareketi (bkz. ayarlar.PARMAK_* aciklamasi) - HER PARMAK UCU
# (bkz. asagidaki dosya-basi aciklamasi) KENDI BAGIMSIZ durumuyla ayri
# izlenir - "sadece bir parmak bile oynasa yakalansin" istegi icin, 5 ucun
# ORTALAMASINI almak (tek parmagin hareketini sulandirir) yerine. KOL/BACAK
# taki hareket_algila DEGIL, gorsellik.parmak_hareket_algila kullanilir -
# "asagi inip tekrar kalkmaya gerek olmadan HEMEN algila" istegi icin (bkz.
# ayarlar.py'deki PARMAK_HIZ_* aciklamasinin tam gerekcesi).
# parmak_durum["sol"][i] / ["sag"][i] (i=0..4, basparmak..serce) - her biri
# {"hz_x","hz_y","son_tetik"} sozlugu (parmak_hareket_algila'nin bir onceki
# karedeki durumu). sayaclar gibi bu da BILEREK sozluk - 2 el x 5 parmak
# icin 30 ayri isimli degisken yerine.
parmak_durum = {
    "sol": [dict(hz_x=None, hz_y=None, son_tetik=9999) for _ in range(5)],
    "sag": [dict(hz_x=None, hz_y=None, son_tetik=9999) for _ in range(5)],
}
# NOT: KOL/BACAK'takinin AKSINE, burada AYRICA bir "onceki_aktif" (edge-
# trigger) durumu YOK - parmak_hareket_algila'nin "tetiklendi" cikisi zaten
# TEK KARELIK bir "az once yeni bir hareket algilandi" darbesi (kendi ic
# refractory sayaci sayesinde), sureklilik/seviye DEGIL - bu yuzden
# dogrudan "tetiklendiyse sayaci artir" yeterli, ayrica bir gecis kontrolu
# gerekmiyor.

# Kimlik kilidi durumu (bkz. ayarlar.py, gorsellik.kilitli_aday_sec) - yuz
# ve govde icin AYRI kilitler (MediaPipe'in iki dedektoru birbirinden
# BAGIMSIZ calisir, ortak bir "kisi ID"si yok).
kilitli_yuz_merkez = None
yuz_kayip_kare = 0
kilitli_govde_merkez = None
govde_kayip_kare = 0

# govde_olcek'in YUMUSATILMIS (EMA) hali - bkz. ayarlar.GOVDE_OLCEK_*
# aciklamasi: KOL+BACAK'IN DORDU DE bu tek degeri PAYDA olarak kullaniyor,
# gurultulu/ham bir omuz karesi bunu ANLIK kucultup DORT sayaci BIRDEN
# yanlis tetikleyebiliyordu - simdi yumusatiliyor. Govde/kimlik kilidi
# KAYBEDILINCE None'a resetlenir (asagida) ki eski/bayat bir deger yeni
# kilitlenen kisiye SIZMASIN.
govde_olcek_yumusak = None

# TAKIP EDEN dijital yakinlastirma icin durum - kilitli yuzun SON bilinen
# konumu, HAM (kirpilmamis) kamera karesinin koordinatlarinda (bkz.
# gorsellik.takip_yakinlastir / raw_konuma_cevir, ayarlar.
# DIJITAL_YAKINLASTIRMA). None = henuz kilit yok / kilit tamamen birakildi
# -> o kare TAM ORTADAN (genis/arama modunda) kirpilir.
takip_merkezi = None

# Yuz henuz bulunmadan once (ilk kareler) cizilmesin diye baslangic degerleri.
cizgi_sol_x = cizgi_sag_x = cizgi_ust_y = cizgi_alt_y = None

BIAS_GX = 0.0
BIAS_GY = 0.0
son_gx_ham = 0.0
son_gy_ham = 0.0
yuz_bulundu_bu_kare = False

yumusak_gx = None
yumusak_gy = None
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
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.KOL_SOL_KLASORU, dosya_on_eki="sol_kol",
)
sag_kol_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.KOL_SAG_KLASORU, dosya_on_eki="sag_kol",
)
sol_bacak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.BACAK_SOL_KLASORU, dosya_on_eki="sol_bacak",
)
sag_bacak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.BACAK_SAG_KLASORU, dosya_on_eki="sag_bacak",
)
kirpma_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_KIRPMA_KLASORU, dosya_on_eki="kirpma",
)
bakis_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_BAKISI_KLASORU, dosya_on_eki="bakis",
)
sol_parmak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.PARMAK_SOL_KLASORU, dosya_on_eki="sol_parmak",
)
sag_parmak_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.PARMAK_SAG_KLASORU, dosya_on_eki="sag_parmak",
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

    # --- KOL/BACAK TANI (DEBUG) OKUMALARI: her karede sifirlanir, "?" =
    # pose/govde bu karede HIC bulunamadi (ya da asagidaki try icinde bir
    # istisna sessizce yutuldu - bkz. except Exception: pass), "gizli" =
    # govde bulundu AMA ilgili uzuv/referans noktasinin gorunurlugu
    # ayarlar.GORUNURLUK_ESIK altinda (hareket_algila HIC CAGRILMADI), sayi =
    # HIZLI/YAVAS EMA arasindaki gercek mesafe (esikle KARSILASTIR - sayac
    # bunu esik/histerezis oranini GECINCE artar). "sayaç hiç değişmiyor"
    # sikayetini kok nedenine indirmek icin eklendi - bkz. ekrandaki KOL/BACAK
    # TANI satiri.
    _dbg_sol_kol = "?"
    _dbg_sag_kol = "?"
    _dbg_sol_bacak = "?"
    _dbg_sag_bacak = "?"
    _dbg_sol_parmak = "?"
    _dbg_sag_parmak = "?"

    # Bu karede POSE'dan (ekrana-gore DUZELTILMIS) sol_bilek/sag_bilek TAZE
    # ve GORUNUR mu? - el/parmak bloğunun MediaPipe'in kendi (guvenilmez)
    # Sol/Sag el etiketi yerine bu bilekleri kullanip elleri dogru govde
    # tarafina eslestirebilmesi icin. False kalirsa (govde/pose yoksa ya da
    # bilek gorunmuyorsa) o taraf icin el-govde eslestirmesi ATLANIR (bkz.
    # asagidaki AKTIF_EL blogu) - yanlis sol/sag atamasi yapmaktansa o
    # kareyi atlamak tercih edildi.
    _sol_bilek_gecerli = False
    _sag_bilek_gecerli = False

    # --- MediaPipe: TEK cagriyla kirpma + bakis(landmark/head-pose) + govde + eller
    try:
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        kare_zaman_damgasi_ms += 33

        landmarker_sonuc = face_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

        # --- Kimlik kilidi: adaylar arasindan "kilitli kisiyi" sec ----------
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

        # --- OpenVINO: bakis yonu -------------------------------------------
        if A.AKTIF_GAZE and gaze is not None and secilen_yuz_i is not None:
            landmarks = landmarker_sonuc.face_landmarks[secilen_yuz_i]

            if landmarker_sonuc.facial_transformation_matrixes:
                rot = np.array(landmarker_sonuc.facial_transformation_matrixes[secilen_yuz_i])[:3, :3]
                # NOT: donus_matrisinden_aci'nin dondurdugu (yaw, pitch) etiketleri
                # MediaPipe'in eksen sirasiyla TERS cikiyor - olcumle dogrulandi
                # (sag/sola donunce asil buyuk degisim "pitch" ciktisinda oluyordu).
                # Bu yuzden burada BILEREK ters atiyoruz.
                pitch, yaw, roll = G.donus_matrisinden_aci(rot)
            else:
                yaw, pitch, roll = 0.0, 0.0, 0.0

            sx1, sy1, sx2, sy2 = G.goz_kutusu(landmarks, G.SAG_GOZ_IDX, w, h)
            lx1, ly1, lx2, ly2 = G.goz_kutusu(landmarks, G.SOL_GOZ_IDX, w, h)
            sag_goz = kare[sy1:sy2, sx1:sx2]
            sol_goz = kare[ly1:ly2, lx1:lx2]

            if sag_goz.size > 0 and sol_goz.size > 0:
                yuz_bulundu_bu_kare = True

                # Roll telafisi: gozleri yataya hizala, modele roll=0 ver.
                sag_goz = G.kirpinti_dondur(sag_goz, roll)
                sol_goz = G.kirpinti_dondur(sol_goz, roll)

                gaze.infer({
                    "left_eye_image": G.kirpinti_hazirla(sol_goz),
                    "right_eye_image": G.kirpinti_hazirla(sag_goz),
                    "head_pose_angles": np.array([[yaw, pitch, 0.0]], dtype=np.float32),
                })
                vektor = gaze.get_output_tensor().data[0].copy()
                vektor = vektor / (np.linalg.norm(vektor) + 1e-9)

                # Roll telafisini geri al.
                rad = np.radians(roll)
                cs, sn = np.cos(rad), np.sin(rad)
                son_gx_ham = float(vektor[0] * cs + vektor[1] * sn)
                son_gy_ham = float(-vektor[0] * sn + vektor[1] * cs)

                gx = son_gx_ham - BIAS_GX
                gy = son_gy_ham - BIAS_GY

                # CAPRAZ EKSEN SIZINTISI DUZELTMESI (bkz. ayarlar.py) - iki
                # AYRI, birbirinden BAGIMSIZ duzeltme; duzeltilmemis ham
                # gx/gy degerleri UZERINDEN hesaplaniyor ki biri digerini
                # etkilemesin (ikisi de ayni ham olcumden turetildi).
                _ham_gx, _ham_gy = gx, gy
                if _ham_gy < 0:  # sadece asagi bakista gx'e sizan pay
                    gx -= A.BAKIS_ASAGI_SIZINTI_K * (-_ham_gy)
                gy += A.BAKIS_YANAL_SIZINTI_K * abs(_ham_gx)  # her zaman yanlara sizan pay

                x_min, y_min, x_max, y_max = G.yuz_bbox_hesapla(landmarks, w, h)
                if A.YUZ_CIZIMI_GOSTER:
                    cv2.rectangle(kare, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)

                merkez_x = (x_min + x_max) / 2.0
                merkez_y = (y_min + y_max) / 2.0
                uzunluk = x_max - x_min

                # Yumusatma: yuz kucukken (uzaktayken) hem landmark hem gaze
                # vektoru tahmini daha gurultulu oluyor - HAM degerler yerine
                # yumusatilmis (EMA) degerleri kullaniyoruz ki ok/kutu
                # titremesin. gx/gy icin ayrica MAKS_BAKIS_SICRAMA ile tek
                # karelik "cilginca" outlier sicramalar da kirpiliyor.
                gx = yumusak_gx = G.yumusat(yumusak_gx, gx, A.MAKS_BAKIS_SICRAMA)
                gy = yumusak_gy = G.yumusat(yumusak_gy, gy, A.MAKS_BAKIS_SICRAMA)
                merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
                merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
                uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

                # Kenar "kutusu" YUZE GORE - yuz genisligiyle olceklenir ve
                # merkezi her kare yuzun merkezine tasinir.
                cizgi_sol_x = int(merkez_x - uzunluk * A.KENAR_MESAFE_YATAY)
                cizgi_sag_x = int(merkez_x + uzunluk * A.KENAR_MESAFE_YATAY)
                cizgi_ust_y = int(merkez_y - uzunluk * A.KENAR_MESAFE_UST)
                cizgi_alt_y = int(merkez_y + uzunluk * A.KENAR_MESAFE_ALT)

                dx = uzunluk * gx
                dy = -uzunluk * gy  # ekranda y asagi buyudugu icin ters cevir
                ucur_x = merkez_x + dx
                ucur_y = merkez_y + dy

                duz_bakiyor = abs(gx) < A.ESIK_BAKIS_XY and abs(gy) < A.ESIK_BAKIS_XY

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

                # DEBUG: yaw/pitch/roll ve gx/gy degerlerini ekrana yazdir -
                # sag/sola donunce sayilarin mantikli davranip davranmadigini
                # (buyuklukce artmasi, isaretin tutarli olmasi) gormek icin.
                cv2.putText(
                    kare, f"yaw:{yaw:6.1f} pitch:{pitch:6.1f} roll:{roll:6.1f}  gx:{gx:5.2f} gy:{gy:5.2f}",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )

        if A.AKTIF_POSE and pose_landmarker is not None:
            pose_sonuc = pose_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

            # --- Kimlik kilidi: adaylar arasindan "kilitli kisiyi" sec -------
            # (yuzden BAGIMSIZ - PoseLandmarker kendi adaylarini kendi
            # sirasiyla dondurur, ayni index yuzdeki ayni kisiyi garanti
            # etmez, bu yuzden omuz orta noktasi uzerinden AYRICA takip
            # edilir.)
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

            if secilen_govde_i is None:
                # Govde/kimlik kilidi bu karede YOK - govde_olcek yumusatmasini
                # sifirla ki (bir sonraki kilitlenmede) BAYAT/eski bir olcek
                # degeri yeni durumu carpitmasin (bkz. govde_olcek_yumusak).
                govde_olcek_yumusak = None

            # --- Kol sayaci: IKI tetikleyiciden HERHANGI BIRI olusunca (asagidan-
            # yukari GECIS anini yakalayip) sayaci artirir - bkz. dosya basindaki
            # docstring. Normalize y kuculdukce ekranda yukari demektir.
            if secilen_govde_i is not None:
                lm = pose_sonuc.pose_landmarks[secilen_govde_i]

                # --- GOVDE OLCEGI: kol/bacak hareketi artik MUTLAK (kare-ici)
                # konum yerine GOVDEYE GORELI konumla izleniyor (bkz.
                # gorsellik.govdeye_goreli_konum ve ayarlar.py'deki
                # KOL_HAREKET_GORELI_*/BACAK_HAREKET_GORELI_* aciklamasi) -
                # "kolu oynatinca bacak sayaci da artiyor, bacagi oynatinca
                # kol sayaci da artiyor" seklindeki CAPRAZ yanlis tetiklenme
                # sorununu (govde/yatak/kamera geneli kaymalar) kokten
                # cozmek icin. Omuzlar da EKRANA_GORE_SOL_SAG ile duzeltiliyor
                # (bilek/ayak bilegi ile AYNI tarafa eslenmesi icin).
                sol_omuz = lm[PoseLandmark.LEFT_SHOULDER]
                sag_omuz = lm[PoseLandmark.RIGHT_SHOULDER]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_omuz, sag_omuz = G.ekran_sol_sag_ayikla(sol_omuz, sag_omuz)
                # YUMUSATILMIS govde_olcek kullan (bkz. ayarlar.GOVDE_OLCEK_*
                # ve yukaridaki govde_olcek_yumusak aciklamasi) - HAM degeri
                # DOGRUDAN kullanmak, tek gurultulu karede KOL+BACAK'IN
                # DORDUNU BIRDEN yanlis tetikleyebiliyordu.
                _govde_olcek_ham = G.govde_olcek_hesapla(sol_omuz, sag_omuz)
                # UZUN SURELI (tek kareden fazla) bozulmalara karsi EK filtre
                # (bkz. ayarlar.GOVDE_OLCEK_KABUL_* aciklamasi): ham okuma su
                # anki yumusatilmis degerden COK sapiyorsa (orn. kol kameraya
                # dogru/tepeye kalkinca omuz tahmini oynayabiliyor), bu kareyi
                # TAMAMEN yoksay - MAKS_SICRAMA'nin adim adim (kare kare
                # birikerek) yine de yanlis degere surunmesini engeller.
                if (
                    govde_olcek_yumusak is None
                    or A.GOVDE_OLCEK_KABUL_MIN_ORAN * govde_olcek_yumusak
                    <= _govde_olcek_ham
                    <= A.GOVDE_OLCEK_KABUL_MAKS_ORAN * govde_olcek_yumusak
                ):
                    govde_olcek_yumusak = G.yumusat(
                        govde_olcek_yumusak, _govde_olcek_ham,
                        A.GOVDE_OLCEK_MAKS_SICRAMA, A.GOVDE_OLCEK_YUMUSATMA_ORANI,
                    )
                govde_olcek = govde_olcek_yumusak

                sol_dirsek = lm[PoseLandmark.LEFT_ELBOW]
                sol_bilek = lm[PoseLandmark.LEFT_WRIST]
                sag_bilek_ham = lm[PoseLandmark.RIGHT_WRIST]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_bilek, sag_bilek_ham = G.ekran_sol_sag_ayikla(sol_bilek, sag_bilek_ham)
                if A.EKRANA_GORE_ETIKET_GOSTER:
                    # SAYACIN kullandigi (ekrana gore duzeltilmis) bilegin
                    # TAM USTUNE - govde_ciz'in HAM/anatomik renkleriyle
                    # KARISMASIN diye (bkz. gorsellik.ekran_etiket_ciz).
                    G.ekran_etiket_ciz(kare, sol_bilek, "SOL", (255, 0, 255), w, h)
                    G.ekran_etiket_ciz(kare, sag_bilek_ham, "SAG", (0, 255, 255), w, h)
                if G.gorunur_mu(sol_bilek) and G.gorunur_mu(sol_omuz):
                    _sol_bilek_gecerli = True
                    _sol_kol_gx, _sol_kol_gy = G.govdeye_goreli_konum(sol_bilek, sol_omuz, govde_olcek)
                    (sol_kol_aktif, sol_kol_hizli_x, sol_kol_hizli_y,
                     sol_kol_yavas_x, sol_kol_yavas_y, sol_kol_cikis_sayaci) = G.hareket_algila(
                        sol_kol_hizli_x, sol_kol_hizli_y, sol_kol_yavas_x, sol_kol_yavas_y,
                        _sol_kol_gx, _sol_kol_gy,
                        onceki_sol_kol_aktif, A.KOL_HAREKET_GORELI_ESIK,
                        A.KOL_HAREKET_GORELI_HISTEREZIS_ORANI, A.KOL_HAREKET_GORELI_HIZLI_ORAN, A.KOL_HAREKET_GORELI_YAVAS_ORAN,
                        sol_kol_cikis_sayaci, A.KOL_HAREKET_GORELI_MIN_CIKIS_KARE,
                    )
                    _dbg_sol_kol = math.hypot(sol_kol_hizli_x - sol_kol_yavas_x, sol_kol_hizli_y - sol_kol_yavas_y)
                    if sol_kol_aktif and not onceki_sol_kol_aktif:
                        if sayaclar_aktif:
                            sayaclar["sol_kol"] += 1
                            if A.KOL_OLAY_KESITI_AKTIF:
                                sol_kol_olay_kaydedici.olay_tetikle("sol_kol")
                    onceki_sol_kol_aktif = sol_kol_aktif
                else:
                    _dbg_sol_kol = "gizli"

                sag_dirsek = lm[PoseLandmark.RIGHT_ELBOW]
                sag_bilek = sag_bilek_ham
                if G.gorunur_mu(sag_bilek) and G.gorunur_mu(sag_omuz):
                    _sag_bilek_gecerli = True
                    _sag_kol_gx, _sag_kol_gy = G.govdeye_goreli_konum(sag_bilek, sag_omuz, govde_olcek)
                    (sag_kol_aktif, sag_kol_hizli_x, sag_kol_hizli_y,
                     sag_kol_yavas_x, sag_kol_yavas_y, sag_kol_cikis_sayaci) = G.hareket_algila(
                        sag_kol_hizli_x, sag_kol_hizli_y, sag_kol_yavas_x, sag_kol_yavas_y,
                        _sag_kol_gx, _sag_kol_gy,
                        onceki_sag_kol_aktif, A.KOL_HAREKET_GORELI_ESIK,
                        A.KOL_HAREKET_GORELI_HISTEREZIS_ORANI, A.KOL_HAREKET_GORELI_HIZLI_ORAN, A.KOL_HAREKET_GORELI_YAVAS_ORAN,
                        sag_kol_cikis_sayaci, A.KOL_HAREKET_GORELI_MIN_CIKIS_KARE,
                    )
                    _dbg_sag_kol = math.hypot(sag_kol_hizli_x - sag_kol_yavas_x, sag_kol_hizli_y - sag_kol_yavas_y)
                    if sag_kol_aktif and not onceki_sag_kol_aktif:
                        if sayaclar_aktif:
                            sayaclar["sag_kol"] += 1
                            if A.KOL_OLAY_KESITI_AKTIF:
                                sag_kol_olay_kaydedici.olay_tetikle("sag_kol")
                    onceki_sag_kol_aktif = sag_kol_aktif
                else:
                    _dbg_sag_kol = "gizli"

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

                # --- Bacak/ayak hareketi: ARTIK govdeye (kalcaya) GORELI
                # konum kullanir, MUTLAK KONUM DEGIL - bkz. gorsellik.
                # govdeye_goreli_konum ve ayarlar.py'deki BACAK_HAREKET_
                # GORELI_* aciklamasi. KOL'daki gibi yon sarti YOK (yatan
                # hastada da calisir) - AMA ARTIK govde-geneli kaymalara ve
                # KOL ile capraz tetiklenmeye karsi da bagisik.
                sol_kalca = lm[PoseLandmark.LEFT_HIP]
                sag_kalca = lm[PoseLandmark.RIGHT_HIP]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_kalca, sag_kalca = G.ekran_sol_sag_ayikla(sol_kalca, sag_kalca)

                sol_ayak_bilegi = lm[PoseLandmark.LEFT_ANKLE]
                sag_ayak_bilegi_ham = lm[PoseLandmark.RIGHT_ANKLE]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_ayak_bilegi, sag_ayak_bilegi_ham = G.ekran_sol_sag_ayikla(sol_ayak_bilegi, sag_ayak_bilegi_ham)
                if A.EKRANA_GORE_ETIKET_GOSTER:
                    G.ekran_etiket_ciz(kare, sol_ayak_bilegi, "SOL", (255, 0, 255), w, h)
                    G.ekran_etiket_ciz(kare, sag_ayak_bilegi_ham, "SAG", (0, 255, 255), w, h)
                if G.gorunur_mu(sol_ayak_bilegi) and G.gorunur_mu(sol_kalca):
                    _sol_bacak_gx, _sol_bacak_gy = G.govdeye_goreli_konum(sol_ayak_bilegi, sol_kalca, govde_olcek)
                    (sol_bacak_hareketli, sol_bacak_hizli_x, sol_bacak_hizli_y,
                     sol_bacak_yavas_x, sol_bacak_yavas_y, sol_bacak_cikis_sayaci) = G.hareket_algila(
                        sol_bacak_hizli_x, sol_bacak_hizli_y, sol_bacak_yavas_x, sol_bacak_yavas_y,
                        _sol_bacak_gx, _sol_bacak_gy,
                        onceki_sol_bacak_hareketli, A.BACAK_HAREKET_GORELI_ESIK,
                        A.BACAK_HAREKET_GORELI_HISTEREZIS_ORANI, A.BACAK_HAREKET_GORELI_HIZLI_ORAN, A.BACAK_HAREKET_GORELI_YAVAS_ORAN,
                        sol_bacak_cikis_sayaci, A.BACAK_HAREKET_GORELI_MIN_CIKIS_KARE,
                    )
                    _dbg_sol_bacak = math.hypot(sol_bacak_hizli_x - sol_bacak_yavas_x, sol_bacak_hizli_y - sol_bacak_yavas_y)
                    if sol_bacak_hareketli and not onceki_sol_bacak_hareketli:
                        if sayaclar_aktif:
                            sayaclar["sol_bacak"] += 1
                            if A.BACAK_OLAY_KESITI_AKTIF:
                                sol_bacak_olay_kaydedici.olay_tetikle("sol_bacak")
                    onceki_sol_bacak_hareketli = sol_bacak_hareketli
                else:
                    _dbg_sol_bacak = "gizli"

                sag_ayak_bilegi = sag_ayak_bilegi_ham
                if G.gorunur_mu(sag_ayak_bilegi) and G.gorunur_mu(sag_kalca):
                    _sag_bacak_gx, _sag_bacak_gy = G.govdeye_goreli_konum(sag_ayak_bilegi, sag_kalca, govde_olcek)
                    (sag_bacak_hareketli, sag_bacak_hizli_x, sag_bacak_hizli_y,
                     sag_bacak_yavas_x, sag_bacak_yavas_y, sag_bacak_cikis_sayaci) = G.hareket_algila(
                        sag_bacak_hizli_x, sag_bacak_hizli_y, sag_bacak_yavas_x, sag_bacak_yavas_y,
                        _sag_bacak_gx, _sag_bacak_gy,
                        onceki_sag_bacak_hareketli, A.BACAK_HAREKET_GORELI_ESIK,
                        A.BACAK_HAREKET_GORELI_HISTEREZIS_ORANI, A.BACAK_HAREKET_GORELI_HIZLI_ORAN, A.BACAK_HAREKET_GORELI_YAVAS_ORAN,
                        sag_bacak_cikis_sayaci, A.BACAK_HAREKET_GORELI_MIN_CIKIS_KARE,
                    )
                    _dbg_sag_bacak = math.hypot(sag_bacak_hizli_x - sag_bacak_yavas_x, sag_bacak_hizli_y - sag_bacak_yavas_y)
                    if sag_bacak_hareketli and not onceki_sag_bacak_hareketli:
                        if sayaclar_aktif:
                            sayaclar["sag_bacak"] += 1
                            if A.BACAK_OLAY_KESITI_AKTIF:
                                sag_bacak_olay_kaydedici.olay_tetikle("sag_bacak")
                    onceki_sag_bacak_hareketli = sag_bacak_hareketli
                else:
                    _dbg_sag_bacak = "gizli"

        if A.AKTIF_EL and hand_landmarker is not None:
            hand_sonuc = hand_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
            if A.EL_CIZIMI_GOSTER:
                G.eller_ciz(kare, hand_sonuc)

            # --- Parmak hareketi (bkz. ayarlar.PARMAK_* aciklamasinin tam
            # gerekcesi). MediaPipe HandLandmarker'in
            # KENDI Sol/Sag (handedness) etiketine GUVENILMIYOR - pose icin
            # bulunan AYNI kamera-acisi sorunu (bkz. ekran_sol_sag_ayikla)
            # burada da olasi. Bunun yerine: bu karede POSE'dan TAZE/gorunur
            # sol_bilek/sag_bilek varsa, tespit edilen her elin bilegini
            # (HandLandmark 0) EN YAKIN govde bilegine esleriz (govde =
            # "yer imi", el kimligini ONDAN alir). Govde bilegi YOKSA ama
            # IKI el tespit edildiyse ekrandaki x konumuna gore ataniriz
            # (kucuk x = sol - projedeki ortak kural). Govde YOK ve TEK el
            # varsa, o el ATLANIR - yanlis sol/sag atamasi yapmaktansa.
            _eller = hand_sonuc.hand_landmarks
            if _eller:
                _el_atama = [None] * len(_eller)  # 'sol' / 'sag' / None (belirsiz -> atla)
                _govde_bilekleri = []
                if _sol_bilek_gecerli:
                    _govde_bilekleri.append(("sol", sol_bilek))
                if _sag_bilek_gecerli:
                    _govde_bilekleri.append(("sag", sag_bilek))

                if _govde_bilekleri:
                    # Acgozlu (greedy) en-yakin eslestirme - en fazla 2 el VE
                    # 2 govde bilegi oldugundan bu yeterli/basit kaliyor.
                    _kalan_el_idx = list(range(len(_eller)))
                    for _taraf, _gbilek in _govde_bilekleri:
                        _en_yakin_i = None
                        _en_yakin_mesafe = None
                        for _i in _kalan_el_idx:
                            _el_bilegi = _eller[_i][0]  # HandLandmark 0 = WRIST
                            _mesafe = math.hypot(_el_bilegi.x - _gbilek.x, _el_bilegi.y - _gbilek.y)
                            if _mesafe <= A.EL_BILEK_ESLESTIRME_MAKS_MESAFE and (
                                _en_yakin_mesafe is None or _mesafe < _en_yakin_mesafe
                            ):
                                _en_yakin_mesafe = _mesafe
                                _en_yakin_i = _i
                        if _en_yakin_i is not None:
                            _el_atama[_en_yakin_i] = _taraf
                            _kalan_el_idx.remove(_en_yakin_i)
                elif len(_eller) == 2:
                    _sirali = sorted(range(len(_eller)), key=lambda i: _eller[i][0].x)
                    _el_atama[_sirali[0]] = "sol"
                    _el_atama[_sirali[1]] = "sag"
                # TEK el + govde bilegi YOK -> _el_atama[i] None kalir, asagida atlanir.

                for _i, _taraf in enumerate(_el_atama):
                    if _taraf is None:
                        continue
                    _el = _eller[_i]
                    _el_bilek = _el[0]        # WRIST
                    _el_orta_kok = _el[9]     # MIDDLE_FINGER_MCP - el buyuklugu referansi
                    if A.EKRANA_GORE_ETIKET_GOSTER:
                        # sol_bilek/sag_bilek icin kullanilan AYNI etiket
                        # mantigi (bkz. ekran_etiket_ciz) - SAYACIN hangi eli
                        # hangi tarafa saydigini gorsel olarak dogrula.
                        _el_renk = (255, 0, 255) if _taraf == "sol" else (0, 255, 255)
                        G.ekran_etiket_ciz(kare, _el_bilek, _taraf.upper(), _el_renk, w, h)
                    _el_olcegi = G.govde_olcek_hesapla(_el_bilek, _el_orta_kok, A.EL_OLCEK_MIN)

                    # HER PARMAK UCU (basparmak, isaret, orta, yuzuk, serce)
                    # KENDI BAGIMSIZ parmak_hareket_algila durumuyla AYRI
                    # izlenir - ORTALAMA ALINMAZ (bkz. yukaridaki aciklama).
                    # El, HERHANGI BIR ucun KENDI esigini gecmesiyle
                    # "hareket algilandi" sayilir (OR) - "asagi inip tekrar
                    # kalkma" beklenmez, sadece KISA bir yeniden-tetiklenme
                    # bekleme suresi vardir (bkz. ayarlar.PARMAK_*).
                    _durumlar = parmak_durum[_taraf]
                    _herhangi_biri_aktif = False
                    _en_buyuk_hiz = 0.0
                    for _pi, _uc_idx in enumerate((4, 8, 12, 16, 20)):
                        _d = _durumlar[_pi]
                        _gx, _gy = G.govdeye_goreli_konum(_el[_uc_idx], _el_bilek, _el_olcegi)
                        (_parmak_tetiklendi, _d["hz_x"], _d["hz_y"], _d["son_tetik"], _hiz) = G.parmak_hareket_algila(
                            _d["hz_x"], _d["hz_y"], _gx, _gy,
                            A.PARMAK_HIZ_ESIK, A.PARMAK_HIZ_HIZLI_ORAN,
                            _d["son_tetik"], A.PARMAK_YENIDEN_TETIK_MIN_KARE,
                        )
                        if _parmak_tetiklendi:
                            _herhangi_biri_aktif = True
                        if _hiz > _en_buyuk_hiz:
                            _en_buyuk_hiz = _hiz

                    if _taraf == "sol":
                        _dbg_sol_parmak = _en_buyuk_hiz
                        if _herhangi_biri_aktif:
                            if sayaclar_aktif:
                                sayaclar["sol_parmak"] += 1
                                if A.PARMAK_OLAY_KESITI_AKTIF:
                                    sol_parmak_olay_kaydedici.olay_tetikle("sol_parmak")
                    else:
                        _dbg_sag_parmak = _en_buyuk_hiz
                        if _herhangi_biri_aktif:
                            if sayaclar_aktif:
                                sayaclar["sag_parmak"] += 1
                                if A.PARMAK_OLAY_KESITI_AKTIF:
                                    sag_parmak_olay_kaydedici.olay_tetikle("sag_parmak")
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
    cv2.putText(kare, f"SOL PARMAK: {sayaclar['sol_parmak']}   SAG PARMAK: {sayaclar['sag_parmak']}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(
        kare, f"SAYAÇLAR: {'AKTIF' if sayaclar_aktif else 'DURAKLATILDI'} (h)",
        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0) if sayaclar_aktif else (0, 0, 255), 2,
    )
    cv2.putText(
        kare, f"YAKINLAŞTIRMA: {'AKTIF' if yakinlastirma_aktif else 'KAPALI'} (z)",
        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0) if yakinlastirma_aktif else (0, 0, 255), 2,
    )
    if gcs_test_aktif:
        _gcs_kalan = max(0.0, A.GCS_PENCERE_SANIYE - (time.time() - gcs_test_baslangic_zaman))
        cv2.putText(kare, f"GCS TESTI: OLCULUYOR ({_gcs_kalan:.1f}sn) - uyaran uygula",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    elif gcs_son_sonuc is not None and time.time() < gcs_sonuc_gosterim_bitis:
        _gcs_sol_e, _gcs_sag_e = gcs_son_sonuc
        cv2.putText(kare, f"GCS SONUC (sezgisel): SOL={_gcs_sol_e or '?'}  SAG={_gcs_sag_e or '?'}",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(kare, "GCS TESTI: HAZIR (g)", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # --- KOL/BACAK/PARMAK TANI satiri: "sayaç hiç değişmiyor" gibi
    # sikayetleri KANITLA (tahmin etmeden) teshis etmek icin - her uzuv icin
    # HIZLI/YAVAS EMA mesafesini ve esigini yan yana gosterir. "?" = govde/el
    # bu karede HIC bulunamadi (ya da try/except icinde sessiz bir istisna
    # oldu, ya da PARMAK icin: govde bilegi yoktu/el sol-sag'a eslenemedi),
    # "gizli" = govde bulundu ama uzuv/referans noktasi GORUNURLUK_ESIK
    # altinda (sayim mantigina hic girmedi), sayi = gercek mesafe (esigi
    # GECERSE sayac artar - esige ne kadar YAKLASIP GECMEDIGINI gorebilirsin).
    def _dbg_fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else v
    cv2.putText(
        kare,
        f"TANI KOL sol:{_dbg_fmt(_dbg_sol_kol)} sag:{_dbg_fmt(_dbg_sag_kol)} (esik {A.KOL_HAREKET_GORELI_ESIK:.2f})  "
        f"BACAK sol:{_dbg_fmt(_dbg_sol_bacak)} sag:{_dbg_fmt(_dbg_sag_bacak)} (esik {A.BACAK_HAREKET_GORELI_ESIK:.2f})",
        (20, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
    )
    cv2.putText(
        kare,
        f"TANI PARMAK sol:{_dbg_fmt(_dbg_sol_parmak)} sag:{_dbg_fmt(_dbg_sag_parmak)} (esik {A.PARMAK_HIZ_ESIK:.2f})",
        (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
    )
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
    if A.PARMAK_OLAY_KESITI_AKTIF:
        sol_parmak_olay_kaydedici.kare_ekle(kare)
        sag_parmak_olay_kaydedici.kare_ekle(kare)
    if A.KIRPMA_OLAY_KESITI_AKTIF:
        kirpma_olay_kaydedici.kare_ekle(kare)
    if A.BAKIS_OLAY_KESITI_AKTIF:
        bakis_olay_kaydedici.kare_ekle(kare)

    if ilk_kare_mi:
        print(f"[zaman] ilk kare islendi (modellerin ilk 'isinmasi' dahil): {time.time() - _t6:.1f}s")
        ilk_kare_mi = False

    cv2.imshow("MediaPipe + OpenVINO: bakis + kirpma + govde + eller (q = cik)", kare)
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        if kaydedici.kayit_yapiliyor:
            kaydedici.bitir()
        sol_kol_olay_kaydedici.bitir()
        sag_kol_olay_kaydedici.bitir()
        sol_bacak_olay_kaydedici.bitir()
        sag_bacak_olay_kaydedici.bitir()
        sol_parmak_olay_kaydedici.bitir()
        sag_parmak_olay_kaydedici.bitir()
        kirpma_olay_kaydedici.bitir()
        bakis_olay_kaydedici.bitir()
        break
    if tus == ord("c") and yuz_bulundu_bu_kare:
        BIAS_GX = son_gx_ham
        BIAS_GY = son_gy_ham
        print(f"Kalibre edildi. BIAS_GX={BIAS_GX:.3f} BIAS_GY={BIAS_GY:.3f}")
    if tus == ord("r"):
        K.kesit_al(kare, sayaclar)
    if tus == ord("v"):
        if not kaydedici.kayit_yapiliyor:
            kaydedici.baslat()
        else:
            kaydedici.bitir()
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