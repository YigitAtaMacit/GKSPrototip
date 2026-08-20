"""UZAK/SABIT KAMERA icin bolunmus-ekran SABIT BOLGE zoom surumu.

gaze_birlesik.py webcam'de (kisiye YAKIN, MediaPipe'in genis karede yuz/el
detaylarini rahat gordugu kurulumda) COK IYI calisiyor. Kamera hastadan
UZAKTA/sabit (orn. tavana/koseye monteli, oda genelini goren bir kamera)
oldugunda ise YUZ ve PARMAK gibi INCE detaylar kadrajda kucuk kalir,
MediaPipe bunlari genis karede guvenilir tespit edemez - govde/kol/bacak
gibi BUYUK hareketler ise genis acidan yine sorunsuz gorulur.

BU DOSYA bu ikinci senaryo icin: SEN elle (nokta_sec.py ile fare tiklayarak)
"yuz nerede", "sol el nerede", "sag el nerede" isaretlersin - bu noktalar
SABITTIR (kamera ve hasta konumu degismedigi surece takip etmeye GEREK
YOK). Uygulama HER karede bu SABIT noktalarin etrafini kirpip buyutur (zoom
yapar) ve YUZ/EL tespitini DOGRUDAN bu buyutulmus goruntude calistirir -
boylece uzak kamerada kaybolan kirpma/bakis/parmak hareketleri yakalanir.
Govde/kol/bacak ise (buyuk hareketler oldugu icin zoom'a ihtiyaci olmadan)
genis-aci PENCEREDE, gaze_birlesik.py ile AYNI mantikla calismaya devam eder.

ONCE CALISTIRMAN GEREKEN: nokta_sec.py (fare ile yuz/sol el/sag el
noktalarini isaretleyip kaydet). Hicbir bolge tanimlanmadiysa bu script
YINE DE calisir ama sadece genis-aci (kol/bacak) gorunumunu gosterir -
bolunmus ekran penceresi acilmaz.

Kontroller (gaze_birlesik.py ile BUYUK OLCUDE AYNI):
  c = kalibre et (yuz bolgesi TANIMLIYSA VE o bolgede yuz GORUNUYORSA -ya
      da ANA KAMERA modundaysa VE genis karede yuz GORUNUYORSA-, kameraya
      duz bakarken bas - bakis sapmasini sifirlar)
  r = kesit al (genis-aci + bolunmus-ekran kareleri, "kesitler/" klasorune JPEG)
  v = video kaydi ac/kapat (genis-aci VE bolgeler izgarasi BIRLIKTE - "videolar/"
      klasorune "video_..." ve "video_bolgeler_..." olarak IKI AYRI MP4)
  h = sayaclari ac/kapat (BASLANGICTA KAPALI)
  z = ANA KAMERA modu ac/kapat (BASLANGICTA KAPALI - yani sabit bolge/zoom
      modu varsayilan). ACIKKEN: nokta_sec.py ile isaretlenmis SABIT
      bolgeler/zoom TAMAMEN DEVRE DISI kalir, YUZ (kirpma/bakis) VE EL
      (parmak) sayaclari DOGRUDAN genis-aci/ana kameradan, gaze_birlesik.py
      ile AYNI yontemle (bilege GORE olcekli parmak takibi, govde
      bilekleriyle sol/sag el eslestirme) beslenir - nokta_sec.py hic
      calistirilmamis olsa BILE bu modda parmak/yuz sayaclari calisir.
      Istediginiz an tekrar 'z' ile SABIT BOLGE/zoom moduna donebilirsiniz.
  g = GCS motor tepki testi (M2-M5, SEZGISEL - bkz. gorsellik.
      gcs_kol_tepkisini_sinifla docstring'i, KESIN tibbi olcum DEGIL)
  m = mikrofon->hoparlor CANLI GECISI ac/kapat (BASLANGICTA KAPALI, bkz.
      ses.py) - ACIKKEN mikrofona konusulan ses ANINDA hoparlorden calinir
      (yerel interkom). SES sayaci (yatan kisinin ses cikarmasi) bundan
      BAGIMSIZ calisir - ayarlar.AKTIF_SES=True oldugu surece HER ZAMAN
      dinlemede, 'm' sadece hoparlore GECISI acar/kapatir.
  q = cikis

SAYAÇLAR gaze_birlesik.py ile AYNI isimlerde (kirpma, sol/sag/yukari/asagi,
sol_kol/sag_kol, sol_bacak/sag_bacak, sol_parmak/sag_parmak) - SADECE
VERI KAYNAGI degisti: yuz/el sayaclari artik genis kareden DEGIL, ilgili
SABIT bolge panelinden besleniyor (bolge tanimlanmadiysa o sayaclar hic
artmaz, 0'da kalir - genis-aci gorunumde bunun NEDENI ekranda yazar).
"kafa" (19.08.2026, rotasyon VEYA konum degisikligi) ve "ses" (mikrofon,
bkz. ses.py) sayaçlari ise SADECE bu dosyada VAR - gaze_birlesik.py'de yok.
"""
import collections
import math
import time
import types

import cv2
import mediapipe as mp
import numpy as np

import ayarlar as A
import bolgeler as B
import gorsellik as G
import kayit as K
import modeller as M
import ses as S

# --- Bolgeleri (nokta_sec.py ile isaretlenmis) yukle -----------------------
bolgeler = B.bolgeleri_yukle()
bolge_turleri = {b["tur"] for b in bolgeler.values()}
if not bolgeler:
    print("[uyari] Hic bolge tanimli degil (zoom_noktalari.json yok/bos). Once 'python "
          "nokta_sec.py' ile en az bir bolge (yuz/sol el/sag el) isaretle. Simdilik SADECE "
          "genis-aci (kol/bacak) gorunumuyle devam ediliyor.")
else:
    print(f"[bilgi] {len(bolgeler)} bolge yuklendi: {sorted(bolgeler)}")

# --- Modelleri yukle ---------------------------------------------------------
# Yuz/el icin AYRI/kendi (tek-hedef) landmarker'lar - bkz. modeller.
# bolge_landmarklarini_yukle docstring'i. SADECE FIILEN gereken (hem bolgesi
# TANIMLI hem ilgili AKTIF_* bayragi acik olan) turler yuklenir - gereksiz
# model yukleme/CPU maliyetinden kacinmak icin.
_bolge_turleri_yuklenecek = set()
if "yuz" in bolge_turleri:
    _bolge_turleri_yuklenecek.add("yuz")
if "el" in bolge_turleri:
    if A.AKTIF_EL:
        _bolge_turleri_yuklenecek.add("el")
    else:
        print("[bilgi] ayarlar.AKTIF_EL=False - el bolgeleri tanimli olsa da PARMAK sayaclari calismayacak.")
bolge_face, bolge_hand = M.bolge_landmarklarini_yukle(
    _bolge_turleri_yuklenecek,
    el_bolge_adlari=[ad for ad, b in bolgeler.items() if b["tur"] == "el"],
)

# --- ANA KAMERA modu ('z' tusu) icin AYRI/bagimsiz landmarker cifti -------
# SABIT BOLGE panellerinden (yukaridaki bolge_face/bolge_hand) TAMAMEN
# BAGIMSIZ - bolgeler hic tanimli olmasa (nokta_sec.py hic calistirilmamis
# olsa) BILE yuklenir, cunku bu modun butun amaci nokta_sec.py'ye ihtiyac
# DUYMADAN dogrudan genis-aci/ana kameradan yuz+el tespiti yapabilmek (bkz.
# yukaridaki dosya docstring'i, "z" kontrolu).
_ana_kamera_turleri = {"yuz"}
if A.AKTIF_EL:
    _ana_kamera_turleri.add("el")
ana_face, ana_hand = M.bolge_landmarklarini_yukle(_ana_kamera_turleri, etiket="ana kamera, zoom kapali")

gaze = None
if A.AKTIF_GAZE:
    gaze = M.gaze_pipeline_yukle()
else:
    print("[bilgi] ayarlar.AKTIF_GAZE=False - KIRPMA (bolgede VE ana kamera modunda) sayilir ama "
          "bakis yonu (SOL/SAG/YUKARI/ASAGI) sayaclari ve goz oku calismayacak.")
# NOT (19.08.2026, 'z' ana kamera modu eklendi): eskiden gaze SADECE "yuz"
# bolgesi TANIMLIYSA yuklenirdi (bolge yoksa nasil olsa kullanilamiyordu).
# ARTIK ana_face/ana_hand HER ZAMAN yuklendigi (bkz. yukarida) icin bolge
# hic tanimli olmasa BILE 'z' ile ana kamera modunda bakis sayaclari
# calisabilmeli - bu yuzden yukleme SADECE A.AKTIF_GAZE'e bagli.

pose_landmarker = M.pose_landmarker_yukle() if A.AKTIF_POSE else None
if not A.AKTIF_POSE:
    print("[bilgi] ayarlar.AKTIF_POSE=False - genis-aci govde/kol/bacak sayaclari calismayacak.")

PoseLandmark = mp.tasks.vision.PoseLandmark

S.baslat()
if not A.AKTIF_SES:
    print("[bilgi] ayarlar.AKTIF_SES=False - SES sayaci ve mikrofon->hoparlor gecisi calismayacak.")

# --- Kamera --------------------------------------------------------------
_kayitli_veri_indeksi = None
if A.BOLGE_NOKTALARI_DOSYASI.exists():
    try:
        import json as _json
        with open(A.BOLGE_NOKTALARI_DOSYASI, "r", encoding="utf-8") as _f:
            _kayitli_veri_indeksi = _json.load(_f).get("kamera_indeksi")
    except (OSError, ValueError):
        pass
if _kayitli_veri_indeksi is not None and _kayitli_veri_indeksi != A.KAMERA_INDEKSI:
    print(f"[uyari] nokta_sec.py bolgeleri kamera indeksi {_kayitli_veri_indeksi} ile isaretlemis, "
          f"ama ayarlar.KAMERA_INDEKSI={A.KAMERA_INDEKSI}. Ayni FIZIKSEL kamera degilse "
          "bolge koordinatlari YANLIS yerlere denk gelebilir - gerekirse nokta_sec.py'yi "
          "tekrar calistir ya da ayarlar.py'yi duzelt.")
_t2 = time.time()
cap = cv2.VideoCapture(A.KAMERA_INDEKSI)
if not cap.isOpened():
    raise SystemExit(
        f"Kamera acilamadi (indeks {A.KAMERA_INDEKSI}). Baska uygulama kullaniyor "
        "olabilir ya da ayarlar.py'deki KAMERA_INDEKSI yanlis kamerayi gosteriyor "
        "olabilir - kamerabul.py ile dogrulayabilirsin."
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
    f"[zaman] kamera acildi (indeks {A.KAMERA_INDEKSI}): {time.time() - _t2:.1f}s "
    f"- kameranin kendi/varsayilan modu: {gercek_genislik}x{gercek_yukseklik}@{gercek_fps:.0f}fps"
)

# --- Sayaclar / durum (bkz. gaze_birlesik.py - AYNI isim/anlam) -----------
sayaclar = {
    "sag": 0, "sol": 0, "yukari": 0, "asagi": 0, "kirpma": 0, "kesit": 0,
    "sol_kol": 0, "sag_kol": 0, "sol_bacak": 0, "sag_bacak": 0,
    "sol_parmak": 0, "sag_parmak": 0, "kafa": 0, "ses": 0,
}
sayaclar_aktif = False
S.aktif = sayaclar_aktif  # SES sayaci da diger sayaclarla AYNI 'h' anahtarina baglı (bkz. asagida 'h' tusu)
ana_kamera_modu = False  # 'z' tusu - True iken SABIT BOLGE/zoom sistemi devre disi, yuz+el DOGRUDAN genis-aci/ana kameradan okunur
onceki_yatay = "merkez"
onceki_dikey = "merkez"
onceki_sol_kol_aktif = False
onceki_sag_kol_aktif = False
gcs_test_aktif = False
gcs_test_baslangic_zaman = None
gcs_sol_omuz_baslangic_y = None
gcs_sag_omuz_baslangic_y = None
gcs_sol_dirsek_baslangic_acisi = None
gcs_sag_dirsek_baslangic_acisi = None
gcs_sol_ornekler = []
gcs_sag_ornekler = []
gcs_son_sonuc = None
gcs_sonuc_gosterim_bitis = 0.0
sol_kol_cikis_sayaci = 0
sag_kol_cikis_sayaci = 0
sol_kol_hizli_x = sol_kol_hizli_y = None
sol_kol_yavas_x = sol_kol_yavas_y = None
sag_kol_hizli_x = sag_kol_hizli_y = None
sag_kol_yavas_x = sag_kol_yavas_y = None
sol_bacak_hizli_x = sol_bacak_hizli_y = None
sol_bacak_yavas_x = sol_bacak_yavas_y = None
sag_bacak_hizli_x = sag_bacak_hizli_y = None
sag_bacak_yavas_x = sag_bacak_yavas_y = None
onceki_sol_bacak_hareketli = False
onceki_sag_bacak_hareketli = False
sol_bacak_cikis_sayaci = 0
sag_bacak_cikis_sayaci = 0

# --- KAFA hareketi: YUZ bloklariyla (SABIT BOLGE VE ana kamera - ikisi de
# ayni anda hic calismadigi icin TEK/paylasilan durum yeterli, bkz.
# onceki_yatay/onceki_dikey ile AYNI mantik) PAYLASILAN durum -----------
kafa_hizli_x = kafa_hizli_y = None
kafa_yavas_x = kafa_yavas_y = None
onceki_kafa_hareketli = False
kafa_cikis_sayaci = 0
kafa_yaw_gecmis_ham = collections.deque(maxlen=2)
kafa_pitch_gecmis_ham = collections.deque(maxlen=2)

# KAFA KONUMU (oteleme - orn. "kafayi yerden yukari kaldirma", bkz. ayarlar.
# KAFA_KONUM_HAREKET_ESIK aciklamasi) - yukaridaki (rotasyon) durumundan
# TAMAMEN BAGIMSIZ, kendi hareket_algila durumu.
kafa_konum_hizli_x = kafa_konum_hizli_y = None
kafa_konum_yavas_x = kafa_konum_yavas_y = None
onceki_kafa_konum_hareketli = False
kafa_konum_cikis_sayaci = 0
kafa_konum_x_gecmis_ham = collections.deque(maxlen=2)
kafa_konum_y_gecmis_ham = collections.deque(maxlen=2)

kilitli_govde_merkez = None
govde_kayip_kare = 0
govde_olcek_yumusak = None

# --- YUZ bolgesi durumu (bkz. gaze_birlesik.py - AYNI degiskenler, kaynak
# SADECE genis kare yerine "yuz" bolge paneli) ------------------------------
BIAS_GX = 0.0
BIAS_GY = 0.0
son_gx_ham = 0.0
son_gy_ham = 0.0
yumusak_gx = None
yumusak_gy = None
yumusak_merkez_x = None
yumusak_merkez_y = None
yumusak_uzunluk = None
goz_kapali_onceki = False
cizgi_sol_x = cizgi_sag_x = cizgi_ust_y = cizgi_alt_y = None
yuz_bulundu_bu_kare = False  # 'c' kalibrasyonu icin - HER karede yeniden hesaplanir

# --- EL bolgeleri durumu: her "el" turundeki bolge (sol_el/sag_el) icin
# KENDI BAGIMSIZ 5-parmak durumu (bkz. gaze_birlesik.py parmak_durum ile
# AYNI yapi - burada dict-of-dict yerine ad ile anahtarlanan bir sozluk).
bolge_parmak_durum = {
    ad: [dict(hz_x=None, hz_y=None, hz_z=None, son_tetik=9999,
              gecmis_ham=collections.deque(maxlen=2)) for _ in range(5)]
    for ad, b in bolgeler.items() if b["tur"] == "el"
}
# GRUP (el-basi, 5-parmak-BIRLIKTE) refractory: yukaridaki 5 parmak ucunun
# HER BIRI kendi son_tetik'ini BAGIMSIZ tutuyor, yani 5 ucun tetiklenmeleri
# birbirine gore KAYDIRILMIS olsa bile (orn. basparmak kare N'de, isaret
# parmagi kare N+2'de tetiklenirse) SAYAC neredeyse HER karede artabiliyor -
# gercek/tek bir elde nadiren olur ama GERCEK OLMAYAN bir el tespitinde
# (bkz. EL_TESPIT_ESIK aciklamasi, ayarlar.py) 5 ucun HEPSI ayni titrek/
# gurultulu kaynaktan geldigi icin sIk sIk olur. Bu YENI sayac SAYACIN
# KENDISININ (5 parmaktan HERHANGI BIRININ tetiklemesiyle) en az
# PARMAK_YENIDEN_TETIK_MIN_KARE kare gecmeden TEKRAR ARTMAMASINI saglar -
# EL_TESPIT_ESIK'ten BAGIMSIZ, ek/ikinci bir guvenlik katmani.
bolge_grup_son_tetik = {ad: 9999 for ad in bolge_parmak_durum}
# ANA KAMERA modu ('z' tusu) icin AYRI parmak durumu: bolge_parmak_durum'un
# (yukarida) MUTLAK/panel-ici konumundan FARKLI OLCEKTE calisir - burada
# gaze_birlesik.py ile AYNI yontemle (bilege GORE, el buyuklugune OLCEKLI,
# bkz. A.PARMAK_HIZ_ESIK_GORELI) izlenir, o yuzden KENDI/bagimsiz durumu var.
ana_grup_son_tetik = {"sol": 9999, "sag": 9999}  # bkz. bolge_grup_son_tetik aciklamasi - AYNI amac
ana_parmak_durum = {
    "sol": [dict(hz_x=None, hz_y=None, son_tetik=9999) for _ in range(5)],
    "sag": [dict(hz_x=None, hz_y=None, son_tetik=9999) for _ in range(5)],
}
# el_olcegi (bilek-orta parmak kok mesafesi, 5 parmagin ORTAK paydasi) icin
# per-taraf YUMUSATILMIS deger - bkz. ayarlar.EL_OLCEK_YUMUSATMA_ORANI
# aciklamasi (19.08.2026, gercek video kanitiyla bulundu: el yandan/profilden
# gorununce bu mesafe ANINDA kisalip 5 parmagi BIRDEN yanlis tetikliyordu).
ana_el_olcek_yumusak = {"sol": None, "sag": None}

# NOT (18.08.2026, kullanici karari): parmak hareketi ARTIK bilege GORELI
# DEGIL, MUTLAK (panel-ici normalize) konumla izleniyor - bkz. asagidaki
# "el" bolgesi blogu. Bu kamera SABIT ve hastanin eli/bilegi buyuk olcude
# ayni yerde durdugu icin (govdeye_goreli_konum'un ana amaci olan "kisi
# kameraya yakin/uzak, govde kayiyor" senaryolari burada GECERSIZ), 
# gorece basit MUTLAK konum yeterli VE bilek kirpma alani disinda
# kaldiginda olusan gurultu sorununu (bkz. proje sohbet gecmisi) KOKTEN
# ortadan kaldiriyor - artik bilek REFERANS olarak hic KULLANILMIYOR.
# BEDELI: hastanin eli/bilegi GERCEKTEN yer degistirirse (parmaklar hic
# kipirdamasa bile) bu da "parmak hareketi" olarak sayilabilir - kamera
# ve el konumu sabit kaldigi surece bu bir sorun degil.

kare_zaman_damgasi_ms = 0  # pose + bolge_face + bolge_hand icin ORTAK, artan sahte zaman damgasi (bkz. gaze_birlesik.py ile AYNI yaklasim)

kaydedici = K.VideoKaydedici()
# Bolgeler (zoom izgarasi) icin AYRI video kaydedici - 'v' tusu ikisini
# BIRLIKTE baslatir/durdurur (bkz. asagidaki 'v' tus isleyicisi), ama
# dosyalari AYRI ('video_...mp4' / 'video_bolgeler_...mp4') kalir.
bolge_kaydedici = K.VideoKaydedici(dosya_on_eki="video_bolgeler")

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

print("'c' ile (yuz bolgesi tanimliysa) kalibre et. 'r' = kesit al. 'v' = video kaydi ac/kapat (genis-aci + bolgeler). "
      "'z' = ana kamera modu ac/kapat (zoom kapali, sayaç dogrudan genis-aci kameradan). Cikis: 'q'.")

ilk_kare_mi = True

while True:
    ok, kare_ham = cap.read()
    if not ok:
        break

    if ilk_kare_mi:
        _t6 = time.time()

    h, w = kare_ham.shape[:2]
    kare = kare_ham.copy()  # genis-aci (kol/bacak) penceresi BU kare uzerine cizilir
    yuz_bulundu_bu_kare = False

    _dbg_sol_kol = "?"
    _dbg_sag_kol = "?"
    _dbg_sol_bacak = "?"
    _dbg_sag_bacak = "?"
    _dbg_sol_parmak = "?"
    _dbg_sag_parmak = "?"
    _dbg_kafa = "?"
    # ANA KAMERA modunda ('z') el-govde eslestirmesi icin - bkz. asagidaki
    # ana kamera EL blogu VE gaze_birlesik.py'deki AYNI mantik/aciklama.
    _sol_bilek_gecerli = False
    _sag_bilek_gecerli = False

    kare_zaman_damgasi_ms += 33

    sayaclar["ses"] = S.ses_sayaci  # ses.py KENDI (PortAudio) thread'inde sayiyor, burada SADECE HUD icin okunuyor

    # =====================================================================
    # GENIS-ACI: govde/kol/bacak (gaze_birlesik.py'deki pose blogu ile
    # NEREDEYSE BIREBIR AYNI mantik - degismedi, sadece degisken/kare adlari
    # bu dosyaya tasindi).
    # =====================================================================
    if A.AKTIF_POSE and pose_landmarker is not None:
        try:
            rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_sonuc = pose_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

            secili_govde_i = None
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
                    secili_govde_i, kilitli_govde_merkez, govde_kayip_kare = G.kilitli_aday_sec(
                        kilitli_govde_merkez, govde_kayip_kare, govde_merkezleri, govde_buyuklukleri,
                        A.KIMLIK_KILIDI_MAKS_SICRAMA_ORANI, A.KIMLIK_KILIDI_KAYIP_KARE_LIMITI,
                    )
                else:
                    secili_govde_i = 0

            if A.GOVDE_CIZIMI_GOSTER:
                _cizilecek = [pose_sonuc.pose_landmarks[secili_govde_i]] if secili_govde_i is not None else []
                G.govde_ciz(kare, types.SimpleNamespace(pose_landmarks=_cizilecek))

            if secili_govde_i is None:
                govde_olcek_yumusak = None

            if secili_govde_i is not None:
                lm = pose_sonuc.pose_landmarks[secili_govde_i]

                sol_omuz = lm[PoseLandmark.LEFT_SHOULDER]
                sag_omuz = lm[PoseLandmark.RIGHT_SHOULDER]
                if A.EKRANA_GORE_SOL_SAG:
                    sol_omuz, sag_omuz = G.ekran_sol_sag_ayikla(sol_omuz, sag_omuz)
                _govde_olcek_ham = G.govde_olcek_hesapla(sol_omuz, sag_omuz)
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
                        print(f"[GCS] SOL KOL: {_gcs_sol_etiket or 'TEPKI YOK/IZLENEMEDI'}  detay={_gcs_sol_detay}")
                        print(f"[GCS] SAG KOL: {_gcs_sag_etiket or 'TEPKI YOK/IZLENEMEDI'}  detay={_gcs_sag_detay}")
                        print("[GCS] UYARI: Bu SEZGISEL bir ON-ONERI, kesin tibbi olcum degil - klinisyen kendi gozlemiyle dogrulamali.")
                        gcs_sonuc_gosterim_bitis = time.time() + A.GCS_SONUC_GOSTERIM_SANIYE
                        gcs_test_aktif = False

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
        except Exception:
            pass

    # =====================================================================
    # SABIT BOLGELER: her nokta_sec.py bolgesi icin kirp+buyut(zoom), genis
    # kare uzerinde alanini ciz, turune gore YUZ ya da EL tespiti calistir.
    # =====================================================================
    pw, ph = A.BOLGE_PANEL_GENISLIK, A.BOLGE_PANEL_YUKSEKLIK
    _renkler = {"yuz": (0, 255, 0), "sol_el": (255, 0, 255), "sag_el": (0, 255, 255)}
    paneller = []

    # ana_kamera_modu AKTIFKEN bolgeler HIC islenmez (bos sozluk uzerinde
    # donulur) - asagida ayni ismi/verisiyi ANA KAMERA blogu besler.
    for ad, bilgi in ({} if ana_kamera_modu else bolgeler).items():
        panel, kirpma_rect = G.bolge_kirp(kare_ham, bilgi["x"], bilgi["y"], bilgi["oran"], pw, ph)
        x1, y1, kw, kh = kirpma_rect
        cv2.rectangle(kare, (x1, y1), (x1 + kw, y1 + kh), _renkler[ad], 2)
        cv2.putText(kare, ad, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _renkler[ad], 2)

        if bilgi["tur"] == "yuz" and bolge_face is not None:
            try:
                rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                yuz_sonuc = bolge_face.detect_for_video(mp_image, kare_zaman_damgasi_ms)

                if yuz_sonuc.face_landmarks:
                    landmarks = yuz_sonuc.face_landmarks[0]

                    if yuz_sonuc.face_blendshapes:
                        skorlar = {b.category_name: b.score for b in yuz_sonuc.face_blendshapes[0]}
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

                    if A.AKTIF_GAZE and gaze is not None:
                        if yuz_sonuc.facial_transformation_matrixes:
                            rot = np.array(yuz_sonuc.facial_transformation_matrixes[0])[:3, :3]
                            pitch, yaw, roll = G.donus_matrisinden_aci(rot)
                        else:
                            yaw, pitch, roll = 0.0, 0.0, 0.0

                        _yaw_ef = G.medyan_3_yumusat(kafa_yaw_gecmis_ham, yaw)
                        _pitch_ef = G.medyan_3_yumusat(kafa_pitch_gecmis_ham, pitch)
                        (kafa_hareketli, kafa_hizli_x, kafa_hizli_y,
                         kafa_yavas_x, kafa_yavas_y, kafa_cikis_sayaci) = G.hareket_algila(
                            kafa_hizli_x, kafa_hizli_y, kafa_yavas_x, kafa_yavas_y,
                            _yaw_ef, _pitch_ef,
                            onceki_kafa_hareketli, A.KAFA_HAREKET_ESIK,
                            A.KAFA_HAREKET_HISTEREZIS_ORANI, A.KAFA_HAREKET_HIZLI_ORAN, A.KAFA_HAREKET_YAVAS_ORAN,
                            kafa_cikis_sayaci, A.KAFA_HAREKET_MIN_CIKIS_KARE,
                        )
                        kafa_rot_tetiklendi = kafa_hareketli and not onceki_kafa_hareketli
                        onceki_kafa_hareketli = kafa_hareketli
                        _dbg_kafa = math.hypot(kafa_hizli_x - kafa_yavas_x, kafa_hizli_y - kafa_yavas_y)

                        sx1, sy1, sx2, sy2 = G.goz_kutusu(landmarks, G.SAG_GOZ_IDX, pw, ph)
                        lx1, ly1, lx2, ly2 = G.goz_kutusu(landmarks, G.SOL_GOZ_IDX, pw, ph)
                        sag_goz = panel[sy1:sy2, sx1:sx2]
                        sol_goz = panel[ly1:ly2, lx1:lx2]

                        if sag_goz.size > 0 and sol_goz.size > 0:
                            yuz_bulundu_bu_kare = True
                            sag_goz = G.kirpinti_dondur(sag_goz, roll)
                            sol_goz = G.kirpinti_dondur(sol_goz, roll)
                            gaze.infer({
                                "left_eye_image": G.kirpinti_hazirla(sol_goz),
                                "right_eye_image": G.kirpinti_hazirla(sag_goz),
                                "head_pose_angles": np.array([[yaw, pitch, 0.0]], dtype=np.float32),
                            })
                            vektor = gaze.get_output_tensor().data[0].copy()
                            vektor = vektor / (np.linalg.norm(vektor) + 1e-9)

                            rad = np.radians(roll)
                            cs, sn = np.cos(rad), np.sin(rad)
                            son_gx_ham = float(vektor[0] * cs + vektor[1] * sn)
                            son_gy_ham = float(-vektor[0] * sn + vektor[1] * cs)
                            gx = son_gx_ham - BIAS_GX
                            gy = son_gy_ham - BIAS_GY

                            _ham_gx, _ham_gy = gx, gy
                            if _ham_gy < 0:
                                gx -= A.BAKIS_ASAGI_SIZINTI_K * (-_ham_gy)
                            gy += A.BAKIS_YANAL_SIZINTI_K * abs(_ham_gx)

                            x_min, y_min, x_max, y_max = G.yuz_bbox_hesapla(landmarks, pw, ph)
                            if A.YUZ_CIZIMI_GOSTER:
                                cv2.rectangle(panel, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)

                            merkez_x = (x_min + x_max) / 2.0
                            merkez_y = (y_min + y_max) / 2.0
                            uzunluk = x_max - x_min

                            gx = yumusak_gx = G.yumusat(yumusak_gx, gx, A.MAKS_BAKIS_SICRAMA)
                            gy = yumusak_gy = G.yumusat(yumusak_gy, gy, A.MAKS_BAKIS_SICRAMA)
                            merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
                            merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
                            uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

                            _konum_x = merkez_x / max(uzunluk, 1.0)
                            _konum_y = merkez_y / max(uzunluk, 1.0)
                            _konum_x_ef = G.medyan_3_yumusat(kafa_konum_x_gecmis_ham, _konum_x)
                            _konum_y_ef = G.medyan_3_yumusat(kafa_konum_y_gecmis_ham, _konum_y)
                            (kafa_konum_hareketli, kafa_konum_hizli_x, kafa_konum_hizli_y,
                             kafa_konum_yavas_x, kafa_konum_yavas_y, kafa_konum_cikis_sayaci) = G.hareket_algila(
                                kafa_konum_hizli_x, kafa_konum_hizli_y, kafa_konum_yavas_x, kafa_konum_yavas_y,
                                _konum_x_ef, _konum_y_ef,
                                onceki_kafa_konum_hareketli, A.KAFA_KONUM_HAREKET_ESIK,
                                A.KAFA_KONUM_HAREKET_HISTEREZIS_ORANI, A.KAFA_KONUM_HAREKET_HIZLI_ORAN,
                                A.KAFA_KONUM_HAREKET_YAVAS_ORAN, kafa_konum_cikis_sayaci,
                                A.KAFA_KONUM_HAREKET_MIN_CIKIS_KARE,
                            )
                            kafa_konum_tetiklendi = kafa_konum_hareketli and not onceki_kafa_konum_hareketli
                            onceki_kafa_konum_hareketli = kafa_konum_hareketli
                            if (kafa_rot_tetiklendi or kafa_konum_tetiklendi) and sayaclar_aktif:
                                sayaclar["kafa"] += 1

                            cizgi_sol_x = int(merkez_x - uzunluk * A.KENAR_MESAFE_YATAY)
                            cizgi_sag_x = int(merkez_x + uzunluk * A.KENAR_MESAFE_YATAY)
                            cizgi_ust_y = int(merkez_y - uzunluk * A.KENAR_MESAFE_UST)
                            cizgi_alt_y = int(merkez_y + uzunluk * A.KENAR_MESAFE_ALT)

                            dx = uzunluk * gx
                            dy = -uzunluk * gy
                            ucur_x = merkez_x + dx
                            ucur_y = merkez_y + dy
                            duz_bakiyor = abs(gx) < A.ESIK_BAKIS_XY and abs(gy) < A.ESIK_BAKIS_XY

                            if A.YUZ_CIZIMI_GOSTER and not duz_bakiyor:
                                cv2.arrowedLine(panel, (int(merkez_x), int(merkez_y)), (int(ucur_x), int(ucur_y)),
                                                 (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)

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

                            if A.YUZ_CIZIMI_GOSTER and cizgi_sol_x is not None:
                                cv2.rectangle(panel, (cizgi_sol_x, cizgi_ust_y), (cizgi_sag_x, cizgi_alt_y), (255, 255, 0), 1)
            except Exception:
                pass

            cv2.putText(panel, "YUZ", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(panel, f"KIRPMA:{sayaclar['kirpma']} S:{sayaclar['sol']} G:{sayaclar['sag']} "
                                f"Y:{sayaclar['yukari']} A:{sayaclar['asagi']} K:{sayaclar['kafa']}",
                        (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

        elif bilgi["tur"] == "el" and bolge_hand is not None and ad in bolge_hand:
            _en_buyuk_hiz = 0.0
            try:
                rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                el_sonuc = bolge_hand[ad].detect_for_video(mp_image, kare_zaman_damgasi_ms)
                if A.EL_CIZIMI_GOSTER:
                    G.eller_ciz(panel, el_sonuc)

                _eller = el_sonuc.hand_landmarks
                if _eller:
                    # Panel MERKEZINE en yakin bilek = bu bolgenin hedef eli
                    # - zone'un adi/konumu ZATEN "sol"/"sag" kimligini
                    # belirledigi icin (nokta_sec.py'de SEN isaretledin),
                    # gaze_birlesik.py'deki govde-bilek eslestirmesine
                    # (MediaPipe'in kendi guvenilmez handedness etiketi
                    # yerine) GEREK YOK.
                    _merkez_x, _merkez_y = pw / 2.0, ph / 2.0
                    _en_yakin_i = min(
                        range(len(_eller)),
                        key=lambda i: math.hypot(_eller[i][0].x * pw - _merkez_x, _eller[i][0].y * ph - _merkez_y),
                    )
                    _el = _eller[_en_yakin_i]

                    # MUTLAK konum: her parmak ucunun PANEL icindeki HAM
                    # (normalize 0..1) konumu DOGRUDAN kullanilir - bilek
                    # referansi/olcegi YOK (bkz. yukaridaki aciklama).
                    _durumlar = bolge_parmak_durum[ad]
                    _herhangi_biri_aktif = False
                    for _pi, _uc_idx in enumerate((4, 8, 12, 16, 20)):
                        _d = _durumlar[_pi]
                        _uc = _el[_uc_idx]
                        # z de veriliyor (18.08.2026, kullanici istegi:
                        # "z eksenini hesaba kat") - parmak kaldirma hareketi
                        # kameraya göre derinlik ekseninde de olabilir, sadece
                        # x,y boyle bir hareketi kacirabiliyordu (bkz.
                        # gorsellik.parmak_hareket_algila docstring'i).
                        (_tetiklendi, _d["hz_x"], _d["hz_y"], _d["hz_z"], _d["son_tetik"], _hiz) = G.parmak_hareket_algila(
                            _d["hz_x"], _d["hz_y"], _uc.x, _uc.y,
                            A.PARMAK_HIZ_ESIK, A.PARMAK_HIZ_HIZLI_ORAN,
                            _d["son_tetik"], A.PARMAK_YENIDEN_TETIK_MIN_KARE,
                            hizli_z=_d["hz_z"], z=_uc.z, gecmis_ham=_d["gecmis_ham"],
                        )
                        if _tetiklendi:
                            _herhangi_biri_aktif = True
                        _en_buyuk_hiz = max(_en_buyuk_hiz, _hiz)

                    _sayac_adi = "sol_parmak" if ad == "sol_el" else "sag_parmak"
                    bolge_grup_son_tetik[ad] += 1
                    if (_herhangi_biri_aktif and sayaclar_aktif
                            and bolge_grup_son_tetik[ad] >= A.PARMAK_YENIDEN_TETIK_MIN_KARE):
                        sayaclar[_sayac_adi] += 1
                        bolge_grup_son_tetik[ad] = 0
                        if A.PARMAK_OLAY_KESITI_AKTIF:
                            (sol_parmak_olay_kaydedici if ad == "sol_el" else sag_parmak_olay_kaydedici).olay_tetikle(_sayac_adi)
            except Exception:
                pass

            _etiket = "SOL EL" if ad == "sol_el" else "SAG EL"
            _sayac_adi_goster = "SOL PARMAK" if ad == "sol_el" else "SAG PARMAK"
            _sayac_deger = sayaclar["sol_parmak"] if ad == "sol_el" else sayaclar["sag_parmak"]
            cv2.putText(panel, _etiket, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _renkler[ad], 2)
            cv2.putText(panel, f"{_sayac_adi_goster}: {_sayac_deger}   hiz:{_en_buyuk_hiz:.3f} (esik {A.PARMAK_HIZ_ESIK:.2f})",
                        (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1)

        paneller.append(panel)

    izgara = G.izgaraya_diz(paneller) if paneller else None
    if izgara is not None:
        bolge_kaydedici.kare_ekle(izgara)
        if bolge_kaydedici.kayit_yapiliyor:
            cv2.circle(izgara, (izgara.shape[1] - 30, 30), 8, (0, 0, 255), -1)
            cv2.putText(izgara, "REC", (izgara.shape[1] - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # =====================================================================
    # ANA KAMERA MODU ('z' tusu ile acilir/kapanir): SABIT BOLGE/zoom
    # sistemi TAMAMEN devre disi - YUZ (kirpma/bakis) VE EL (parmak)
    # DOGRUDAN genis-aci/ana kameradan (kare_ham/kare, w x h) okunur.
    # Mantik gaze_birlesik.py'nin (yakin/webcam surumu) YUZ ve EL bloklariyla
    # NEREDEYSE BIREBIR AYNI - farkli olan SADECE veri kaynagi (kucuk/zoom'lu
    # bir panel yerine dogrudan genis kare) VE bagimsiz landmarker/durum
    # (ana_face/ana_hand, ana_parmak_durum) kullanilmasi.
    # =====================================================================
    if ana_kamera_modu:
        try:
            rgb = cv2.cvtColor(kare_ham, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if ana_face is not None:
                yuz_sonuc = ana_face.detect_for_video(mp_image, kare_zaman_damgasi_ms)

                if yuz_sonuc.face_landmarks:
                    landmarks = yuz_sonuc.face_landmarks[0]

                    if yuz_sonuc.face_blendshapes:
                        skorlar = {b.category_name: b.score for b in yuz_sonuc.face_blendshapes[0]}
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

                    if A.AKTIF_GAZE and gaze is not None:
                        if yuz_sonuc.facial_transformation_matrixes:
                            rot = np.array(yuz_sonuc.facial_transformation_matrixes[0])[:3, :3]
                            pitch, yaw, roll = G.donus_matrisinden_aci(rot)
                        else:
                            yaw, pitch, roll = 0.0, 0.0, 0.0

                        _yaw_ef = G.medyan_3_yumusat(kafa_yaw_gecmis_ham, yaw)
                        _pitch_ef = G.medyan_3_yumusat(kafa_pitch_gecmis_ham, pitch)
                        (kafa_hareketli, kafa_hizli_x, kafa_hizli_y,
                         kafa_yavas_x, kafa_yavas_y, kafa_cikis_sayaci) = G.hareket_algila(
                            kafa_hizli_x, kafa_hizli_y, kafa_yavas_x, kafa_yavas_y,
                            _yaw_ef, _pitch_ef,
                            onceki_kafa_hareketli, A.KAFA_HAREKET_ESIK,
                            A.KAFA_HAREKET_HISTEREZIS_ORANI, A.KAFA_HAREKET_HIZLI_ORAN, A.KAFA_HAREKET_YAVAS_ORAN,
                            kafa_cikis_sayaci, A.KAFA_HAREKET_MIN_CIKIS_KARE,
                        )
                        kafa_rot_tetiklendi = kafa_hareketli and not onceki_kafa_hareketli
                        onceki_kafa_hareketli = kafa_hareketli
                        _dbg_kafa = math.hypot(kafa_hizli_x - kafa_yavas_x, kafa_hizli_y - kafa_yavas_y)

                        sx1, sy1, sx2, sy2 = G.goz_kutusu(landmarks, G.SAG_GOZ_IDX, w, h)
                        lx1, ly1, lx2, ly2 = G.goz_kutusu(landmarks, G.SOL_GOZ_IDX, w, h)
                        sag_goz = kare_ham[sy1:sy2, sx1:sx2]
                        sol_goz = kare_ham[ly1:ly2, lx1:lx2]

                        if sag_goz.size > 0 and sol_goz.size > 0:
                            yuz_bulundu_bu_kare = True
                            sag_goz = G.kirpinti_dondur(sag_goz, roll)
                            sol_goz = G.kirpinti_dondur(sol_goz, roll)
                            gaze.infer({
                                "left_eye_image": G.kirpinti_hazirla(sol_goz),
                                "right_eye_image": G.kirpinti_hazirla(sag_goz),
                                "head_pose_angles": np.array([[yaw, pitch, 0.0]], dtype=np.float32),
                            })
                            vektor = gaze.get_output_tensor().data[0].copy()
                            vektor = vektor / (np.linalg.norm(vektor) + 1e-9)

                            rad = np.radians(roll)
                            cs, sn = np.cos(rad), np.sin(rad)
                            son_gx_ham = float(vektor[0] * cs + vektor[1] * sn)
                            son_gy_ham = float(-vektor[0] * sn + vektor[1] * cs)
                            gx = son_gx_ham - BIAS_GX
                            gy = son_gy_ham - BIAS_GY

                            _ham_gx, _ham_gy = gx, gy
                            if _ham_gy < 0:
                                gx -= A.BAKIS_ASAGI_SIZINTI_K * (-_ham_gy)
                            gy += A.BAKIS_YANAL_SIZINTI_K * abs(_ham_gx)

                            x_min, y_min, x_max, y_max = G.yuz_bbox_hesapla(landmarks, w, h)
                            if A.YUZ_CIZIMI_GOSTER:
                                cv2.rectangle(kare, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)

                            merkez_x = (x_min + x_max) / 2.0
                            merkez_y = (y_min + y_max) / 2.0
                            uzunluk = x_max - x_min

                            gx = yumusak_gx = G.yumusat(yumusak_gx, gx, A.MAKS_BAKIS_SICRAMA)
                            gy = yumusak_gy = G.yumusat(yumusak_gy, gy, A.MAKS_BAKIS_SICRAMA)
                            merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
                            merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
                            uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

                            _konum_x = merkez_x / max(uzunluk, 1.0)
                            _konum_y = merkez_y / max(uzunluk, 1.0)
                            _konum_x_ef = G.medyan_3_yumusat(kafa_konum_x_gecmis_ham, _konum_x)
                            _konum_y_ef = G.medyan_3_yumusat(kafa_konum_y_gecmis_ham, _konum_y)
                            (kafa_konum_hareketli, kafa_konum_hizli_x, kafa_konum_hizli_y,
                             kafa_konum_yavas_x, kafa_konum_yavas_y, kafa_konum_cikis_sayaci) = G.hareket_algila(
                                kafa_konum_hizli_x, kafa_konum_hizli_y, kafa_konum_yavas_x, kafa_konum_yavas_y,
                                _konum_x_ef, _konum_y_ef,
                                onceki_kafa_konum_hareketli, A.KAFA_KONUM_HAREKET_ESIK,
                                A.KAFA_KONUM_HAREKET_HISTEREZIS_ORANI, A.KAFA_KONUM_HAREKET_HIZLI_ORAN,
                                A.KAFA_KONUM_HAREKET_YAVAS_ORAN, kafa_konum_cikis_sayaci,
                                A.KAFA_KONUM_HAREKET_MIN_CIKIS_KARE,
                            )
                            kafa_konum_tetiklendi = kafa_konum_hareketli and not onceki_kafa_konum_hareketli
                            onceki_kafa_konum_hareketli = kafa_konum_hareketli
                            if (kafa_rot_tetiklendi or kafa_konum_tetiklendi) and sayaclar_aktif:
                                sayaclar["kafa"] += 1

                            cizgi_sol_x = int(merkez_x - uzunluk * A.KENAR_MESAFE_YATAY)
                            cizgi_sag_x = int(merkez_x + uzunluk * A.KENAR_MESAFE_YATAY)
                            cizgi_ust_y = int(merkez_y - uzunluk * A.KENAR_MESAFE_UST)
                            cizgi_alt_y = int(merkez_y + uzunluk * A.KENAR_MESAFE_ALT)

                            dx = uzunluk * gx
                            dy = -uzunluk * gy
                            ucur_x = merkez_x + dx
                            ucur_y = merkez_y + dy
                            duz_bakiyor = abs(gx) < A.ESIK_BAKIS_XY and abs(gy) < A.ESIK_BAKIS_XY

                            if A.YUZ_CIZIMI_GOSTER and not duz_bakiyor:
                                cv2.arrowedLine(kare, (int(merkez_x), int(merkez_y)), (int(ucur_x), int(ucur_y)),
                                                 (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)

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

                            if A.YUZ_CIZIMI_GOSTER and cizgi_sol_x is not None:
                                cv2.rectangle(kare, (cizgi_sol_x, cizgi_ust_y), (cizgi_sag_x, cizgi_alt_y), (255, 255, 0), 1)

            # --- EL: bilege GORE + el buyuklugune OLCEKLI parmak takibi
            # (gaze_birlesik.py ile AYNI yontem - bkz. ayarlar.
            # PARMAK_HIZ_ESIK_GORELI aciklamasi). MediaPipe'in kendi Sol/Sag
            # (handedness) etiketine GUVENILMIYOR - bu karede POSE'dan TAZE/
            # gorunur sol_bilek/sag_bilek varsa (bkz. yukaridaki POSE blogu,
            # _sol_bilek_gecerli/_sag_bilek_gecerli) her el KENDI en yakin
            # govde bilegine eslenir; govde bilegi YOKSA ama IKI el tespit
            # edildiyse ekrandaki x konumuna gore atanir; govde YOK VE TEK el
            # varsa o el ATLANIR (yanlis sol/sag atamasi yapmaktansa).
            if A.AKTIF_EL and ana_hand is not None:
                el_sonuc = ana_hand.detect_for_video(mp_image, kare_zaman_damgasi_ms)
                if A.EL_CIZIMI_GOSTER:
                    G.eller_ciz(kare, el_sonuc)

                _eller = el_sonuc.hand_landmarks
                if _eller:
                    _el_atama = [None] * len(_eller)  # 'sol' / 'sag' / None (belirsiz -> atla)
                    _govde_bilekleri = []
                    if _sol_bilek_gecerli:
                        _govde_bilekleri.append(("sol", sol_bilek))
                    if _sag_bilek_gecerli:
                        _govde_bilekleri.append(("sag", sag_bilek))

                    if _govde_bilekleri:
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

                    for _i, _taraf in enumerate(_el_atama):
                        if _taraf is None:
                            continue
                        _el = _eller[_i]
                        _el_bilek = _el[0]        # WRIST
                        _el_orta_kok = _el[9]     # MIDDLE_FINGER_MCP - el buyuklugu referansi
                        if A.EKRANA_GORE_ETIKET_GOSTER:
                            _el_renk = (255, 0, 255) if _taraf == "sol" else (0, 255, 255)
                            G.ekran_etiket_ciz(kare, _el_bilek, _taraf.upper(), _el_renk, w, h)

                        # el_olcegi'ni YUMUSAT (govde_olcek ile AYNI teknik,
                        # bkz. ayarlar.EL_OLCEK_YUMUSATMA_ORANI) - el yandan/
                        # profilden gorununce bu mesafe ANINDA kucul(ebil)ir,
                        # yumusatma OLMAZSA 5 parmagin TUMU (gercekte
                        # hareketsizken bile) BIRDEN yanlis tetiklenir.
                        _el_olcegi_ham = G.govde_olcek_hesapla(_el_bilek, _el_orta_kok, A.EL_OLCEK_MIN)
                        _el_olcegi_yumusak = ana_el_olcek_yumusak[_taraf]
                        if (
                            _el_olcegi_yumusak is None
                            or A.EL_OLCEK_KABUL_MIN_ORAN * _el_olcegi_yumusak
                            <= _el_olcegi_ham
                            <= A.EL_OLCEK_KABUL_MAKS_ORAN * _el_olcegi_yumusak
                        ):
                            _el_olcegi_yumusak = G.yumusat(
                                _el_olcegi_yumusak, _el_olcegi_ham,
                                A.EL_OLCEK_MAKS_SICRAMA, A.EL_OLCEK_YUMUSATMA_ORANI,
                            )
                        ana_el_olcek_yumusak[_taraf] = _el_olcegi_yumusak
                        _el_olcegi = _el_olcegi_yumusak

                        _durumlar = ana_parmak_durum[_taraf]
                        _herhangi_biri_aktif = False
                        _en_buyuk_hiz = 0.0
                        for _pi, _uc_idx in enumerate((4, 8, 12, 16, 20)):
                            _d = _durumlar[_pi]
                            _gx, _gy = G.govdeye_goreli_konum(_el[_uc_idx], _el_bilek, _el_olcegi)
                            (_parmak_tetiklendi, _d["hz_x"], _d["hz_y"], _, _d["son_tetik"], _hiz) = G.parmak_hareket_algila(
                                _d["hz_x"], _d["hz_y"], _gx, _gy,
                                A.PARMAK_HIZ_ESIK_GORELI, A.PARMAK_HIZ_HIZLI_ORAN,
                                _d["son_tetik"], A.PARMAK_YENIDEN_TETIK_MIN_KARE,
                            )
                            if _parmak_tetiklendi:
                                _herhangi_biri_aktif = True
                            _en_buyuk_hiz = max(_en_buyuk_hiz, _hiz)

                        ana_grup_son_tetik[_taraf] += 1
                        if _taraf == "sol":
                            _dbg_sol_parmak = _en_buyuk_hiz
                            if (_herhangi_biri_aktif and sayaclar_aktif
                                    and ana_grup_son_tetik["sol"] >= A.PARMAK_YENIDEN_TETIK_MIN_KARE):
                                sayaclar["sol_parmak"] += 1
                                ana_grup_son_tetik["sol"] = 0
                                if A.PARMAK_OLAY_KESITI_AKTIF:
                                    sol_parmak_olay_kaydedici.olay_tetikle("sol_parmak")
                        else:
                            _dbg_sag_parmak = _en_buyuk_hiz
                            if (_herhangi_biri_aktif and sayaclar_aktif
                                    and ana_grup_son_tetik["sag"] >= A.PARMAK_YENIDEN_TETIK_MIN_KARE):
                                sayaclar["sag_parmak"] += 1
                                ana_grup_son_tetik["sag"] = 0
                                if A.PARMAK_OLAY_KESITI_AKTIF:
                                    sag_parmak_olay_kaydedici.olay_tetikle("sag_parmak")
        except Exception:
            pass

        # Bolgeler penceresi bu modda islenmiyor (izgara=None kalirdi) -
        # kullaniciya modun neden bos gozuktugunu acikca soylemek icin
        # kucuk bir bilgi karesi gosteriliyor.
        izgara = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.putText(izgara, "ANA KAMERA MODU AKTIF", (10, ph // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(izgara, "(sabit bolge/zoom KAPALI - 'z' ile geri don)", (10, ph // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Genis-aci pencereye SAYAÇLAR/kontroller yaz ------------------------
    cv2.putText(kare, f"KIRPMA: {sayaclar['kirpma']}   KESIT: {sayaclar['kesit']}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(kare, f"SOL: {sayaclar['sol']}  SAG: {sayaclar['sag']}  YUKARI: {sayaclar['yukari']}  ASAGI: {sayaclar['asagi']}  KAFA: {sayaclar['kafa']}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(kare, f"SOL KOL: {sayaclar['sol_kol']}   SAG KOL: {sayaclar['sag_kol']}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(kare, f"SOL BACAK: {sayaclar['sol_bacak']}   SAG BACAK: {sayaclar['sag_bacak']}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(kare, f"SOL PARMAK: {sayaclar['sol_parmak']}   SAG PARMAK: {sayaclar['sag_parmak']}   "
                       f"SES: {sayaclar['ses']} (mikrofon gecisi {'ACIK' if S.passthrough_durumu() else 'KAPALI'}, m)",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(kare, f"SAYAÇLAR: {'AKTIF' if sayaclar_aktif else 'DURAKLATILDI'} (h)",
                (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if sayaclar_aktif else (0, 0, 255), 2)
    cv2.putText(kare, f"MOD: {'ANA KAMERA (zoom kapali)' if ana_kamera_modu else 'SABIT BOLGE (zoom)'} (z)",
                (300, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if ana_kamera_modu else (0, 255, 0), 2)
    if not bolgeler and not ana_kamera_modu:
        cv2.putText(kare, "BOLGE TANIMLI DEGIL - once 'python nokta_sec.py' calistir ya da "
                          "'z' ile ANA KAMERA moduna gec",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    if gcs_test_aktif:
        _gcs_kalan = max(0.0, A.GCS_PENCERE_SANIYE - (time.time() - gcs_test_baslangic_zaman))
        cv2.putText(kare, f"GCS TESTI: OLCULUYOR ({_gcs_kalan:.1f}sn) - uyaran uygula",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    elif gcs_son_sonuc is not None and time.time() < gcs_sonuc_gosterim_bitis:
        _gcs_sol_e, _gcs_sag_e = gcs_son_sonuc
        cv2.putText(kare, f"GCS SONUC (sezgisel): SOL={_gcs_sol_e or '?'}  SAG={_gcs_sag_e or '?'}",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(kare, "GCS TESTI: HAZIR (g)", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    def _dbg_fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else v
    cv2.putText(
        kare,
        f"TANI KOL sol:{_dbg_fmt(_dbg_sol_kol)} sag:{_dbg_fmt(_dbg_sag_kol)} (esik {A.KOL_HAREKET_GORELI_ESIK:.2f})  "
        f"BACAK sol:{_dbg_fmt(_dbg_sol_bacak)} sag:{_dbg_fmt(_dbg_sag_bacak)} (esik {A.BACAK_HAREKET_GORELI_ESIK:.2f})  "
        f"KAFA:{_dbg_fmt(_dbg_kafa)} (esik {A.KAFA_HAREKET_ESIK:.2f})  "
        f"SES giris:{S.son_giris_rms:.3f} cikis:{S.son_cikis_rms:.3f} (esik {A.SES_ALGILAMA_ESIK:.2f})",
        (20, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
    )
    if ana_kamera_modu:
        cv2.putText(
            kare,
            f"TANI PARMAK (ana kamera) sol:{_dbg_fmt(_dbg_sol_parmak)} sag:{_dbg_fmt(_dbg_sag_parmak)} "
            f"(esik {A.PARMAK_HIZ_ESIK_GORELI:.2f})",
            (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
        )
    cv2.putText(kare, "c: kalibre  |  r: kesit al  |  v: video kaydi (genis+bolgeler)  |  h: sayac ac/kapat  |  "
                      "m: mikrofon gecisi  |  z: ana kamera modu ac/kapat  |  g: GCS testi  |  q: cikis",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if kaydedici.kayit_yapiliyor:
        cv2.circle(kare, (w - 30, 30), 8, (0, 0, 255), -1)
        cv2.putText(kare, "REC", (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    kaydedici.kare_ekle(kare)
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

    cv2.imshow("Genis aci: govde/kol/bacak (q = cik)", kare)
    if izgara is not None:
        cv2.imshow("Bolgeler (zoom): yuz / sol el / sag el", izgara)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        if kaydedici.kayit_yapiliyor:
            kaydedici.bitir()
        if bolge_kaydedici.kayit_yapiliyor:
            bolge_kaydedici.bitir()
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
        if izgara is not None:
            _zaman_etiketi = time.strftime(A.ZAMAN_DAMGASI_FORMATI)
            _bolge_dosya_adi = A.KESIT_KLASORU / f"kesit_bolgeler_{_zaman_etiketi}.jpg"
            cv2.imwrite(str(_bolge_dosya_adi), izgara)
            print(f"Bolgeler kesiti kaydedildi: {_bolge_dosya_adi}")
    if tus == ord("v"):
        if not kaydedici.kayit_yapiliyor:
            kaydedici.baslat()
            bolge_kaydedici.baslat()
        else:
            kaydedici.bitir()
            bolge_kaydedici.bitir()
    if tus == ord("h"):
        sayaclar_aktif = not sayaclar_aktif
        S.aktif = sayaclar_aktif
        print(f"Sayaçlar {'AKTIF' if sayaclar_aktif else 'DURAKLATILDI'}.")
    if tus == ord("m"):
        _passthrough_simdi = S.passthrough_ac_kapat()
        print(f"Mikrofon->hoparlor gecisi {'AKTIF' if _passthrough_simdi else 'KAPALI'}.")
    if tus == ord("z"):
        ana_kamera_modu = not ana_kamera_modu
        if ana_kamera_modu:
            print("Ana kamera modu AKTIF - sabit bolge/zoom sistemi KAPALI, "
                  "yuz/el sayaclari dogrudan genis-aci kameradan besleniyor.")
        else:
            print("Ana kamera modu KAPALI - sabit bolge/zoom sistemine donuldu.")
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
S.durdur()
if pose_landmarker is not None:
    pose_landmarker.close()
if bolge_face is not None:
    bolge_face.close()
if bolge_hand is not None:
    for _lm in bolge_hand.values():
        _lm.close()
if ana_face is not None:
    ana_face.close()
if ana_hand is not None:
    ana_hand.close()
if kaydedici.kayit_yapiliyor:
    kaydedici.bitir()
if bolge_kaydedici.kayit_yapiliyor:
    bolge_kaydedici.bitir()
sol_kol_olay_kaydedici.bitir()
sag_kol_olay_kaydedici.bitir()
sol_bacak_olay_kaydedici.bitir()
sag_bacak_olay_kaydedici.bitir()
sol_parmak_olay_kaydedici.bitir()
sag_parmak_olay_kaydedici.bitir()
kirpma_olay_kaydedici.bitir()
bakis_olay_kaydedici.bitir()
print("Sayaclar:", sayaclar)