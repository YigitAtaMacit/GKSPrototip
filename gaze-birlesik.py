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
    "sol_kol": 0, "sag_kol": 0,
}
onceki_yatay = "merkez"
onceki_dikey = "merkez"
onceki_sol_kol_aktif = False
onceki_sag_kol_aktif = False

# Kimlik kilidi durumu (bkz. ayarlar.py, gorsellik.kilitli_aday_sec) - yuz
# ve govde icin AYRI kilitler (MediaPipe'in iki dedektoru birbirinden
# BAGIMSIZ calisir, ortak bir "kisi ID"si yok).
kilitli_yuz_merkez = None
yuz_kayip_kare = 0
kilitli_govde_merkez = None
govde_kayip_kare = 0

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
kirpma_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_KIRPMA_KLASORU, dosya_on_eki="kirpma",
)
bakis_olay_kaydedici = K.OlayKlibiYoneticisi(
    once_saniye=A.OLAY_ONCE_SANIYE, sonra_saniye=A.OLAY_SONRA_SANIYE,
    klasor=A.GOZ_BAKISI_KLASORU, dosya_on_eki="bakis",
)

print("Kameraya DUZ bakip 'c' ile kalibre et. 'r' = kesit al. 'v' = video kaydi ac/kapat. Cikis: 'q'.")

ilk_kare_mi = True

while True:
    ok, kare = cap.read()
    if not ok:
        break

    # Yakinlastirma en basta uygulanir - sonrasindaki HER SEY (tespit,
    # cizim, kayit) zaten yakinlastirilmis kare uzerinden calisir.
    kare = G.dijital_yakinlastir(kare, A.DIJITAL_YAKINLASTIRMA)

    if ilk_kare_mi:
        _t6 = time.time()

    h, w = kare.shape[:2]
    yuz_bulundu_bu_kare = False

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

        if secilen_yuz_i is not None and landmarker_sonuc.face_blendshapes:
            skorlar = {b.category_name: b.score for b in landmarker_sonuc.face_blendshapes[secilen_yuz_i]}
            sol_kirpma = skorlar.get("eyeBlinkLeft", 0.0)
            sag_kirpma = skorlar.get("eyeBlinkRight", 0.0)
            kirpma_skoru = (sol_kirpma + sag_kirpma) / 2.0

            goz_kapali_simdi = kirpma_skoru > A.ESIK_BLINK
            if goz_kapali_simdi and not goz_kapali_onceki:
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
                    sayaclar[yatay] += 1
                    if A.BAKIS_OLAY_KESITI_AKTIF:
                        bakis_olay_kaydedici.olay_tetikle(yatay)
                onceki_yatay = yatay

                if dikey != "merkez" and dikey != onceki_dikey:
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

            # --- Kol sayaci: IKI tetikleyiciden HERHANGI BIRI olusunca (asagidan-
            # yukari GECIS anini yakalayip) sayaci artirir - bkz. dosya basindaki
            # docstring. Normalize y kuculdukce ekranda yukari demektir.
            if secilen_govde_i is not None:
                lm = pose_sonuc.pose_landmarks[secilen_govde_i]

                sol_omuz = lm[PoseLandmark.LEFT_SHOULDER]
                sol_dirsek = lm[PoseLandmark.LEFT_ELBOW]
                sol_bilek = lm[PoseLandmark.LEFT_WRIST]
                if G.gorunur_mu(sol_bilek) and G.gorunur_mu(sol_omuz):
                    sol_kol_aktif = G.kol_aktif_mi(onceki_sol_kol_aktif, sol_omuz, sol_dirsek, sol_bilek)
                    if sol_kol_aktif and not onceki_sol_kol_aktif:
                        sayaclar["sol_kol"] += 1
                        if A.KOL_OLAY_KESITI_AKTIF:
                            sol_kol_olay_kaydedici.olay_tetikle("sol_kol")
                    onceki_sol_kol_aktif = sol_kol_aktif

                sag_omuz = lm[PoseLandmark.RIGHT_SHOULDER]
                sag_dirsek = lm[PoseLandmark.RIGHT_ELBOW]
                sag_bilek = lm[PoseLandmark.RIGHT_WRIST]
                if G.gorunur_mu(sag_bilek) and G.gorunur_mu(sag_omuz):
                    sag_kol_aktif = G.kol_aktif_mi(onceki_sag_kol_aktif, sag_omuz, sag_dirsek, sag_bilek)
                    if sag_kol_aktif and not onceki_sag_kol_aktif:
                        sayaclar["sag_kol"] += 1
                        if A.KOL_OLAY_KESITI_AKTIF:
                            sag_kol_olay_kaydedici.olay_tetikle("sag_kol")
                    onceki_sag_kol_aktif = sag_kol_aktif

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
    cv2.putText(kare, "c: kalibre et  |  r: kesit al  |  v: video kaydi  |  q: cikis",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if kaydedici.kayit_yapiliyor:
        cv2.circle(kare, (w - 30, 30), 8, (0, 0, 255), -1)
        cv2.putText(kare, "REC", (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Video kaydi - TUM overlay'ler cizildikten SONRA bellege ekleniyor.
    kaydedici.kare_ekle(kare)
    # Olay kesitleri - HER karede, SADECE aktif olan kategoriler icin cagrilir.
    if A.KOL_OLAY_KESITI_AKTIF:
        sol_kol_olay_kaydedici.kare_ekle(kare)
        sag_kol_olay_kaydedici.kare_ekle(kare)
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
kirpma_olay_kaydedici.bitir()
bakis_olay_kaydedici.bitir()
print("Sayaclar:", sayaclar)