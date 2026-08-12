"""L2CS-Net + MediaPipe: TEK webcam'de bakis yonu + goz kirpma + el/kol/vucut
iskeleti - HEPSI AYNI ANDA, AYNI PENCEREDE.

BU DOSYA, calistirdigin TEK dosya (`python l2cs_birlesik.py`). Kod
okunabilirlik icin AYNI klasordeki kucuk modullere bolundu, ama hepsi bu
dosya uzerinden TEK PARCA halinde calisir:
  ayarlar.py   - TUM sabitler/esikler/acik-kapali bayraklar (davranisi
                 degistirmek istersen SADECE burayi duzenle)
  modeller.py  - L2CS-Net + MediaPipe model yukleme/indirme
  gorsellik.py - iskelet cizimi + geometri yardimcilari (dirsek acisi,
                 gorunurluk kontrolu, EMA yumusatma)
  kayit.py     - kesit (JPEG) / video (MP4) kaydi (VideoKaydedici sinifi)
  l2cs_birlesik.py (BU DOSYA) - hepsini birbirine baglayan ana webcam dongusu

TEK bir cv2.VideoCapture ile hem L2CS-Net (bakis) hem MediaPipe FaceLandmarker
(kirpma) hem MediaPipe PoseLandmarker (govde) hem MediaPipe HandLandmarker
(eller) calistirir, hepsinin sonucunu AYNI karenin uzerine cizer.

ONCE agirlik dosyasi gerekli: L2CSNet_gaze360.pkl - GitHub'daki L2CS-Net
sayfasindan indirip BU klasore koy. MediaPipe model dosyalari
(face_landmarker.task, pose_landmarker_lite.task, hand_landmarker.task)
ilk calistirmada otomatik indirilir.

PERFORMANS UYARISI: Tek karede DORT model (RetinaFace+L2CS, FaceLandmarker,
PoseLandmarker, HandLandmarker) calisiyor, bu yuzden FPS bunlari ayri ayri
calistirmaktan dusuk olabilir. Yavas gelirse ayarlar.py'deki AKTIF_POSE /
AKTIF_EL bayraklarini False yap.

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
Bir kategorinin penceresi ACIKKEN (henuz yazilmadan) O KATEGORIDE ikinci bir
olay olursa klip BITMEZ, penceresi o yeni olayin da +SONRA_SANIYE'sini
kapsayacak sekilde UZAR - yani ayni kategoride art arda gelen olaylar TEK,
daha UZUN bir klipte birlesir; hicbir olayin "sonrasi" kirpilmaz. Kategori
tamamen OLAY_SONRA_SANIYE kadar sakin kalinca o kategorinin klibi yazilir.
Kategoriler birbirinden BAGIMSIZDIR (orn. ayni anda sol kol kalkip goz
kirparsa, iki AYRI klip - biri sol_kol/'a biri goz_kirpma/'ya - yazilir).

Kontroller:
  c = kalibre et (kameraya duz bakarken bas, bakis sapmasini sifirlar)
  r = kesit al (o anki kare, TUM overlay'lerle, "kesitler/" klasorune JPEG)
  v = video kaydi ac/kapat ("videolar/" klasorune MP4, gercek FPS ile)
  q = cikis
"""
import time

import cv2
import mediapipe as mp
import numpy as np

import ayarlar as A
import gorsellik as G
import kayit as K
import modeller as M

# --- Modelleri yukle ----------------------------------------------------
gaze = M.gaze_pipeline_yukle()

_t2 = time.time()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam acilamadi. Baska uygulama kullaniyor olabilir.")
print(f"[zaman] webcam acildi: {time.time() - _t2:.1f}s")

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

    if ilk_kare_mi:
        _t6 = time.time()

    h, w = kare.shape[:2]
    yuz_bulundu_bu_kare = False

    # --- L2CS-Net: bakis yonu ------------------------------------------------
    try:
        sonuc = gaze.step(kare)
        if len(sonuc.pitch) > 0:
            son_pitch_ham = float(sonuc.pitch[0])
            son_yaw_ham = float(sonuc.yaw[0])
            yuz_bulundu_bu_kare = True

            pitch = son_pitch_ham - BIAS_PITCH
            yaw = son_yaw_ham - BIAS_YAW

            bbox = sonuc.bboxes[0]
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            if A.YUZ_CIZIMI_GOSTER:
                cv2.rectangle(kare, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            merkez_x = (bbox[0] + bbox[2]) / 2.0
            merkez_y = (bbox[1] + bbox[3]) / 2.0
            uzunluk = bbox[2] - bbox[0]

            # Yumusatma: ozellikle yuz kucukken (uzaktayken) pitch/yaw ve bbox
            # tahmini daha gurultulu oluyor - HAM degerler yerine yumusatilmis
            # (EMA) degerleri kullaniyoruz ki ok/kutu titremesin. pitch/yaw
            # icin ayrica MAKS_ACI_SICRAMA ile tek karelik "cilginca" outlier
            # sicramalar da kirpiliyor (bkz. gorsellik.yumusat).
            pitch = yumusak_pitch = G.yumusat(yumusak_pitch, pitch, A.MAKS_ACI_SICRAMA)
            yaw = yumusak_yaw = G.yumusat(yumusak_yaw, yaw, A.MAKS_ACI_SICRAMA)
            merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
            merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
            uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

            # Kenar "kutusu" YUZE GORE - yuz kutusunun genisligiyle olceklenir
            # ve merkezi her kare yuzun merkezine tasinir, yani sen hareket
            # ettikce kutu da SENINLE birlikte kayar.
            cizgi_sol_x = int(merkez_x - uzunluk * A.KENAR_MESAFE_YATAY)
            cizgi_sag_x = int(merkez_x + uzunluk * A.KENAR_MESAFE_YATAY)
            cizgi_ust_y = int(merkez_y - uzunluk * A.KENAR_MESAFE_UST)
            cizgi_alt_y = int(merkez_y + uzunluk * A.KENAR_MESAFE_ALT)

            dx = -uzunluk * np.sin(pitch) * np.cos(yaw)
            dy = -uzunluk * np.sin(yaw)
            ucur_x = merkez_x + dx
            ucur_y = merkez_y + dy

            duz_bakiyor = abs(pitch) < A.ESIK_ACI and abs(yaw) < A.ESIK_ACI

            if A.YUZ_CIZIMI_GOSTER and not duz_bakiyor:
                cv2.arrowedLine(
                    kare, (int(merkez_x), int(merkez_y)), (int(ucur_x), int(ucur_y)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18,
                )

            yatay = "sol" if ucur_x > cizgi_sag_x else "sag" if ucur_x < cizgi_sol_x else "merkez"
            dikey = "asagi" if ucur_y > cizgi_alt_y else "yukari" if ucur_y < cizgi_ust_y else "merkez"

            if yatay != "merkez" and yatay != onceki_yatay:
                sayaclar[yatay] += 1
                bakis_olay_kaydedici.olay_tetikle(yatay)
            onceki_yatay = yatay

            if dikey != "merkez" and dikey != onceki_dikey:
                sayaclar[dikey] += 1
                bakis_olay_kaydedici.olay_tetikle(dikey)
            onceki_dikey = dikey
    except Exception:
        pass

    # --- MediaPipe: kirpma + govde + eller (AYNI kare, AYNI mp_image) -------
    try:
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        kare_zaman_damgasi_ms += 33

        landmarker_sonuc = face_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
        if landmarker_sonuc.face_blendshapes:
            skorlar = {b.category_name: b.score for b in landmarker_sonuc.face_blendshapes[0]}
            sol_kirpma = skorlar.get("eyeBlinkLeft", 0.0)
            sag_kirpma = skorlar.get("eyeBlinkRight", 0.0)
            kirpma_skoru = (sol_kirpma + sag_kirpma) / 2.0

            goz_kapali_simdi = kirpma_skoru > A.ESIK_BLINK
            if goz_kapali_simdi and not goz_kapali_onceki:
                sayaclar["kirpma"] += 1
                kirpma_olay_kaydedici.olay_tetikle("kirpma")
            goz_kapali_onceki = goz_kapali_simdi

        if A.AKTIF_POSE and pose_landmarker is not None:
            pose_sonuc = pose_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
            if A.GOVDE_CIZIMI_GOSTER:
                G.govde_ciz(kare, pose_sonuc)

            # --- Kol sayaci: IKI tetikleyiciden HERHANGI BIRI olusunca (asagidan-
            # yukari GECIS anini yakalayip) sayaci artirir - bkz. dosya basindaki
            # docstring. Normalize y kuculdukce ekranda yukari demektir.
            if pose_sonuc.pose_landmarks:
                lm = pose_sonuc.pose_landmarks[0]

                sol_omuz = lm[PoseLandmark.LEFT_SHOULDER]
                sol_dirsek = lm[PoseLandmark.LEFT_ELBOW]
                sol_bilek = lm[PoseLandmark.LEFT_WRIST]
                if G.gorunur_mu(sol_bilek) and G.gorunur_mu(sol_omuz):
                    sol_kol_aktif = G.kol_aktif_mi(onceki_sol_kol_aktif, sol_omuz, sol_dirsek, sol_bilek)
                    if sol_kol_aktif and not onceki_sol_kol_aktif:
                        sayaclar["sol_kol"] += 1
                        sol_kol_olay_kaydedici.olay_tetikle("sol_kol")
                    onceki_sol_kol_aktif = sol_kol_aktif
                # else: bilek/omuz kadraj disinda/belirsiz - guvenilmez tahmini
                # SAYMA, onceki durumu da DEGISTIRME (gurultuden sayma).

                sag_omuz = lm[PoseLandmark.RIGHT_SHOULDER]
                sag_dirsek = lm[PoseLandmark.RIGHT_ELBOW]
                sag_bilek = lm[PoseLandmark.RIGHT_WRIST]
                if G.gorunur_mu(sag_bilek) and G.gorunur_mu(sag_omuz):
                    sag_kol_aktif = G.kol_aktif_mi(onceki_sag_kol_aktif, sag_omuz, sag_dirsek, sag_bilek)
                    if sag_kol_aktif and not onceki_sag_kol_aktif:
                        sayaclar["sag_kol"] += 1
                        sag_kol_olay_kaydedici.olay_tetikle("sag_kol")
                    onceki_sag_kol_aktif = sag_kol_aktif
                # else: bilek/omuz kadraj disinda/belirsiz - guvenilmez tahmini
                # SAYMA, onceki durumu da DEGISTIRME (gurultuden sayma).

        if A.AKTIF_EL and hand_landmarker is not None:
            hand_sonuc = hand_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
            if A.EL_CIZIMI_GOSTER:
                G.eller_ciz(kare, hand_sonuc)
    except Exception:
        pass

    # --- Yuze bagli kenar kutusu + sayaclar (en ustte, en sonda ciziliyor ki
    # okunsun). Kutu SADECE bu karede yuz bulunduysa cizilir (yuz_bulundu_bu_
    # kare) - bulunamazsa eski konumda "hayalet" kutu kalmasin diye gizlenir.
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
    # Olay kesitleri - HER karede, HER kategori icin cagrilir (onbellegi
    # surekli guncel tutar, olay olsun olmasin).
    sol_kol_olay_kaydedici.kare_ekle(kare)
    sag_kol_olay_kaydedici.kare_ekle(kare)
    kirpma_olay_kaydedici.kare_ekle(kare)
    bakis_olay_kaydedici.kare_ekle(kare)

    if ilk_kare_mi:
        print(f"[zaman] ilk kare islendi (4 modelin ilk 'isinmasi' dahil): {time.time() - _t6:.1f}s")
        ilk_kare_mi = False

    cv2.imshow("L2CS + MediaPipe: bakis + kirpma + govde + eller (q = cik)", kare)
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
        BIAS_PITCH = son_pitch_ham
        BIAS_YAW = son_yaw_ham
        print(f"Kalibre edildi. BIAS_PITCH={BIAS_PITCH:.3f} BIAS_YAW={BIAS_YAW:.3f}")
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