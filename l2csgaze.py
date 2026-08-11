"""L2CS-Net ile CANLI bakis yonu takibi + kenar sayaclari + goz kirpma sayaci.

ONCE agirlik dosyasi gerekli: L2CSNet_gaze360.pkl -
https://github.com/Ahmednull/L2CS-Net adresindeki Google Drive linkinden
indirip BU klasore (04-gaze/) koy.

Goz kirpma tespiti icin MediaPipe'in YENI Tasks API'si (FaceLandmarker +
blendshape) kullanilir. Eski "mp.solutions.face_mesh" API'si son mediapipe
surumlerinde bozuk oldugu icin ARTIK KULLANILMIYOR - bu yuzden mediapipe'i
eski bir surume sabitlemene gerek YOK, en guncel surumu kurabilirsin:
    pip install --upgrade mediapipe
face_landmarker.task model dosyasi ilk calistirmada BU klasore otomatik
indirilir (internet gerekir, sadece ilk seferde).

Calistirinca webcam ACILIR. Yuz kutusu her zaman cizilir. Bakis oku SADECE
gozler/kafa ekrana duz bakmanin disinda bir yone kaydiginda (esik acisini
gectiginde) gorunur - duz ekrana bakarken ok gizlenir. Ekranin ortasina
yakin 4 cizgi vardir (SOL/SAG/YUKARI/ASAGI). Ok ucu o cizgiyi gecince
ilgili sayac bir artar. Goz kapanip acildiginda KIRPMA sayaci artar
(MediaPipe FaceLandmarker'in eyeBlinkLeft/eyeBlinkRight blendshape
skorlari kullanilir). Sayaclar ekranda canli gosterilir.

KESIT: 'r' tusuna basildiginda o anki kare (ekranda gordugun TUM
overlay'lerle birlikte) "kesitler/" klasorune JPEG olarak kaydedilir,
KESIT sayaci artar.

VIDEO: 'v' tusuna basildiginda kayit BASLAR (ekranda kirmizi "REC"
gorunur), tekrar 'v'ye basinca DURUR ve dosya "videolar/" klasorune
MP4 olarak kaydedilir (ekranda gordugun TUM overlay'lerle birlikte).
Kareler once bellekte biriktirilir, kayit durunca gercek gecen sureden
FPS hesaplanip OYLE yazilir - boylece video hizli/sok gorunmez, gercek
zamanla ayni surer.

KALIBRASYON: L2CS-Net'in pitch/yaw ciktisi kameranin goz hizasinda
olmamasi / yuz kadraji gibi nedenlerle sabit bir sapma (bias) icerebilir -
yani ekrana DUZ bakarken bile ok bir yone kaymis gorunebilir. Kameraya
DUZ bakarken 'c' tusuna bas: o anki aci "sifir" kabul edilir ve sapma
duzeltilir. Cikis: 'q'. GPU'da akici calisir.

GKS/iletisim: "hasta kameraya/nesneye bakiyor ve takip ediyor mu" (E4).
Test onerisi: once 'c' ile kalibre et, sonra ekranin 4 kosesine sirayla
bak, okun ilgili kenar cizgisini gectigini ve sayacin bir kez arttigini
dogrula. Birkac kez kirpip KIRPMA sayacinin arttigini, 'r' ile de KESIT
sayacinin arttigini ve dosyanin kaydedildigini dogrula.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from l2cs import Pipeline

BURASI = Path(__file__).resolve().parent
AGIRLIK = BURASI / "L2CSNet_gaze360.pkl"
if not AGIRLIK.exists():
    raise SystemExit(f"Once L2CSNet_gaze360.pkl dosyasini indirip {BURASI} icine koy (docstring'e bak).")

_t0 = time.time()
# cuDNN "benchmark" modu ACIKKEN, her yeni girdi boyutu icin PyTorch en hizli
# algoritmayi ARAR - bu arama ilk cagrida (RetinaFace'in cok-olcekli
# konvolusyonlarinda) 20-40 saniye surebiliyor. Kapatinca ilk kare cok daha
# hizli acilir, kare-basi hiz farki bu kullanim icin gozle gorulmez.
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[zaman] torch/cuda hazir: {time.time() - _t0:.1f}s")

_t1 = time.time()
gaze = Pipeline(weights=str(AGIRLIK), arch="ResNet50", device=device)
print(f"[zaman] L2CS Pipeline (ResNet50 + RetinaFace) yuklendi: {time.time() - _t1:.1f}s")

_t2 = time.time()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam acilamadi. Baska uygulama kullaniyor olabilir.")
print(f"[zaman] webcam acildi: {time.time() - _t2:.1f}s")

# --- Kenar / sayac ayarlari ----------------------------------------------
# Ok kisa oldugu icin (yuz kutusu genisligi kadar) cizgileri kareye
# yakin degil, MERKEZE yakin tut ki ok gercekten erisebilsin.
KENAR_ORANI = 0.30       # SOL/SAG/ASAGI cizgileri icin (kare boyutunun bu orani kadar icerde)
KENAR_ORANI_UST = 0.42   # YUKARI cizgisi icin - daha buyuk = cizgi daha ASAGIDA (merkeze yakin)

# Duz ekrana bakarken ok gizlensin diye olculen "olu bolge" - radyan.
# Kucultursen daha az kipirdanmada bile ok belirir, buyutursen daha
# belirgin bakis degisiminde belirir.
ESIK_ACI = 0.08  # ~4.5 derece

sayaclar = {"sag": 0, "sol": 0, "yukari": 0, "asagi": 0, "kirpma": 0, "kesit": 0}
onceki_yatay = "merkez"
onceki_dikey = "merkez"

# --- Kalibrasyon (sabit sapma / bias duzeltmesi) --------------------------
# 'c' tusuna basildiginda o anki ham pitch/yaw buraya yazilir ve sonraki
# tum hesaplamalardan cikarilir.
BIAS_PITCH = 0.0
BIAS_YAW = 0.0
son_pitch_ham = 0.0
son_yaw_ham = 0.0
yuz_bulundu_bu_kare = False

# --- Goz kirpma (blink) ayarlari - MediaPipe Tasks / FaceLandmarker -------
FACE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_TASK_YOLU = BURASI / "face_landmarker.task"
if not FACE_TASK_YOLU.exists():
    print("face_landmarker.task indiriliyor (bir kereye mahsus, internet gerekir)...")
    urllib.request.urlretrieve(FACE_TASK_URL, FACE_TASK_YOLU)

ESIK_BLINK = 0.5  # eyeBlinkLeft/Right skoru (0-1) bu deger USTUNDE ise goz "kapali" sayilir

# --- Kesit (ekran goruntusu) klasoru --------------------------------------
KESIT_KLASORU = BURASI / "kesitler"
KESIT_KLASORU.mkdir(exist_ok=True)

# --- Video kaydi ayarlari --------------------------------------------------
# ONEMLI: cap.get(cv2.CAP_PROP_FPS) veya sabit bir FPS varsayimi kullanip
# VideoWriter'i o hizla yazdirmak, gercek isleme hizindan (L2CS + MediaPipe
# yuzunden genelde daha DUSUK) YUKSEK cikinca video hizli/sok gorunuyordu.
# Bunun yerine kareleri BELLEKTE biriktirip, kayit DURUNCA gercek gecen
# sureden (kare sayisi / gecen saniye) dogru FPS'i hesaplayip OYLE yaziyoruz.
VIDEO_KLASORU = BURASI / "videolar"
VIDEO_KLASORU.mkdir(exist_ok=True)
VIDEO_FPS_VARSAYILAN = 15.0  # sure olcumu guvenilmezse (cok kisa kayit) kullanilir
kayit_yapiliyor = False
kayit_karesi_listesi = []
kayit_baslangic_zamani = None


def video_kaydini_bitir():
    global kayit_yapiliyor, kayit_karesi_listesi, kayit_baslangic_zamani
    if not kayit_karesi_listesi:
        kayit_yapiliyor = False
        kayit_baslangic_zamani = None
        return

    gecen_sure = time.time() - kayit_baslangic_zamani
    kare_sayisi = len(kayit_karesi_listesi)
    if gecen_sure > 0.5:
        gercek_fps = kare_sayisi / gecen_sure
    else:
        gercek_fps = VIDEO_FPS_VARSAYILAN
    gercek_fps = max(1.0, min(gercek_fps, 30.0))  # makul araliga sikistir

    zaman_etiketi = time.strftime("%Y%m%d_%H%M%S")
    video_dosya_adi = VIDEO_KLASORU / f"video_{zaman_etiketi}.mp4"
    yukseklik, genislik = kayit_karesi_listesi[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    yazici = cv2.VideoWriter(str(video_dosya_adi), fourcc, gercek_fps, (genislik, yukseklik))
    for k in kayit_karesi_listesi:
        yazici.write(k)
    yazici.release()

    print(f"Video kaydedildi: {video_dosya_adi} ({kare_sayisi} kare, {gecen_sure:.1f}s, {gercek_fps:.1f} fps ile yazildi)")
    kayit_karesi_listesi = []
    kayit_yapiliyor = False
    kayit_baslangic_zamani = None

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_t3 = time.time()
face_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_TASK_YOLU)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
)
print(f"[zaman] FaceLandmarker yuklendi: {time.time() - _t3:.1f}s")

goz_kapali_onceki = False
kare_zaman_damgasi_ms = 0  # detect_for_video icin artan sahte zaman damgasi

print("Kameraya DUZ bakip 'c' tusuna basarak kalibrasyon yap. 'r' = kesit al. 'v' = video kaydi ac/kapat. Cikis: 'q'.")

ilk_kare_mi = True

while True:
    ok, kare = cap.read()
    if not ok:
        break

    if ilk_kare_mi:
        _t4 = time.time()

    h, w = kare.shape[:2]
    cizgi_sol_x = int(w * KENAR_ORANI)
    cizgi_sag_x = int(w * (1 - KENAR_ORANI))
    cizgi_ust_y = int(h * KENAR_ORANI_UST)
    cizgi_alt_y = int(h * (1 - KENAR_ORANI))

    yuz_bulundu_bu_kare = False

    try:
        if ilk_kare_mi:
            _t_gaze = time.time()
        sonuc = gaze.step(kare)
        if ilk_kare_mi:
            print(f"[zaman]   -> gaze.step() (RetinaFace+L2CS ilk cagrisi): {time.time() - _t_gaze:.1f}s")

        if len(sonuc.pitch) > 0:
            son_pitch_ham = float(sonuc.pitch[0])
            son_yaw_ham = float(sonuc.yaw[0])
            yuz_bulundu_bu_kare = True

            pitch = son_pitch_ham - BIAS_PITCH
            yaw = son_yaw_ham - BIAS_YAW

            bbox = sonuc.bboxes[0]
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            cv2.rectangle(kare, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            merkez_x = (bbox[0] + bbox[2]) / 2.0
            merkez_y = (bbox[1] + bbox[3]) / 2.0
            uzunluk = bbox[2] - bbox[0]  # render()'in kullandigi ayni uzunluk (bbox genisligi)

            # l2cs'in kendi draw_gaze() fonksiyonuyla BIREBIR ayni formul
            # (orijinal kutuphane formulu).
            dx = -uzunluk * np.sin(pitch) * np.cos(yaw)
            dy = -uzunluk * np.sin(yaw)
            ucur_x = merkez_x + dx
            ucur_y = merkez_y + dy

            duz_bakiyor = abs(pitch) < ESIK_ACI and abs(yaw) < ESIK_ACI

            if not duz_bakiyor:
                cv2.arrowedLine(
                    kare,
                    (int(merkez_x), int(merkez_y)),
                    (int(ucur_x), int(ucur_y)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18,
                )

            yatay = "sol" if ucur_x > cizgi_sag_x else "sag" if ucur_x < cizgi_sol_x else "merkez"
            dikey = "asagi" if ucur_y > cizgi_alt_y else "yukari" if ucur_y < cizgi_ust_y else "merkez"

            if yatay != "merkez" and yatay != onceki_yatay:
                sayaclar[yatay] += 1
            onceki_yatay = yatay

            if dikey != "merkez" and dikey != onceki_dikey:
                sayaclar[dikey] += 1
            onceki_dikey = dikey
    except Exception:
        pass  # karede yuz bulunamazsa ham goruntuyu goster

    # --- Goz kirpma tespiti (MediaPipe Tasks FaceLandmarker + blendshape) ---
    try:
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        kare_zaman_damgasi_ms += 33  # ~30 fps varsayimiyla artan sahte zaman damgasi
        if ilk_kare_mi:
            _t_land = time.time()
        landmarker_sonuc = face_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)
        if ilk_kare_mi:
            print(f"[zaman]   -> face_landmarker.detect_for_video() ilk cagrisi: {time.time() - _t_land:.1f}s")

        if landmarker_sonuc.face_blendshapes:
            skorlar = {b.category_name: b.score for b in landmarker_sonuc.face_blendshapes[0]}
            sol_kirpma = skorlar.get("eyeBlinkLeft", 0.0)
            sag_kirpma = skorlar.get("eyeBlinkRight", 0.0)
            kirpma_skoru = (sol_kirpma + sag_kirpma) / 2.0

            goz_kapali_simdi = kirpma_skoru > ESIK_BLINK
            if goz_kapali_simdi and not goz_kapali_onceki:
                sayaclar["kirpma"] += 1
            goz_kapali_onceki = goz_kapali_simdi
    except Exception:
        pass

    # --- Kenar cizgileri + sayaclar ---
    renk = (255, 255, 0)
    cv2.line(kare, (cizgi_sol_x, 0), (cizgi_sol_x, h), renk, 1)
    cv2.line(kare, (cizgi_sag_x, 0), (cizgi_sag_x, h), renk, 1)
    cv2.line(kare, (0, cizgi_ust_y), (w, cizgi_ust_y), renk, 1)
    cv2.line(kare, (0, cizgi_alt_y), (w, cizgi_alt_y), renk, 1)

    # Tum sayaclar ekranin en ustunde, KIRPMA'nin yaninda/altinda toplu gosterilir.
    cv2.putText(kare, f"KIRPMA: {sayaclar['kirpma']}   KESIT: {sayaclar['kesit']}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(kare, f"SOL: {sayaclar['sol']}  SAG: {sayaclar['sag']}  YUKARI: {sayaclar['yukari']}  ASAGI: {sayaclar['asagi']}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
    cv2.putText(kare, "c: kalibre et (duz bak)  |  r: kesit al  |  v: video kaydi", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if kayit_yapiliyor:
        cv2.circle(kare, (w - 30, 30), 8, (0, 0, 255), -1)
        cv2.putText(kare, "REC", (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Video kaydi - TUM overlay'ler cizildikten SONRA bellege ekleniyor ki
    # dosyada da ekranda gordugun her sey olsun. Gercek dosyaya yazma islemi
    # kayit DURUNCA (dogru FPS hesaplanip) yapilir - bkz. video_kaydini_bitir().
    if kayit_yapiliyor:
        kayit_karesi_listesi.append(kare.copy())

    if ilk_kare_mi:
        print(f"[zaman] ilk kare islendi (CUDA/RetinaFace ilk 'isinma' dahil): {time.time() - _t4:.1f}s")
        ilk_kare_mi = False

    cv2.imshow("L2CS bakis takibi (q = cik)", kare)
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        if kayit_yapiliyor:
            video_kaydini_bitir()
        break
    if tus == ord("c") and yuz_bulundu_bu_kare:
        BIAS_PITCH = son_pitch_ham
        BIAS_YAW = son_yaw_ham
        print(f"Kalibre edildi. BIAS_PITCH={BIAS_PITCH:.3f} BIAS_YAW={BIAS_YAW:.3f}")
    if tus == ord("r"):
        # 'kare' burada TUM overlay'lerle (kutu, ok, cizgiler, sayaclar)
        # birlikte - yani JPEG'de ekranda gordugun her sey de olacak.
        zaman_etiketi = time.strftime("%Y%m%d_%H%M%S")
        kesit_dosya_adi = KESIT_KLASORU / f"kesit_{zaman_etiketi}.jpg"
        cv2.imwrite(str(kesit_dosya_adi), kare)
        sayaclar["kesit"] += 1
        print(f"Kesit kaydedildi: {kesit_dosya_adi}")
    if tus == ord("v"):
        if not kayit_yapiliyor:
            kayit_yapiliyor = True
            kayit_karesi_listesi = []
            kayit_baslangic_zamani = time.time()
            print("Video kaydi basladi.")
        else:
            video_kaydini_bitir()

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
if kayit_yapiliyor:
    video_kaydini_bitir()
print("Sayaclar:", sayaclar)

# --- Alternatif: kamera yerine FOTOGRAF dosyasiyla deneme -------------------
# VERI = BURASI.parent / "veri"
# img = cv2.imread(str(VERI / "test_yuz.jpg"))
# sonuc = gaze.step(img)
# print("pitch:", sonuc.pitch, "yaw:", sonuc.yaw)