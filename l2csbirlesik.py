"""L2CS-Net + MediaPipe: TEK webcam'de bakis yonu + goz kirpma + el/kol/vucut
iskeleti - HEPSI AYNI ANDA, AYNI PENCEREDE.

BU DOSYA, ana OpenVINO surumune (gaze_birlesik.py) ALTERNATIF, YEDEK bir
bakis motoru - L2CS-Net (RetinaFace+ResNet50, Gaze360 ile egitilmis).
Neden ayri/yedek: L2CS kendi yuz dedektorunu (RetinaFace) calistirdigi icin
kadraja ikinci bir kisi/el girdiginde OpenVINO surumune gore daha yavas/
donmaya yatkin oluyordu (bkz. proje gecmisi) - ama Gaze360 ResNet50'si daha
"olgun"/test edilmis bir model oldugu icin referans/karsilastirma amacli
veya OpenVINO modeli yetersiz kaldiginda geri donmek icin burada HAZIR
tutuluyor. Calistirmak icin: `python l2cs_birlesik.py`.

KENDINE YETERLI (self-contained) dosya: bu dosyaya OZEL ayarlar (agirlik
yolu, esikler, kenar mesafeleri) asagida BU DOSYANIN ICINDE tanimli -
ayarlar.py'deki OpenVINO'ya ozel degerlerden (BAKIS_ASAGI_SIZINTI_K,
ESIK_BAKIS_XY, GOZ_KIRPINTI_MARJI vb.) BAGIMSIZDIR, boylece OpenVINO
tarafinda yapilan ince ayarlar bunu bozmaz/etkilemez. Kamera, kol/kirpma
esikleri, olay kesiti sureleri, klasor yollari gibi BAKISTAN BAGIMSIZ
ayarlar yine ORTAK ayarlar.py'den (A.) gelir.

ONCE agirlik dosyasi gerekli: L2CSNet_gaze360.pkl - GitHub'daki L2CS-Net
sayfasindan (https://github.com/Ahmednull/L2CS-Net) indirip BU klasore koy.
Ayrica `pip install torch l2cs` gerekir (OpenVINO surumunden FARKLI
bagimliliklar - ikisini ayni anda kurman gerekebilir). MediaPipe model
dosyalari (face_landmarker.task, pose_landmarker_lite.task,
hand_landmarker.task) ilk calistirmada otomatik indirilir (gaze_birlesik.py
ile ORTAK, zaten inmisse tekrar inmez).

PERFORMANS UYARISI: Tek karede DORT model (RetinaFace+L2CS, FaceLandmarker,
PoseLandmarker, HandLandmarker) calisiyor, bu yuzden FPS bunlari ayri ayri
calistirmaktan dusuk olabilir; kadraja ikinci bir yuz girerse (RetinaFace
onu da islemeye calisir) yavaslama/donma gorulebilir. Yavas gelirse
ayarlar.py'deki AKTIF_POSE / AKTIF_EL bayraklarini False yap.

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

# --- Bu dosyaya OZEL L2CS ayarlari (ortak ayarlar.py'den BAGIMSIZ) --------
AKTIF_GAZE_L2CS = True  # False yaparsan L2CS hic yuklenmez, sadece kirpma/kol/govde calisir

L2CS_AGIRLIK = A.BURASI / "L2CSNet_gaze360.pkl"
L2CS_ESIK_ACI = 0.08              # radyan, ~4.5 derece - duz bakis "olu bolgesi"
L2CS_MAKS_ACI_SICRAMA = 0.35      # radyan, ~20 derece - karede izin verilen maks pitch/yaw degisimi
L2CS_KENAR_MESAFE_YATAY = 0.50    # SOL/SAG icin merkezden mesafe (yuz genisligi carpani)
L2CS_KENAR_MESAFE_UST = 0.20      # YUKARI icin merkezden mesafe
L2CS_KENAR_MESAFE_ALT = 0.40      # ASAGI icin merkezden mesafe


def _l2cs_pipeline_yukle():
    """L2CS-Net (ResNet50 + RetinaFace) pipeline'ini yukler.

    NOT: torch/l2cs import'lari BILEREK burada, fonksiyon icinde (lazy) -
    AKTIF_GAZE_L2CS=False iken bu fonksiyon hic cagrilmiyor, yani bu agir
    import'larin suresi de HIC harcanmiyor.
    """
    import torch
    from l2cs import Pipeline

    if not L2CS_AGIRLIK.exists():
        raise SystemExit(
            f"Once L2CSNet_gaze360.pkl dosyasini indirip {A.BURASI} icine koy "
            "(l2cs_birlesik.py docstring'ine bak)."
        )

    _t0 = time.time()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[zaman] torch/cuda hazir: {time.time() - _t0:.1f}s")

    _t1 = time.time()
    gaze = Pipeline(weights=str(L2CS_AGIRLIK), arch="ResNet50", device=device)
    print(f"[zaman] L2CS Pipeline (ResNet50 + RetinaFace) yuklendi: {time.time() - _t1:.1f}s")
    return gaze


# --- Modelleri yukle ----------------------------------------------------
if AKTIF_GAZE_L2CS:
    gaze = _l2cs_pipeline_yukle()
else:
    gaze = None
    print("[bilgi] AKTIF_GAZE_L2CS=False - L2CS-Net YUKLENMEDI. Bakis yonu (SOL/SAG/YUKARI/ASAGI) sayaclari ve yuz oku calismayacak.")

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

# Kimlik kilidi durumu - SADECE MediaPipe tarafini (kirpma + kol) kapsar.
# L2CS'in KENDI bakis tespiti (RetinaFace, asagidaki gaze.step() cagrisi)
# BAGIMSIZ bir dedektor oldugu icin bu kilidin disinda kalir - kadrajda
# ikinci bir kisi varsa L2CS'in bakis kismi hala "ilk bulunan yuz"
# mantigiyla calisir (bkz. ayarlar.py KIMLIK_KILIDI_AKTIF notu).
kilitli_yuz_merkez = None
yuz_kayip_kare = 0
kilitli_govde_merkez = None
govde_kayip_kare = 0

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

    # Yakinlastirma en basta uygulanir - sonrasindaki HER SEY (tespit,
    # cizim, kayit) zaten yakinlastirilmis kare uzerinden calisir.
    kare = G.dijital_yakinlastir(kare, A.DIJITAL_YAKINLASTIRMA)

    if ilk_kare_mi:
        _t6 = time.time()

    h, w = kare.shape[:2]
    yuz_bulundu_bu_kare = False

    # --- L2CS-Net: bakis yonu ------------------------------------------------
    if AKTIF_GAZE_L2CS and gaze is not None:
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
                # icin ayrica L2CS_MAKS_ACI_SICRAMA ile tek karelik "cilginca"
                # outlier sicramalar da kirpiliyor (bkz. gorsellik.yumusat).
                pitch = yumusak_pitch = G.yumusat(yumusak_pitch, pitch, L2CS_MAKS_ACI_SICRAMA)
                yaw = yumusak_yaw = G.yumusat(yumusak_yaw, yaw, L2CS_MAKS_ACI_SICRAMA)
                merkez_x = yumusak_merkez_x = G.yumusat(yumusak_merkez_x, merkez_x)
                merkez_y = yumusak_merkez_y = G.yumusat(yumusak_merkez_y, merkez_y)
                uzunluk = yumusak_uzunluk = G.yumusat(yumusak_uzunluk, uzunluk)

                # Kenar "kutusu" YUZE GORE - yuz kutusunun genisligiyle olceklenir
                # ve merkezi her kare yuzun merkezine tasinir, yani sen hareket
                # ettikce kutu da SENINLE birlikte kayar.
                cizgi_sol_x = int(merkez_x - uzunluk * L2CS_KENAR_MESAFE_YATAY)
                cizgi_sag_x = int(merkez_x + uzunluk * L2CS_KENAR_MESAFE_YATAY)
                cizgi_ust_y = int(merkez_y - uzunluk * L2CS_KENAR_MESAFE_UST)
                cizgi_alt_y = int(merkez_y + uzunluk * L2CS_KENAR_MESAFE_ALT)

                dx = -uzunluk * np.sin(pitch) * np.cos(yaw)
                dy = -uzunluk * np.sin(yaw)
                ucur_x = merkez_x + dx
                ucur_y = merkez_y + dy

                duz_bakiyor = abs(pitch) < L2CS_ESIK_ACI and abs(yaw) < L2CS_ESIK_ACI

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
        except Exception:
            pass

    # --- MediaPipe: kirpma + govde + eller (AYNI kare, AYNI mp_image) -------
    try:
        rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        kare_zaman_damgasi_ms += 33

        landmarker_sonuc = face_landmarker.detect_for_video(mp_image, kare_zaman_damgasi_ms)

        # --- Kimlik kilidi: adaylar arasindan "kilitli kisiyi" sec (SADECE
        # kirpma/MediaPipe tarafi icin - L2CS'in kendi bakis tespiti bunun
        # disinda, bkz. dosya basindaki durum degiskeni notu).
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
                if G.gorunur_mu(sol_bilek) and G.gorunur_mu(sol_omuz):
                    sol_kol_aktif = G.kol_aktif_mi(onceki_sol_kol_aktif, sol_omuz, sol_dirsek, sol_bilek)
                    if sol_kol_aktif and not onceki_sol_kol_aktif:
                        sayaclar["sol_kol"] += 1
                        if A.KOL_OLAY_KESITI_AKTIF:
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
                        if A.KOL_OLAY_KESITI_AKTIF:
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
    # Olay kesitleri - HER karede, SADECE aktif olan kategoriler icin cagrilir
    # (onbellegi surekli guncel tutar, olay olsun olmasin).
    if A.KOL_OLAY_KESITI_AKTIF:
        sol_kol_olay_kaydedici.kare_ekle(kare)
        sag_kol_olay_kaydedici.kare_ekle(kare)
    if A.KIRPMA_OLAY_KESITI_AKTIF:
        kirpma_olay_kaydedici.kare_ekle(kare)
    if A.BAKIS_OLAY_KESITI_AKTIF:
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