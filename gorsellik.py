"""Cizim (govde/el iskeleti) ve geometri yardimcilari (dirsek acisi,
gorunurluk kontrolu, EMA yumusatma, OpenVINO gaze modeli icin goz kirpintisi
cikarma / head-pose hesabi).

Cizim icin mediapipe.tasks.python.vision.drawing_utils / drawing_styles
kullanilir - bu, GUNCEL mediapipe pip paketiyle (0.10.32+, 1.0.0) birlikte
gelen, Tasks API'nin KENDI resmi cizim modulu.
"""
import math

import cv2
import mediapipe as mp
import numpy as np

import ayarlar as A

PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
mp_cizim = mp.tasks.vision.drawing_utils
mp_stil = mp.tasks.vision.drawing_styles


def govde_ciz(kare, pose_sonuc):
    for pose_landmarks in pose_sonuc.pose_landmarks:
        mp_cizim.draw_landmarks(
            kare, pose_landmarks, PoseLandmarksConnections.POSE_LANDMARKS,
            mp_stil.get_default_pose_landmarks_style(),
        )


def eller_ciz(kare, hand_sonuc):
    for hand_landmarks in hand_sonuc.hand_landmarks:
        mp_cizim.draw_landmarks(
            kare, hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS,
            mp_stil.get_default_hand_landmarks_style(),
            mp_stil.get_default_hand_connections_style(),
        )


def gorunur_mu(nokta):
    """Bir pose landmark'in visibility skoru ayarlar.GORUNURLUK_ESIK'in
    ustunde mi?"""
    return nokta.visibility is not None and nokta.visibility >= A.GORUNURLUK_ESIK


def dirsek_acisi_derece(omuz, dirsek, bilek):
    """Omuz-dirsek-bilek uc noktasindan dirsekteki acinin (derece) hesabi."""
    v1x, v1y = omuz.x - dirsek.x, omuz.y - dirsek.y
    v2x, v2y = bilek.x - dirsek.x, bilek.y - dirsek.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_aci = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cos_aci))


def kol_aktif_mi(onceki_aktif, omuz, dirsek, bilek):
    """Kol "kalkik" ya da "kivrik" mi - HISTEREZISLI (Schmitt trigger)."""
    if onceki_aktif:
        y_esik = A.KOL_Y_ESIK + A.KOL_HISTEREZIS_Y
        aci_esik = A.DIRSEK_ACI_ESIK + A.KOL_HISTEREZIS_ACI
        kalkik = bilek.y < omuz.y + A.KOL_HISTEREZIS_Y
    else:
        y_esik = A.KOL_Y_ESIK
        aci_esik = A.DIRSEK_ACI_ESIK
        kalkik = bilek.y < omuz.y - A.KOL_HISTEREZIS_Y

    kivrik = (
        gorunur_mu(dirsek)
        and abs(bilek.y - omuz.y) < y_esik
        and dirsek_acisi_derece(omuz, dirsek, bilek) < aci_esik
    )
    return kalkik or kivrik


def omuz_genisligi_piksel(sol_omuz, sag_omuz, w, h):
    """Iki omuz landmark'i arasindaki piksel mesafesi - kimlik kilidinde
    govdenin "buyuklugu" (sicrama esigini olceklemek) icin kullanilir."""
    dx = (sol_omuz.x - sag_omuz.x) * w
    dy = (sol_omuz.y - sag_omuz.y) * h
    return math.hypot(dx, dy)


def dijital_yakinlastir(kare, oran):
    """Kareyi TAM ORTASINDAN kirpip eski boyutuna geri buyutur (bkz.
    ayarlar.DIJITAL_YAKINLASTIRMA). oran<=1.0 ise kareyi OLDUGU GIBI
    dondurur (kopyasiz, maliyetsiz)."""
    if oran <= 1.0:
        return kare
    h, w = kare.shape[:2]
    kirp_w = max(1, int(w / oran))
    kirp_h = max(1, int(h / oran))
    x1 = (w - kirp_w) // 2
    y1 = (h - kirp_h) // 2
    kirpilmis = kare[y1:y1 + kirp_h, x1:x1 + kirp_w]
    return cv2.resize(kirpilmis, (w, h), interpolation=cv2.INTER_CUBIC)


def yumusat(onceki, yeni, maks_sicrama=None, oran=None):
    """Ustel hareketli ortalama (EMA) - onceki=None ise dogrudan yeni'yi dondurur.

    maks_sicrama verilirse, EMA'ya girmeden ONCE tek karedeki degisim bu
    deger ile SINIRLANIR - ani "sicrama" (outlier) degerlerin sonucu bir
    anda yanlis yone firlatmasini engeller.
    """
    if onceki is None:
        return yeni
    if maks_sicrama is not None:
        fark = yeni - onceki
        if fark > maks_sicrama:
            yeni = onceki + maks_sicrama
        elif fark < -maks_sicrama:
            yeni = onceki - maks_sicrama
    oran = A.YUMUSATMA_ORANI if oran is None else oran
    return oran * yeni + (1 - oran) * onceki


# --- OpenVINO gaze-estimation-adas-0002 icin yardimcilar --------------------
# Landmark indeksleri: MediaPipe 478 noktalik yuz mesh'inde goz disi/ici/ust/
# alt koseleri. SAG_GOZ_IDX kisinin GERCEK sag gozu (kameraya bakan goruntude
# SOL tarafta), SOL_GOZ_IDX kisinin GERCEK sol gozudur.
SAG_GOZ_IDX = [33, 133, 159, 145]   # disi, ici, ust, alt
SOL_GOZ_IDX = [263, 362, 386, 374]  # disi, ici, ust, alt


def goz_kutusu(landmarks, idxs, w, h, marj=None):
    marj = A.GOZ_KIRPINTI_MARJI if marj is None else marj
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * marj
    size = max(size, 20)  # kutu asla cok kucuk/sifir olmasin
    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = int(cx + size / 2), int(cy + size / 2)
    return max(x1, 0), max(y1, 0), x2, y2


def donus_matrisinden_aci(rot_3x3):
    """3x3 donus matrisinden yaw, pitch, roll (derece) hesabi."""
    sy = math.sqrt(rot_3x3[0, 0] ** 2 + rot_3x3[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rot_3x3[2, 1], rot_3x3[2, 2])
        pitch = math.atan2(-rot_3x3[2, 0], sy)
        yaw = math.atan2(rot_3x3[1, 0], rot_3x3[0, 0])
    else:
        roll = math.atan2(-rot_3x3[1, 2], rot_3x3[1, 1])
        pitch = math.atan2(-rot_3x3[2, 0], sy)
        yaw = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def kirpinti_dondur(img, aci_derece):
    """Goz kirpintisini kendi merkezi etrafinda dondurur (roll telafisi)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), aci_derece, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def kirpinti_hazirla(img):
    """OpenVINO gaze modeli icin goz kirpintisini 60x60 NCHW float32'e cevirir."""
    img = cv2.resize(img, (60, 60)).astype(np.float32)
    return img.transpose(2, 0, 1)[np.newaxis, ...]


def yuz_bbox_hesapla(landmarks, w, h):
    """478 landmark'in tumunden yuz sinirlayici kutusunu (piksel) hesaplar -
    L2CS surumundeki RetinaFace bbox'inin yerini tutar (kenar kutusu / ok
    olcegi icin)."""
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


# --- Kimlik kilidi (tek kisiye odaklanma) -----------------------------------
def kilitli_aday_sec(kilitli_merkez, kayip_kare_sayaci, merkezler, buyuklukler,
                      maks_sicrama_orani, kayip_kare_limiti):
    """Bir onceki karede kilitlenen kisiye EN YAKIN adayi secer.

    - kilitli_merkez=None ise (henuz kilit yok VEYA kilit uzun sure kayip):
      EN BUYUK adayi kilitler - rastgele bir davetsiz misafire kilitlenmeyi
      onlemek icin (genelde en yakin/on plandaki kisi kadrajda en buyuk
      gorunur).
    - kilitli_merkez varsa: merkeze EN YAKIN adayi bulur, ama sicrama
      kisinin KENDI buyuklugune (yuz genisligi / omuz genisligi) gore
      SINIRLIDIR - cok uzaktaki bir aday (baska bir kisi) REDDEDILIR.
    - Hicbir aday esik icinde degilse VEYA hic aday yoksa (kilitli kisi bu
      karede gorunmuyor - kafasini cevirmis, el kapatmis vb.) secim None
      doner ama kilit HEMEN birakilmaz - kayip_kare_limiti kadar "sabir"
      gosterilir, o kadar kare boyunca hic eslesme olmazsa kilit tamamen
      birakilir (bir sonraki karede en buyuk adaya yeniden kilitlenir).

    Donus: (secilen_indeks_veya_None, yeni_kilitli_merkez, yeni_kayip_kare_sayaci)
    """
    if not merkezler:
        kayip_kare_sayaci += 1
        if kayip_kare_sayaci > kayip_kare_limiti:
            return None, None, 0
        return None, kilitli_merkez, kayip_kare_sayaci

    if kilitli_merkez is None:
        en_buyuk_i = max(range(len(buyuklukler)), key=lambda i: buyuklukler[i])
        return en_buyuk_i, merkezler[en_buyuk_i], 0

    en_yakin_i = None
    en_yakin_mesafe = None
    for i, m in enumerate(merkezler):
        d = math.hypot(m[0] - kilitli_merkez[0], m[1] - kilitli_merkez[1])
        if en_yakin_mesafe is None or d < en_yakin_mesafe:
            en_yakin_mesafe = d
            en_yakin_i = i

    esik = buyuklukler[en_yakin_i] * maks_sicrama_orani
    if en_yakin_mesafe <= esik:
        return en_yakin_i, merkezler[en_yakin_i], 0

    kayip_kare_sayaci += 1
    if kayip_kare_sayaci > kayip_kare_limiti:
        return None, None, 0
    return None, kilitli_merkez, kayip_kare_sayaci