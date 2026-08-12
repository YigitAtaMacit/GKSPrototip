"""Cizim (govde/el iskeleti) ve geometri yardimcilari (dirsek acisi,
gorunurluk kontrolu, EMA yumusatma).

Cizim icin mediapipe.tasks.python.vision.drawing_utils / drawing_styles
kullanilir - bu, GUNCEL mediapipe pip paketiyle (0.10.32+, 1.0.0) birlikte
gelen, Tasks API'nin KENDI resmi cizim modulu (eski mp.solutions.drawing_utils
YERINE gecti, o artik pakette yok). Baglanti tablolari da yine Tasks API'nin
kendi PoseLandmarksConnections / HandLandmarksConnections siniflarindan
geliyor - elle hicbir baglanti/landmark tablosu tanimlanmiyor.
"""
import math

import mediapipe as mp

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
    ustunde mi? (Kadraj disinda kalan uzuvlar icin model yine de bir tahmin
    uretir ama guvenilmezdir - bu yuzden kol sayaclari bu kontrolu kullanir.)
    """
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
    """Kol "kalkik" ya da "kivrik" mi - HISTEREZISLI (Schmitt trigger).

    Kol henuz AKTIF DEGILKEN GIRMEK icin normal esikler kullanilir; kol
    AKTIFKEN o durumdan CIKMAK icin esikler GEVSETILIR (KOL_HISTEREZIS_*).
    Bu sayede sinirin tam ustunde/civarinda titreyen landmark'lar sayaci
    ard arda artirmaz - once belirgin sekilde "inmesi" gerekir.
    """
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


def yumusat(onceki, yeni, maks_sicrama=None):
    """Ustel hareketli ortalama (EMA) - onceki=None ise dogrudan yeni'yi dondurur.

    maks_sicrama verilirse (orn. pitch/yaw icin radyan cinsinden), EMA'ya
    girmeden ONCE tek karedeki degisim bu deger ile SINIRLANIR - yuz kucukken
    L2CS'in urettigi ani "sicrama" (outlier) degerlerin oku bir anda yanlis
    yone firlatmasini engeller. Boylece hem duzenli titreme (EMA ile) hem de
    tek kareli ani sapmalar (sicrama sinirlamasi ile) ayri ayri kontrol edilir.
    """
    if onceki is None:
        return yeni
    if maks_sicrama is not None:
        fark = yeni - onceki
        if fark > maks_sicrama:
            yeni = onceki + maks_sicrama
        elif fark < -maks_sicrama:
            yeni = onceki - maks_sicrama
    return A.YUMUSATMA_ORANI * yeni + (1 - A.YUMUSATMA_ORANI) * onceki