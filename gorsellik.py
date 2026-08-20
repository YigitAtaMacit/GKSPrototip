"""Cizim (govde/el iskeleti) ve geometri yardimcilari: dirsek acisi, gorunurluk kontrolu, EMA yumusatma, hareket algilama, OpenVINO gaze icin goz kirpintisi/head-pose hesabi."""
import collections
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
    """Bir pose landmark'in visibility skoru ayarlar.GORUNURLUK_ESIK'in ustunde mi?"""
    return nokta.visibility is not None and nokta.visibility >= A.GORUNURLUK_ESIK


def ekran_sol_sag_ayikla(sol_aday, sag_aday):
    """MediaPipe'in anatomik LEFT_*/RIGHT_* etiketi yukaridan/sirtustu kamera acisinda tutarsiz oldugu icin, iki noktadan EKRANDA daha soldaki (x kucuk) "sol" sayilir - bkz. ayarlar.EKRANA_GORE_SOL_SAG. Donus: (ekran_solu, ekran_sagi)."""
    if sol_aday.x <= sag_aday.x:
        return sol_aday, sag_aday
    return sag_aday, sol_aday


def ekran_etiket_ciz(kare, nokta, etiket, renk, w, h):
    """Sayaçlarin GERCEKTEN hangi noktayi SOL/SAG saydigini o noktanin ustune yaziyor - MediaPipe'in kendi cizimi HAM/guvenilmez anatomik etikete gore renklendirdigi icin ikisi celisebiliyordu."""
    x = int(nokta.x * w)
    y = int(nokta.y * h)
    cv2.circle(kare, (x, y), 12, renk, 2)
    cv2.putText(kare, etiket, (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2)


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


def gcs_kol_tepkisini_sinifla(ornekler, baslangic_dirsek_acisi, omuz_baslangic_y):
    """SEZGISEL (heuristik) GCS motor tepki (M2-M5) tahmini - KESIN TIBBI OLCUM DEGIL, klinisyenin kendi gozlemiyle dogrulamasi gereken bir ON-ONERI. Donus: (etiket veya None, detaylar sozlugu)."""
    if not ornekler or baslangic_dirsek_acisi is None or omuz_baslangic_y is None:
        return None, {"sebep": "izlenemedi"}

    acilar = [o[0] for o in ornekler]
    xler = [o[1] for o in ornekler]
    yler = [o[2] for o in ornekler]

    min_acisi = min(acilar)
    maks_acisi = max(acilar)
    min_bilek_y = min(yler)
    yer_degistirme = math.hypot(max(xler) - min(xler), max(yler) - min(yler))
    aci_degisimi = max(baslangic_dirsek_acisi - min_acisi, maks_acisi - baslangic_dirsek_acisi)  # dirsek acisindaki en buyuk degisim - bilek fazla yer degistirmese bile fleksiyon/ekstansiyonu yakalar

    detaylar = {
        "baslangic_acisi": round(baslangic_dirsek_acisi, 1),
        "min_acisi": round(min_acisi, 1),
        "maks_acisi": round(maks_acisi, 1),
        "min_bilek_y": round(min_bilek_y, 3),
        "omuz_baslangic_y": round(omuz_baslangic_y, 3),
        "yer_degistirme": round(yer_degistirme, 3),
        "aci_degisimi": round(aci_degisimi, 1),
    }

    _aci_degisim_esik = min(A.GCS_DEKORTIKE_DEGISIM_ESIK, A.GCS_DESEREBRE_DEGISIM_ESIK)
    if yer_degistirme < A.GCS_HAREKETSIZ_ESIK and aci_degisimi < _aci_degisim_esik:
        detaylar["sebep"] = "hareketsiz"
        return None, detaylar

    if min_bilek_y <= omuz_baslangic_y - A.GCS_LOKALIZE_PAY:
        return "M5", detaylar
    if (baslangic_dirsek_acisi - min_acisi) >= A.GCS_DEKORTIKE_DEGISIM_ESIK:
        return "M3", detaylar
    if (maks_acisi - baslangic_dirsek_acisi) >= A.GCS_DESEREBRE_DEGISIM_ESIK:
        return "M2", detaylar
    return "M4", detaylar


def kol_aktif_mi(onceki_aktif, omuz, dirsek, bilek, cikis_kare_sayaci=0,
                  min_cikis_kare=0):
    """Kol "kalkik"/"kivrik" mi - histerezisli (Schmitt trigger) + debounce (min_cikis_kare=0 ile debounce'suz eski davranis). Donus: (aktif, yeni_cikis_kare_sayaci)."""
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
    su_anki_kosul = kalkik or kivrik

    if onceki_aktif:
        if su_anki_kosul:
            yeni_cikis_kare_sayaci = 0
            aktif = True
        else:
            yeni_cikis_kare_sayaci = cikis_kare_sayaci + 1
            if yeni_cikis_kare_sayaci >= min_cikis_kare:
                aktif = False
                yeni_cikis_kare_sayaci = 0
            else:
                aktif = True
    else:
        aktif = su_anki_kosul
        yeni_cikis_kare_sayaci = 0

    return aktif, yeni_cikis_kare_sayaci


def hareket_algila(hizli_x, hizli_y, yavas_x, yavas_y, x, y, onceki_hareketli,
                    esik, histerezis_orani=0.5, hizli_oran=0.6, yavas_oran=0.03,
                    cikis_kare_sayaci=0, min_cikis_kare=8):
    """Yon-bagimsiz hareket algilama: HIZLI ve YAVAS iki EMA arasindaki fark (MACD fikri) esigi (histerezisli) asinca "hareketli" olur, min_cikis_kare kare art arda sakin kalinca debounce ile biter. Donus: (hareketli, yeni_hizli_x/y, yeni_yavas_x/y, yeni_cikis_kare_sayaci)."""
    if hizli_x is None:
        return False, x, y, x, y, 0
    yeni_hizli_x = hizli_oran * x + (1 - hizli_oran) * hizli_x
    yeni_hizli_y = hizli_oran * y + (1 - hizli_oran) * hizli_y
    yeni_yavas_x = yavas_oran * x + (1 - yavas_oran) * yavas_x
    yeni_yavas_y = yavas_oran * y + (1 - yavas_oran) * yavas_y
    mesafe = math.hypot(yeni_hizli_x - yeni_yavas_x, yeni_hizli_y - yeni_yavas_y)
    esik_kullan = esik * histerezis_orani if onceki_hareketli else esik
    if onceki_hareketli:
        # debounce: cikis esiginin altina ARKA ARKAYA min_cikis_kare kare boyunca dusmesi gerekiyor
        if mesafe <= esik_kullan:
            yeni_cikis_kare_sayaci = cikis_kare_sayaci + 1
        else:
            yeni_cikis_kare_sayaci = 0
        if yeni_cikis_kare_sayaci >= min_cikis_kare:
            hareketli = False
            yeni_cikis_kare_sayaci = 0
        else:
            hareketli = True
    else:
        hareketli = mesafe > esik_kullan
        yeni_cikis_kare_sayaci = 0
    return hareketli, yeni_hizli_x, yeni_hizli_y, yeni_yavas_x, yeni_yavas_y, yeni_cikis_kare_sayaci


def medyan_3_yumusat(gecmis_ham, x):
    """hareket_algila'ya verilecek HAM degeri bu+onceki 2 karenin medyanini alarak yumusatir - tek karelik sapan degerleri eler, gercek/surekli hareketi korur. gecmis_ham: deque(maxlen=2), yerinde guncellenir. Donus: medyan-filtrelenmis deger."""
    if len(gecmis_ham) == 2:
        efektif = sorted((gecmis_ham[0], gecmis_ham[1], x))[1]
    else:
        efektif = x
    gecmis_ham.append(x)
    return efektif


def parmak_hareket_algila(hizli_x, hizli_y, x, y, esik, hizli_oran=0.6,
                           son_tetik_uzerinden_kare=9999, min_yeniden_tetik_kare=5,
                           hizli_z=None, z=None, gecmis_ham=None):
    """Parmak ucu icin hareket_algila'dan farkli, daha tepkisel tetikleyici: TEK hizli EMA'nin ardisik-kare hizina bakar (el kalkik dururken bile her yeni kipirdanmayi ayri yakalar), esik+min_yeniden_tetik_kare (refractory) ile tetiklenir; gecmis_ham (deque(maxlen=2)) verilirse tek karelik sapan degerler medyanla elenir; z/hizli_z verilirse hiz 3 boyutlu hesaplanir. Donus: (tetiklendi_mi, yeni_hizli_x/y/z, yeni_son_tetik_uzerinden_kare, hiz)."""
    uc_boyutlu = z is not None

    if gecmis_ham is not None:
        if len(gecmis_ham) == 2:
            (ox1, oy1, oz1), (ox2, oy2, oz2) = gecmis_ham[0], gecmis_ham[1]
            x_efektif = sorted((ox1, ox2, x))[1]
            y_efektif = sorted((oy1, oy2, y))[1]
            z_efektif = sorted((oz1, oz2, z))[1] if uc_boyutlu else None
        else:
            x_efektif, y_efektif, z_efektif = x, y, z
        gecmis_ham.append((x, y, z if uc_boyutlu else 0.0))
    else:
        x_efektif, y_efektif, z_efektif = x, y, z

    if hizli_x is None:
        return False, x_efektif, y_efektif, (z_efektif if uc_boyutlu else hizli_z), son_tetik_uzerinden_kare + 1, 0.0

    yeni_hizli_x = hizli_oran * x_efektif + (1 - hizli_oran) * hizli_x
    yeni_hizli_y = hizli_oran * y_efektif + (1 - hizli_oran) * hizli_y
    if uc_boyutlu:
        yeni_hizli_z = hizli_oran * z_efektif + (1 - hizli_oran) * hizli_z
        hiz = math.sqrt((yeni_hizli_x - hizli_x) ** 2 + (yeni_hizli_y - hizli_y) ** 2
                         + (yeni_hizli_z - hizli_z) ** 2)
    else:
        yeni_hizli_z = hizli_z
        hiz = math.hypot(yeni_hizli_x - hizli_x, yeni_hizli_y - hizli_y)
    tetiklendi = hiz > esik and son_tetik_uzerinden_kare >= min_yeniden_tetik_kare
    yeni_sayac = 0 if tetiklendi else son_tetik_uzerinden_kare + 1
    return tetiklendi, yeni_hizli_x, yeni_hizli_y, yeni_hizli_z, yeni_sayac, hiz


def omuz_genisligi_piksel(sol_omuz, sag_omuz, w, h):
    """Iki omuz landmark'i arasindaki piksel mesafesi - kimlik kilidinde govde buyuklugunu (sicrama esigini olceklemek icin) temsil eder."""
    dx = (sol_omuz.x - sag_omuz.x) * w
    dy = (sol_omuz.y - sag_omuz.y) * h
    return math.hypot(dx, dy)


def govde_olcek_hesapla(sol_omuz, sag_omuz, min_olcek=0.03):
    """Omuzlar arasi normalize mesafe - kol/bacak hareketini kisi kameraya yakin/uzak fark etmeksizin ayni sekilde normalize etmek icin (bkz. govdeye_goreli_konum); min_olcek sifira bolmeye karsi taban."""
    olcek = math.hypot(sol_omuz.x - sag_omuz.x, sol_omuz.y - sag_omuz.y)
    return max(olcek, min_olcek)


def govdeye_goreli_konum(nokta, referans, olcek):
    """Uzvun (bilek/ayak bilegi) govdeye (ayni taraf omuz/kalca) gore, govde olcegine bolunmus konumu - govde-geneli kaymalarin kol/bacak sayaclarini capraz yanlis tetiklemesini matematiksel olarak iptal eder. Donus: (goreli_x, goreli_y)."""
    return (nokta.x - referans.x) / olcek, (nokta.y - referans.y) / olcek


def dijital_yakinlastir(kare, oran):
    """Kareyi tam ortasindan kirpip eski boyutuna buyutur (oran<=1.0 ise oldugu gibi doner) - eski sabit/merkez-odakli yontem, artik takip_yakinlastir tercih ediliyor."""
    if oran <= 1.0:
        return kare
    h, w = kare.shape[:2]
    kirp_w = max(1, int(w / oran))
    kirp_h = max(1, int(h / oran))
    x1 = (w - kirp_w) // 2
    y1 = (h - kirp_h) // 2
    kirpilmis = kare[y1:y1 + kirp_h, x1:x1 + kirp_w]
    return cv2.resize(kirpilmis, (w, h), interpolation=cv2.INTER_CUBIC)


def takip_yakinlastir(kare, takip_merkezi, oran):
    """dijital_yakinlastir gibi ama kirpma alani sabit merkez yerine takip_merkezi etrafinda (None ise tam ortadan) konumlanir. Donus: (yakinlastirilmis_kare, kirpma_dikdortgeni)."""
    h, w = kare.shape[:2]
    if oran <= 1.0:
        return kare, (0, 0, w, h)
    kirp_w = max(1, int(w / oran))
    kirp_h = max(1, int(h / oran))
    if takip_merkezi is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = takip_merkezi
    x1 = int(min(max(cx - kirp_w / 2.0, 0), w - kirp_w))
    y1 = int(min(max(cy - kirp_h / 2.0, 0), h - kirp_h))
    kirpilmis = kare[y1:y1 + kirp_h, x1:x1 + kirp_w]
    yakinlastirilmis = cv2.resize(kirpilmis, (w, h), interpolation=cv2.INTER_CUBIC)
    return yakinlastirilmis, (x1, y1, kirp_w, kirp_h)


def raw_konuma_cevir(x, y, kirpma_dikdortgeni, w, h):
    """takip_yakinlastir'in urettigi karedeki (x,y) pikselini HAM (kirpilmamis) kare koordinatlarina cevirir - sonraki karenin takip_merkezi'ni guncellemek icin."""
    x1, y1, kirp_w, kirp_h = kirpma_dikdortgeni
    raw_x = x1 + (x / w) * kirp_w
    raw_y = y1 + (y / h) * kirp_h
    return raw_x, raw_y


def yumusat(onceki, yeni, maks_sicrama=None, oran=None):
    """Ustel hareketli ortalama (EMA) - onceki=None ise yeni'yi dondurur; maks_sicrama verilirse EMA'ya girmeden once tek karedeki degisim buna sinirlanir (outlier korumasi)."""
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
SAG_GOZ_IDX = [33, 133, 159, 145]   # MediaPipe 478-nokta mesh, kisinin GERCEK sag gozu (kamerada SOL tarafta): disi,ici,ust,alt
SOL_GOZ_IDX = [263, 362, 386, 374]  # disi, ici, ust, alt


def goz_kutusu(landmarks, idxs, w, h, marj=None):
    marj = A.GOZ_KIRPINTI_MARJI if marj is None else marj
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * marj
    size = max(size, 20)  # kutu hicbir zaman cok kucuk/sifir olmasin
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
    """478 landmark'in tumunden yuz sinirlayici kutusunu (piksel) hesaplar - RetinaFace bbox'inin yerini tutar (kenar kutusu/ok olcegi icin)."""
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


# --- Kimlik kilidi (tek kisiye odaklanma) -----------------------------------
def kilitli_aday_sec(kilitli_merkez, kayip_kare_sayaci, merkezler, buyuklukler,
                      maks_sicrama_orani, kayip_kare_limiti):
    """Onceki karede kilitlenen kisiye en yakin adayi secer (kilit yoksa en buyugu; hicbiri esikte degilse kayip_kare_limiti kadar sabirla kilit korunur, sonra birakilir). Donus: (secilen_indeks_veya_None, yeni_kilitli_merkez, yeni_kayip_kare_sayaci)."""
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

# --- UZAK KAMERA: SABIT BOLGE ZOOM (nokta_sec.py ile bir kez elle isaretlenmis nokta, takip_yakinlastir'daki gibi her karede yeniden hesaplanmaz) ---
def bolge_kirp(kare_ham, nx, ny, oran, panel_genislik, panel_yukseklik):
    """HAM karede normalize (nx,ny) noktasi etrafinda oran'a gore kirpip panel boyutuna buyutur. Donus: (panel, kirpma_dikdortgeni=(x1,y1,kirp_w,kirp_h))."""
    h, w = kare_ham.shape[:2]
    oran = max(oran, 1.0)
    kirp_w = max(1, int(w / oran))
    kirp_h = max(1, int(h / oran))
    cx, cy = nx * w, ny * h
    x1 = int(min(max(cx - kirp_w / 2.0, 0), max(w - kirp_w, 0)))
    y1 = int(min(max(cy - kirp_h / 2.0, 0), max(h - kirp_h, 0)))
    kirpilmis = kare_ham[y1:y1 + kirp_h, x1:x1 + kirp_w]
    panel = cv2.resize(kirpilmis, (panel_genislik, panel_yukseklik), interpolation=cv2.INTER_CUBIC)
    return panel, (x1, y1, kirp_w, kirp_h)


def izgaraya_diz(paneller, sutun_sayisi=None):
    """Ayni boyuttaki panelleri tek izgara/"bolunmus ekran" goruntusune dizer - sutun_sayisi=None ise hepsi tek satirda, sigmayan hucreler siyahla doldurulur."""
    if not paneller:
        return None
    n = len(paneller)
    sutun_sayisi = sutun_sayisi or n
    satirlar = []
    for i in range(0, n, sutun_sayisi):
        satir_panelleri = paneller[i:i + sutun_sayisi]
        satirlar.append(np.hstack(satir_panelleri) if len(satir_panelleri) > 1 else satir_panelleri[0])
    if len(satirlar) == 1:
        return satirlar[0]
    maks_genislik = max(s.shape[1] for s in satirlar)
    for i, s in enumerate(satirlar):
        if s.shape[1] < maks_genislik:
            dolgu = np.zeros((s.shape[0], maks_genislik - s.shape[1], 3), dtype=s.dtype)
            satirlar[i] = np.hstack([s, dolgu])
    return np.vstack(satirlar)