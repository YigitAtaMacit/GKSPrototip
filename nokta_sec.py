"""SABIT BOLGE (zoom) noktalarini fare ile isaretleme araci - gaze_birlesik_uzak.py'nin hangi ekran bolgesinin yuz/sol el/sag el oldugunu bilmesi icin.

Kullanim: python nokta_sec.py [kamera_indeksi].
Adimlar: kamerabul.py ile indeks bul -> bu scripti calistir -> 1/2/3 ile yuz/sol el/sag el sec, sol tikla -> +/- ile zoom orani -> 's' ile kaydet.
Tuslar: 1/2/3=bolge sec, sol tik=nokta yerlestir, +/-=zoom orani, u=sil, s=kaydet, q=kaydetmeden cik.
"""
import sys

import cv2

import ayarlar as A
import bolgeler as B

TUR_RENK = {"yuz": (0, 255, 0), "sol_el": (255, 0, 255), "sag_el": (0, 255, 255)}
TUR_ETIKET = {"yuz": "YUZ", "sol_el": "SOL EL", "sag_el": "SAG EL"}
TUR_SIRASI = ["yuz", "sol_el", "sag_el"]

kamera_indeksi = int(sys.argv[1]) if len(sys.argv) > 1 else A.KAMERA_INDEKSI

bolgeler = B.bolgeleri_yukle()  # onceden kaydedilmis bolgeler varsa yuklenir, uzerine duzenlenebilir
secili_tur = "yuz"
kaydedilmemis_degisiklik = False

_son_kare = None  # fare geri cagirimi (callback) icin - o anki karenin boyutu lazim


def _fare_olayi(event, x, y, flags, param):
    global bolgeler, kaydedilmemis_degisiklik
    if event == cv2.EVENT_LBUTTONDOWN and _son_kare is not None:
        hh, ww = _son_kare.shape[:2]
        onceki_oran = bolgeler.get(secili_tur, {}).get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN)
        bolgeler[secili_tur] = {"x": x / ww, "y": y / hh, "oran": onceki_oran}
        kaydedilmemis_degisiklik = True
        print(f"[{TUR_ETIKET[secili_tur]}] nokta yerlestirildi: piksel=({x},{y})  oran={onceki_oran:.1f}x")


cap = cv2.VideoCapture(kamera_indeksi)
if not cap.isOpened():
    raise SystemExit(
        f"Kamera acilamadi (indeks {kamera_indeksi}). Once kamerabul.py ile dogru "
        "indeksi bul (python kamerabul.py), sonra 'python nokta_sec.py <indeks>' ile dene."
    )

cv2.namedWindow("Nokta sec")
cv2.setMouseCallback("Nokta sec", _fare_olayi)

print(__doc__)
print(f"[bilgi] kamera indeksi {kamera_indeksi} ile acildi. Onceden kayitli bolge sayisi: {len(bolgeler)}")

while True:
    ok, kare = cap.read()
    if not ok:
        print("[hata] kameradan kare okunamadi - kamera baska bir uygulama tarafindan mi kullaniliyor?")
        break
    _son_kare = kare
    h, w = kare.shape[:2]

    # Tanimli TUM bolgeleri + (o anki oranla) kirpma alani onizlemesini ciz.
    for ad, bilgi in bolgeler.items():
        renk = TUR_RENK[ad]
        cx, cy = int(bilgi["x"] * w), int(bilgi["y"] * h)
        oran = max(bilgi.get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN), 1.0)
        kirp_w, kirp_h = int(w / oran), int(h / oran)
        x1 = min(max(cx - kirp_w // 2, 0), max(w - kirp_w, 0))
        y1 = min(max(cy - kirp_h // 2, 0), max(h - kirp_h, 0))
        kalinlik = 3 if ad == secili_tur else 1
        cv2.rectangle(kare, (x1, y1), (x1 + kirp_w, y1 + kirp_h), renk, kalinlik)
        cv2.circle(kare, (cx, cy), 6, renk, -1)
        cv2.putText(kare, f"{TUR_ETIKET[ad]} ({oran:.1f}x)", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2)

    cv2.putText(kare, f"SECILI: {TUR_ETIKET[secili_tur]}  (1:yuz  2:sol el  3:sag el)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TUR_RENK[secili_tur], 2)
    _durum = "   ".join(f"{TUR_ETIKET[t]}:{'OK' if t in bolgeler else '-'}" for t in TUR_SIRASI)
    cv2.putText(kare, _durum, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if kaydedilmemis_degisiklik:
        cv2.putText(kare, "KAYDEDILMEMIS DEGISIKLIK VAR - 's' ile kaydet!", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(kare, "sol tik: yerlestir | +/-: zoom orani | u: sil | s: kaydet | q: cik",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Nokta sec", kare)
    tus = cv2.waitKey(1) & 0xFF

    if tus == ord("q"):
        break
    if tus == ord("s"):
        B.bolgeleri_kaydet(bolgeler, kamera_indeksi)
        kaydedilmemis_degisiklik = False
        print(f"Kaydedildi: {A.BOLGE_NOKTALARI_DOSYASI}  ({len(bolgeler)} bolge: {sorted(bolgeler)})")
    if tus == ord("1"):
        secili_tur = "yuz"
    if tus == ord("2"):
        secili_tur = "sol_el"
    if tus == ord("3"):
        secili_tur = "sag_el"
    if tus == ord("u"):
        if secili_tur in bolgeler:
            del bolgeler[secili_tur]
            kaydedilmemis_degisiklik = True
            print(f"[{TUR_ETIKET[secili_tur]}] nokta silindi (kaydetmeyi unutma: 's').")
    if tus in (ord("+"), ord("=")):
        if secili_tur in bolgeler:
            bolgeler[secili_tur]["oran"] = min(bolgeler[secili_tur].get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN) + 0.5, 15.0)
            kaydedilmemis_degisiklik = True
    if tus in (ord("-"), ord("_")):
        if secili_tur in bolgeler:
            bolgeler[secili_tur]["oran"] = max(bolgeler[secili_tur].get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN) - 0.5, 1.0)
            kaydedilmemis_degisiklik = True

cap.release()
cv2.destroyAllWindows()

if kaydedilmemis_degisiklik:
    print("UYARI: kaydedilmemis degisiklikler vardi, KAYBOLDU (cikmadan once 's' ile kaydetmen gerekiyordu).")