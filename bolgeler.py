"""SABIT BOLGE (zoom) noktalarinin dosyaya kaydedilmesi/okunmasi.

nokta_sec.py bu modulle noktalari zoom_noktalari.json'a YAZAR,
gaze_birlesik_uzak.py AYNI moduIle OKUR - format ikisi arasinda TEK bir
yerde (burada) tanimli, boylece iki dosya ARASINDA tutarsizlik olusmaz.

DESTEKLENEN BOLGELER (bkz. ayarlar.py'deki BOLGE_* aciklamasi): sadece 3
SABIT ad/slot var - "yuz", "sol_el", "sag_el" (projedeki sol_parmak/
sag_parmak, sol/sag/yukari/asagi SAYAÇLARIYLA birebir eslesecek sekilde
BILEREK sabit tutuldu; hasta/kamera basina zaten TEK yuz + TEK sol el +
TEK sag el olur). Her biri en fazla BIR noktaya sahip olabilir - nokta_sec.py
ayni turu tekrar tikladiginda ESKI noktanin YERINE yenisini yazar.
"""
import json

import ayarlar as A

# ad (JSON anahtari) -> tur (gaze_birlesik_uzak.py'nin hangi MediaPipe
# modelini/sayacini kullanacagini belirler).
TUR_ESLESTIRME = {"yuz": "yuz", "sol_el": "el", "sag_el": "el"}


def bolgeleri_yukle(dosya_yolu=None):
    """Donus: {ad: {"x":float, "y":float, "oran":float, "tur":str}}.

    Dosya yoksa/bossa/bozuksa BOS sozluk doner (hata FIRLATMAZ) - proje
    genelinde benimsenen "supheliyse/eksikse sessizce atla, uygulamayi
    COKERTME" ilkesiyle tutarli: gaze_birlesik_uzak.py bolge tanimlanmamis
    bir turu sadece o turu ISLEMEDEN atlar, genis-aci (kol/bacak) gorunumu
    yine de calismaya devam eder.
    """
    dosya_yolu = dosya_yolu or A.BOLGE_NOKTALARI_DOSYASI
    if not dosya_yolu.exists():
        return {}
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            veri = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[uyari] {dosya_yolu} okunamadi ({e}) - bolge zoom modu bu tur icin KAPALI kalacak.")
        return {}

    ham_bolgeler = veri.get("bolgeler", {})
    sonuc = {}
    for ad, b in ham_bolgeler.items():
        if ad not in TUR_ESLESTIRME:
            print(f"[uyari] {dosya_yolu} icinde bilinmeyen bolge adi '{ad}' - yoksayildi "
                  f"(gecerli adlar: {sorted(TUR_ESLESTIRME)}).")
            continue
        try:
            sonuc[ad] = {
                "x": float(b["x"]),
                "y": float(b["y"]),
                "oran": float(b.get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN)),
                "tur": TUR_ESLESTIRME[ad],
            }
        except (KeyError, TypeError, ValueError) as e:
            print(f"[uyari] '{ad}' bolgesi okunamadi ({e}) - yoksayildi.")
    return sonuc


def bolgeleri_kaydet(bolgeler, kamera_indeksi, dosya_yolu=None):
    """bolgeler: {ad: {"x","y","oran",...}} (nokta_sec.py'nin kendi ic
    sozlugu - "tur" alani olsa da olmasa da fark etmez, YAZILMAZ, cunku
    OKURKEN zaten TUR_ESLESTIRME'den yeniden turetiliyor).
    """
    dosya_yolu = dosya_yolu or A.BOLGE_NOKTALARI_DOSYASI
    kaydedilecek = {
        ad: {"x": b["x"], "y": b["y"], "oran": b.get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN)}
        for ad, b in bolgeler.items()
    }
    veri = {"kamera_indeksi": kamera_indeksi, "bolgeler": kaydedilecek}
    with open(dosya_yolu, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)