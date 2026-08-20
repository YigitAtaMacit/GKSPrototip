"""SABIT BOLGE (zoom) noktalarinin zoom_noktalari.json'a kaydi/okumasi - nokta_sec.py yazar, gaze_birlesik_uzak.py okur; sadece "yuz"/"sol_el"/"sag_el" adlari gecerli, her biri tek nokta tutar."""
import json

import ayarlar as A

# ad (JSON anahtari) -> tur (hangi MediaPipe modelini/sayacini kullanacagini belirler).
TUR_ESLESTIRME = {"yuz": "yuz", "sol_el": "el", "sag_el": "el"}


def bolgeleri_yukle(dosya_yolu=None):
    """Donus: {ad: {"x","y","oran","tur"}} - dosya yoksa/bozuksa hata firlatmadan bos sozluk doner."""
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
    """bolgeler: {ad: {"x","y","oran",...}} - "tur" alani yazilmaz, okurken TUR_ESLESTIRME'den yeniden turetilir."""
    dosya_yolu = dosya_yolu or A.BOLGE_NOKTALARI_DOSYASI
    kaydedilecek = {
        ad: {"x": b["x"], "y": b["y"], "oran": b.get("oran", A.BOLGE_ZOOM_ORANI_VARSAYILAN)}
        for ad, b in bolgeler.items()
    }
    veri = {"kamera_indeksi": kamera_indeksi, "bolgeler": kaydedilecek}
    with open(dosya_yolu, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)