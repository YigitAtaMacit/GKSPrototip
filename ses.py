"""Mikrofon<->hoparlor: hastanin ses cikarmasini ALGILAMA + kullanici sesini hoparlore CANLI GECIRME (sounddevice/PortAudio, full-duplex, ayri PortAudio thread'inde)."""
import time

import numpy as np
import sounddevice as sd

import ayarlar as A

_stream = None
_passthrough_acik = False

aktif = True  # ana dongudeki sayaclar_aktif ile birlikte guncellenir (bkz. gaze_birlesik_uzak.py 'h' tusu)

ses_sayaci = 0
son_giris_rms = 0.0
son_cikis_rms = 0.0

_ses_hareketli = False
_sakin_baslangic_zamani = None


def _callback(indata, outdata, frames, zaman_bilgisi, durum):
    global ses_sayaci, son_giris_rms, son_cikis_rms, _ses_hareketli, _sakin_baslangic_zamani

    giris = indata[:, 0]
    rms = float(np.sqrt(np.mean(np.square(giris), dtype=np.float64)))
    son_giris_rms = rms

    if _passthrough_acik:
        outdata[:, 0] = giris
        cikis_rms = rms
    else:
        outdata[:, 0] = 0.0
        cikis_rms = 0.0
    son_cikis_rms = cikis_rms

    if cikis_rms > A.SES_PASSTHROUGH_ESIK:
        _sakin_baslangic_zamani = None  # kendi sesimiz caliniyor - akustik geri beslemeyi onlemek icin algilama yapma
        return

    if not aktif:
        return

    esik_kullan = A.SES_ALGILAMA_ESIK * A.SES_ALGILAMA_HISTEREZIS_ORANI if _ses_hareketli else A.SES_ALGILAMA_ESIK
    simdi = time.time()
    if rms > esik_kullan:
        if not _ses_hareketli:
            ses_sayaci += 1
            _ses_hareketli = True
        _sakin_baslangic_zamani = None
    elif _ses_hareketli:
        if _sakin_baslangic_zamani is None:
            _sakin_baslangic_zamani = simdi
        elif simdi - _sakin_baslangic_zamani >= A.SES_ALGILAMA_MIN_CIKIS_SANIYE:
            _ses_hareketli = False
            _sakin_baslangic_zamani = None


def baslat():
    """Ses akisini baslatir (AKTIF_SES=False ya da zaten baslamissa hicbir sey yapmaz)."""
    global _stream
    if not A.AKTIF_SES or _stream is not None:
        return
    try:
        _stream = sd.Stream(
            samplerate=A.SES_ORNEKLEME_HIZI, blocksize=A.SES_BLOK_BOYUTU,
            channels=1, dtype="float32",
            device=(A.SES_GIRIS_CIHAZI, A.SES_CIKIS_CIHAZI),
            callback=_callback,
        )
        _stream.start()
        print("[ses] mikrofon/hoparlor akisi baslatildi "
              f"({A.SES_ORNEKLEME_HIZI} Hz, blok={A.SES_BLOK_BOYUTU}).")
    except Exception as e:
        _stream = None
        print(f"[uyari] ses akisi baslatilamadi ({e}) - SES sayaci ve mikrofon "
              "gecisi bu oturumda CALISMAYACAK, geri kalan her sey normal calisir. "
              "'python -c \"import sounddevice as sd; print(sd.query_devices())\"' "
              "ile cihazlari kontrol edip ayarlar.SES_GIRIS_CIHAZI/SES_CIKIS_CIHAZI'i "
              "dogru indekse ayarlamayi deneyin.")


def durdur():
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None


def passthrough_ac_kapat():
    global _passthrough_acik
    _passthrough_acik = not _passthrough_acik
    return _passthrough_acik


def passthrough_durumu():
    return _passthrough_acik
