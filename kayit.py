"""Kesit (JPEG) ve video (MP4) kaydi.

Bu dosya gaze backend'inden (L2CS/OpenVINO fark etmez) tamamen bagimsizdir,
oldugu gibi tasindi.

VIDEO: kareler once bellekte biriktirilir, kayit DURUNCA gercek gecen
sureden (kare sayisi / gecen saniye) dogru FPS hesaplanip OYLE yazilir -
boylece video hizli/yavas gorunmez, gercek zamanla ayni surer.
"""
import collections
import time

import cv2

import ayarlar as A

A.KESIT_KLASORU.mkdir(exist_ok=True)
A.VIDEO_KLASORU.mkdir(exist_ok=True)
A.KOL_SOL_KLASORU.mkdir(parents=True, exist_ok=True)
A.KOL_SAG_KLASORU.mkdir(parents=True, exist_ok=True)
A.GOZ_KIRPMA_KLASORU.mkdir(parents=True, exist_ok=True)
A.GOZ_BAKISI_KLASORU.mkdir(parents=True, exist_ok=True)


class VideoKaydedici:
    """Video kaydi durumunu tutan kucuk bir sinif - main dongusu sadece
    baslat()/kare_ekle()/bitir() cagirir, FPS hesabi ve dosya yazma burada."""

    def __init__(self):
        self.kayit_yapiliyor = False
        self.kareler = []
        self.baslangic_zamani = None

    def baslat(self):
        self.kayit_yapiliyor = True
        self.kareler = []
        self.baslangic_zamani = time.time()
        print("Video kaydi basladi.")

    def kare_ekle(self, kare):
        """TUM overlay'ler cizildikten SONRA cagrilmali ki dosyada da
        ekranda gordugun her sey olsun."""
        if self.kayit_yapiliyor:
            self.kareler.append(kare.copy())

    def bitir(self):
        if not self.kareler:
            self.kayit_yapiliyor = False
            self.baslangic_zamani = None
            return

        gecen_sure = time.time() - self.baslangic_zamani
        kare_sayisi = len(self.kareler)
        gercek_fps = kare_sayisi / gecen_sure if gecen_sure > 0.5 else A.VIDEO_FPS_VARSAYILAN
        gercek_fps = max(1.0, min(gercek_fps, 30.0))

        zaman_etiketi = time.strftime(A.ZAMAN_DAMGASI_FORMATI)
        video_dosya_adi = A.VIDEO_KLASORU / f"video_{zaman_etiketi}.mp4"
        yukseklik, genislik = self.kareler[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        yazici = cv2.VideoWriter(str(video_dosya_adi), fourcc, gercek_fps, (genislik, yukseklik))
        for k in self.kareler:
            yazici.write(k)
        yazici.release()

        print(
            f"Video kaydedildi: {video_dosya_adi} "
            f"({kare_sayisi} kare, {gecen_sure:.1f}s, {gercek_fps:.1f} fps ile yazildi)"
        )
        self.kareler = []
        self.kayit_yapiliyor = False
        self.baslangic_zamani = None


class OlayKlibiYoneticisi:
    """Bir OLAY (orn. kol kaldirma) tetiklendiginde, olay anindan ONCEKI ve
    SONRAKI birkac saniyeyi TEK bir MP4'te kaydeder (guvenlik kamerasi
    "event clip" mantigi).

    Kullanim: her karede kare_ekle(kare) cagir (surekli, olay olsun olmasin -
    bu, "once_saniye" kadar bir onbellek/circular buffer tutar). Bir olay
    olustugunda olay_tetikle() cagir - o andan itibaren "sonra_saniye" kadar
    daha kare toplanir, sonra onbellek + yeni kareler birlikte tek dosyaya
    yazilir. Program kapanirken bitir()'i cagirmayi unutma (yarim kalan
    klibi de yazar).

    NOT: Bir olay toplanirken (aktifken) YENI bir olay_tetikle() cagrisi
    klibi BITIRMEZ, pencereyi UZATIR (olay_zamani'ni simdiki zamana
    ceker) - yani ayni ~4sn'lik pencere icinde ikinci (ucuncu, ...) bir
    olay olursa (orn. iki kol art arda kalkarsa) TEK, daha UZUN bir klip
    cikar ve HER olayin kendi "sonrasi" da tam olarak dahil olur. Sadece
    kol tamamen "sonra_saniye" kadar sakin kalirsa klip yazilir.
    """

    def __init__(self, once_saniye, sonra_saniye, klasor, dosya_on_eki):
        self.once_saniye = once_saniye
        self.sonra_saniye = sonra_saniye
        self.klasor = klasor
        self.dosya_on_eki = dosya_on_eki
        self.onbellek = collections.deque()  # (zaman, kare) - sadece son once_saniye tutulur
        self.olay_aktif = False
        self.olay_zamani = None       # EN SON tetiklenme zamani (pencere buradan +sonra_saniye kadar acik kalir)
        self.ilk_tetik_zamani = None  # bu klibin ILK tetiklendigi zaman (gercek sure hesabi icin)
        self.sonra_kareleri = []
        self._etiketler = []

    def olay_tetikle(self, etiket=""):
        if self.olay_aktif:
            # Zaten toplanan bir klip var - BITIRMEK yerine penceresini
            # UZATIYORUZ ki bu yeni olayin da "sonrasi" tam dahil olsun.
            self.olay_zamani = time.time()
            if etiket and etiket not in self._etiketler:
                self._etiketler.append(etiket)
            print(f"Olay kesiti UZATILDI ({etiket or 'olay'}) - pencere +{self.sonra_saniye:.1f}s ileri tasindi.")
            return
        self.olay_aktif = True
        self.olay_zamani = time.time()
        self.ilk_tetik_zamani = self.olay_zamani
        self.sonra_kareleri = []
        self._etiketler = [etiket] if etiket else []
        print(f"Olay kesiti basladi ({etiket or 'olay'}) - {self.once_saniye:.1f}s once + {self.sonra_saniye:.1f}s sonra kaydedilecek...")

    def kare_ekle(self, kare):
        """TUM overlay'ler cizildikten SONRA, HER karede cagrilmali (olay
        olsun olmasin) - onbellegi surekli guncel tutar."""
        simdi = time.time()
        if self.olay_aktif:
            self.sonra_kareleri.append(kare.copy())
            if simdi - self.olay_zamani >= self.sonra_saniye:
                self._klibi_yaz()
        else:
            self.onbellek.append((simdi, kare.copy()))
            sinir = simdi - self.once_saniye
            while self.onbellek and self.onbellek[0][0] < sinir:
                self.onbellek.popleft()

    def bitir(self):
        """Program kapanirken cagir - o an toplanmakta olan (eksik bile
        olsa) klibi yazar."""
        if self.olay_aktif:
            self._klibi_yaz()

    def _klibi_yaz(self):
        simdi = time.time()
        once_kareleri = [k for (_, k) in self.onbellek]
        tum_kareler = once_kareleri + self.sonra_kareleri
        # GERCEK gecen sure: pencere UZATILMIS olabilir (birden fazla olay
        # ust uste geldiyse), bu yuzden sabit once+sonra yerine ilk tetikten
        # (once_saniye kadar geriden) simdiye kadarki GERCEK sureyi kullan.
        pencere_baslangici = (self.ilk_tetik_zamani - self.once_saniye) if self.ilk_tetik_zamani else simdi
        gecen_sure = simdi - pencere_baslangici
        etiketler = list(self._etiketler)

        self.olay_aktif = False
        self.sonra_kareleri = []
        self.onbellek.clear()
        self.ilk_tetik_zamani = None

        if not tum_kareler:
            return

        gercek_fps = len(tum_kareler) / gecen_sure if gecen_sure > 0 else A.VIDEO_FPS_VARSAYILAN
        gercek_fps = max(1.0, min(gercek_fps, 30.0))

        zaman_etiketi = time.strftime(A.ZAMAN_DAMGASI_FORMATI)
        etiket_parcasi = f"_{'+'.join(etiketler)}" if etiketler else ""
        video_dosya_adi = self.klasor / f"{self.dosya_on_eki}{etiket_parcasi}_{zaman_etiketi}.mp4"
        yukseklik, genislik = tum_kareler[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        yazici = cv2.VideoWriter(str(video_dosya_adi), fourcc, gercek_fps, (genislik, yukseklik))
        for k in tum_kareler:
            yazici.write(k)
        yazici.release()

        print(
            f"Olay kesiti kaydedildi: {video_dosya_adi} "
            f"({len(tum_kareler)} kare, ~{gecen_sure:.1f}s, {gercek_fps:.1f} fps ile yazildi)"
        )


def kesit_al(kare, sayaclar):
    """O anki kareyi (TUM overlay'lerle) kesitler/ klasorune JPEG olarak kaydeder."""
    zaman_etiketi = time.strftime(A.ZAMAN_DAMGASI_FORMATI)
    kesit_dosya_adi = A.KESIT_KLASORU / f"kesit_{zaman_etiketi}.jpg"
    cv2.imwrite(str(kesit_dosya_adi), kare)
    sayaclar["kesit"] += 1
    print(f"Kesit kaydedildi: {kesit_dosya_adi}")