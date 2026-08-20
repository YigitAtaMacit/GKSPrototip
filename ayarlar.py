"""MediaPipe + Intel OpenVINO (gaze-estimation-adas-0002) birlesik uygulamasi
icin TUM ayarlar/sabitler/yollar.

Davranisi degistirmek istedigin zaman (esik degerleri, hangi cizimlerin
gorunecegi, hangi modellerin aktif olacagi vb.) SADECE bu dosyayi duzenlemen
yeterli - diger dosyalar (modeller.py, gorsellik.py, kayit.py,
gaze_birlesik.py) buradaki degerleri kullanir.

LISANS NOTU: Bu surum, L2CS-Net'in Gaze360 ile egitilmis agirliklari yerine
Intel OpenVINO'nun gaze-estimation-adas-0002 modelini kullanir. Hem bu model
hem MediaPipe Apache 2.0 lisanslidir ve Gaze360/MPIIGaze'deki "sadece
arastirma amacli" kisitlamasini TASIMAZ - ticari kullanima uygundur.
"""
from pathlib import Path

BURASI = Path(__file__).resolve().parent

# --- Acik/kapali ozellikler --------------------------------------------------
# Performans icin acik/kapali yapilabilir modeller - yavas gelirse False yap.
# AKTIF_GAZE=False: OpenVINO gaze modeli HIC YUKLENMEZ/CALISMAZ - bakis yonu
# sayaclari (SOL/SAG/YUKARI/ASAGI) ve goz oku bu durumda calismaz.
AKTIF_GAZE = True
AKTIF_POSE = True
AKTIF_EL = True
AKTIF_SES = True  

# --- Kimlik kilidi (tek kisiye odaklanma) --------------------------------
# Ekrana ilk giren kisi "kilitlenir" - sonradan giren baska biri ne kadar
# yakinda/onde olursa olsun sayaclar/bakis SADECE o kilitli kisi icin
# calismaya devam eder, digerleri TAMAMEN yok sayilir. GORUNTUYU KIRPMAZ -
# sadece MediaPipe'in dondurdugu adaylar arasindan (asagidaki *_ADAY_SAYISI
# kadar) hangisinin "kilitli kisi" oldugunu her karede yeniden bulur (en
# yakin komsu + boyut esigi ile). Pahali olan OpenVINO bakis modeli YINE
# SADECE kilitli kisi icin calisir (adaylarin geri kalani icin calismaz),
# bu yuzden ikinci bir kisi girdiginde performans/donma sorunu YARATMAZ.
KIMLIK_KILIDI_AKTIF = True
YUZ_ADAY_SAYISI = 3    # FaceLandmarker'in dondurecegi maks aday sayisi
GOVDE_ADAY_SAYISI = 3  # PoseLandmarker'in dondurecegi maks aday sayisi

# Bir karede kilitli kisiye "en yakin" aday, kilitli kisinin SON pozisyonuna
# gore kendi buyuklugunun (yuz genisligi / omuz genisligi) bu ORANINDAN
# fazla uzaklastiysa REDDEDILIR (baska biri sayilir). Kucultursen kilit
# daha "siki" olur (kolayca baska birine gecmez) ama hizli hareketlerde
# kilidi kaybetme ihtimali artar.
KIMLIK_KILIDI_MAKS_SICRAMA_ORANI = 0.9

# Kilitli kisi bu kadar KARE boyunca hic eslesmezse (kafasini tam cevirmis,
# kadraj disina cikmis vb.) kilit tamamen birakilir - bir sonraki karede en
# BUYUK/on plandaki kisiye yeniden kilitlenir.
KIMLIK_KILIDI_KAYIP_KARE_LIMITI = 45  # ~1.5 saniye, ~30fps'te

# Otomatik OLAY KESITI (once/sonra video klibi) kategori basina ac/kapa -
# simdilik UCU DE kapali. SAYACLAR (KIRPMA/SOL KOL/SAG KOL/SOL BACAK/SAG
# BACAK/SOL/SAG/YUKARI/ASAGI) bundan ETKILENMEZ, hepsi normal calismaya
# devam eder - sadece otomatik VIDEO KLIBI alinmiyor. Tekrar acmak istersen
# ilgiliyi True yap.
KOL_OLAY_KESITI_AKTIF = False
BACAK_OLAY_KESITI_AKTIF = False
KIRPMA_OLAY_KESITI_AKTIF = False
BAKIS_OLAY_KESITI_AKTIF = False

# --- Bacak/ayak hareketi (bkz. gorsellik.hareket_algila) ------------------
# KOL'daki "yukari kalkti mi" gibi YONE OZGU bir kural DEGIL - hasta yatarken
# bile (bacagini "yukari kaldirma" imkani olmasa da) ayak bilegi konumundaki
# HERHANGI BIR yone dogru (ani sicrama VEYA yavas/surekli hareket, IKISI DE)
# yeterince buyuk bir degisikligi "hareket" sayar - IKI FARKLI HIZDA EMA
# (HIZLI/YAVAS) arasindaki farka bakilarak (MACD benzeri, bkz. fonksiyonun
# docstring'i - TEK EMA hem "hareket sonrasi sonsuza dek kilitlenme" hem de
# "yavas hareketi hic yakalayamama" sorunlarina yol acmisti, gercek videoyla
# dogrulandi). Ayak bilegi secildi (bkz. proje karari) - battaniye/carsaf
# altinda bile genelde ayak ucundan daha guvenilir gorunur kalir.

BACAK_HAREKET_ESIK = 0.035            # normalize (0..1) - HIZLI/YAVAS EMA farki bu kadar olunca "hareket" baslar
BACAK_HAREKET_HISTEREZIS_ORANI = 0.55 # cikis esigi = ESIK * bu oran
BACAK_HAREKET_HIZLI_ORAN = 0.6        # HIZLI EMA'nin HAM konuma yaklasma hizi - dusuruldu (0.75->0.6), tek-kare gurultusune daha az duyarli olsun diye
BACAK_HAREKET_YAVAS_ORAN = 0.035      # YAVAS EMA'nin HAM konuma yaklasma hizi

# DEBOUNCE: bacak KALDIRILIP HAVADA TUTULURKEN dogal kas titremesi, yukaridaki
# duyarli esik/histerezis ile birlesince AYNI TEK hareketi 2-3 kez saydiriyordu
# (gercek kullanici testiyle bulundu ve dogrulandi - bkz. gorsellik.hareket_algila
# docstring'i). "Hareketli" durumundan cikmak icin mesafenin TEK karede degil,
# ARKA ARKAYA en az bu kadar kare boyunca cikis esiginin ALTINDA kalmasi sart -
# 30 senaryoluk simulasyonla (bacak kaldirip titreyerek tutma) dogrulandi:
# min_cikis_kare=8 ile tum senaryolarda tek hareket = tek sayim, yanlis pozitif
# yok, ve gercekten AYRI iki hareket (~2 saniyeden fazla arayla) hala AYRI sayiliyor.
BACAK_HAREKET_MIN_CIKIS_KARE = 8

# False yaparsan ilgili gorsel EKRANDA CIZILMEZ ama tespit (ve dolayisiyla
# ilgili sayaclar/kalibrasyon) YINE DE CALISIR - sadece gorsel gizlenir.
GOVDE_CIZIMI_GOSTER = True
EL_CIZIMI_GOSTER = True
YUZ_CIZIMI_GOSTER = True

# FaceLandmarker'in yuzu "yuz" olarak kabul etmesi icin gereken minimum
# guven skoru (0-1). Varsayilan MediaPipe degeri 0.5'tir. TEPEDEN/DIREK KAMERA gibi asiri acili kurulumlarda (yuz
# normalden COK farkli/sikismis gorunuyor) bunu daha da dusurmek tespiti
# kolaylastirir - ama COK dusurursen yuz OLMAYAN seyleri de "yuz" sanma
# riski artar (yanlis pozitif). 0.1-0.15 arasi genelde makul bir sinir.

YUZ_TESPIT_ESIK = 0.08

# HandLandmarker'in bir bolgeyi/kareyi "el" olarak kabul etmesi icin gereken
# minimum guven skoru (0-1, hem tespit HEM izleme/tracking icin - bkz.
# modeller.py'de nerede kullanildigi). Varsayilan MediaPipe degeri 0.5'tir.

EL_TESPIT_ESIK = 0.3

EL_IZLEME_ESIK = 0.6


DIJITAL_YAKINLASTIRMA = 1.0

# --- Kamera --------------------------------------------------------------
# USB kamera (A4Tech FHD 1080p) takinca isletim sistemi ona 0'dan farkli bir
# indeks verebilir (ozellikle laptop'ta dahili kamera zaten 0'i kullaniyorsa
# USB kamera genelde 1 olur). Uygulamayi calistirinca acilan pencerede YANLIS
# kamera goruntusu geliyorsa bu degeri 1, 2... diye degistirip tekrar dene.
KAMERA_INDEKSI = 1

# DENENDI: cv2.set() ile 1920x1080@30 + MJPG'yi ACIKCA istemek bu kamerada
# Windows Kamera uygulamasindan DAHA KOTU goruntu verdi (muhtemelen surucu
# bu explicit istegi otomatik/varsayilan modundan daha kotu bir moda
# dusuruyor). Bu yuzden KAMERA_COZUNURLUK_ZORLA=False - kamera artik HICBIR
# .set() cagrisi olmadan, sadece KAMERA_INDEKSI ile aciliyor (eskiden iyi
# calisan hal). True yaparsan asagidaki degerler zorlanir - farkli bir
# kamerada/surucude ise yarayabilir, dene istersen.
KAMERA_COZUNURLUK_ZORLA = False
KAMERA_GENISLIK = 1920
KAMERA_YUKSEKLIK = 1080
KAMERA_FPS = 30
KAMERA_FOURCC = "MJPG"

# --- Dosya yollari -----------------------------------------------------------
# gaze-estimation-adas-0002.xml VE .bin - ilk calistirmada BURAYA otomatik
# indirilir (internet gerekir, sadece ilk sefer; MediaPipe .task dosyalari
# gibi). Indirme basarisiz olursa elle bu iki URL'den indirip BURASI'na
# koyabilirsin:
GAZE_MODEL_XML_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
GAZE_MODEL_BIN_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.bin"
GAZE_MODEL_XML = BURASI / "gaze-estimation-adas-0002.xml"
GAZE_MODEL_BIN = BURASI / "gaze-estimation-adas-0002.bin"
GAZE_CIHAZ = "CPU"  # "GPU" da denenebilir ama SADECE Intel GPU'larda calisir (NVIDIA'da degil)

FACE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_TASK_YOLU = BURASI / "face_landmarker.task"

POSE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_TASK_YOLU = BURASI / "pose_landmarker_lite.task"

HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_TASK_YOLU = BURASI / "hand_landmarker.task"

KESIT_KLASORU = BURASI / "kesitler"
VIDEO_KLASORU = BURASI / "videolar"

# Olay kesitleri (otomatik once/sonra klipleri) kategoriye gore AYRI alt
# klasorlere yazilir - hepsi videolar/ altinda.
KOL_SOL_KLASORU = VIDEO_KLASORU / "sol_kol"
KOL_SAG_KLASORU = VIDEO_KLASORU / "sag_kol"
BACAK_SOL_KLASORU = VIDEO_KLASORU / "sol_bacak"
BACAK_SAG_KLASORU = VIDEO_KLASORU / "sag_bacak"
GOZ_KIRPMA_KLASORU = VIDEO_KLASORU / "goz_kirpma"
GOZ_BAKISI_KLASORU = VIDEO_KLASORU / "goz_bakisi"

# Kayit dosya adlarindaki zaman damgasinin bicimi: gun_ay_yil_saat_dakika_saniye
ZAMAN_DAMGASI_FORMATI = "%d_%m_%Y_%H_%M_%S"

# --- Kenar/kol/kirpma esikleri ------------------------------------------------
# Kenar "kutusu" YUZE GORE (merkez_x, merkez_y, uzunluk - yuz genisligi)
# hesaplanir, ekrana sabit degildir - hareket ettikce kutu da SENINLE
# birlikte kayar. Degerler "yuz genisligi carpi bu oran" olarak merkezden
# mesafedir.
KENAR_MESAFE_YATAY = 0.30  # SOL/SAG icin merkezden mesafe
KENAR_MESAFE_UST = 0.20    # YUKARI icin merkezden mesafe (kucuk = daha kolay tetiklenir)
# ASAGI icin merkezden mesafe - BAKIS_YANAL_SIZINTI_K duzeltmesi asagi
# bakisin kendi gy sinyalini de biraz zayiflatiyor (ham ~-0.38 iken
# duzeltmeden sonra ~-0.23'e dusuyor) - 0.40'ta hemen hic tetiklenmiyordu,
# bu yuzden UST ile ayni seviyeye (0.20) cektik.
KENAR_MESAFE_ALT = 0.20

# OpenVINO modeli normalize edilmis 3D gaze VEKTORU (x,y,z) donduruyor -
# L2CS'teki gibi pitch/yaw radyan degil. "Duz bakis" olu bolgesi bu yuzden
# vektorun x/y bilesenleri uzerinden (boyutsuz, -1..1 araligi) tanimlaniyor.
ESIK_BAKIS_XY = 0.10

# CAPRAZ EKSEN SIZINTISI: 5 yonde (duz/yukari/asagi/sol/sag) ham gx/gy
# olculdu, sizinti TEK bir dogrusal sabitle (gx = k*gy gibi) ACIKLANAMADI -
# iki AYRI, ASIMETRIK olay var:
#   1) SADECE asagi bakinca (gy<0) gx'e POZITIF sizinti biniyor - yukari
#      bakinca (gy>0) bu sizinti YOK (gx~0 kaldi). Olculen: asagida
#      gx=+0.33..0.35 iken gy=-0.38 -> oran ~0.87.
#   2) YANLARA (sol VEYA sag, |gx| buyudukce) bakinca gy'e NEGATIF (asagi
#      yonlu) bir sizinti biniyor - bu YONDEN BAGIMSIZ, |gx| ile orantili.
#      Olculen: sagda gx=-0.43/gy=-0.20, solda gx=+0.52/gy=-0.24 -> ikisinde
#      de oran ~0.46.
# Bu muhtemelen modelin kendi zaafi (goz kapaklari asagi/yana bakista
# kirpintinin gorunumunu degistiriyor) - matematiksel bir hata degil, bu
# yuzden tek sabitle tam duzelmiyor, sadece BUYUK kismi giderilebiliyor.
BAKIS_ASAGI_SIZINTI_K = 0.87  # sadece gy<0 iken gx'ten cikarilir
BAKIS_YANAL_SIZINTI_K = 0.46  # her zaman |gx| ile orantili, gy'e eklenir




# --- YENI: Kol hareketi (bkz. gorsellik.hareket_algila) -------------------
# BACAK_HAREKET_* ile TAM AYNI mantik/parametreler, ayni fonksiyon - "kol
# kalkik mi" gibi YONE OZGU bir kural DEGIL, bilek konumundaki HERHANGI BIR
# yone dogru (ani sicrama VEYA yavas/surekli hareket) yeterince buyuk bir
# degisikligi "hareket" sayar. Boylece hasta SIRTUSTU yatip kolunu govdesi
# uzerinde/yana dogru hareket ettirse bile (bilek hicbir zaman omuzden
# "yukari kalkmasa" bile) yakalanir.
KOL_HAREKET_ESIK = 0.035
KOL_HAREKET_HISTEREZIS_ORANI = 0.55
KOL_HAREKET_HIZLI_ORAN = 0.6
KOL_HAREKET_YAVAS_ORAN = 0.035
KOL_HAREKET_MIN_CIKIS_KARE = 8

# --- YENI (SADECE gaze_birlesik.py): GOVDEYE GORELI kol/bacak hareketi ---
# !!! ESKI YONTEMIN (yukaridaki KOL_HAREKET_*/BACAK_HAREKET_*, MUTLAK/kare-
# ici konum kullanan) YERINE gaze_birlesik.py'de KULLANILIYOR - l2cs_birlesik.py
# ve uniface_birlesik.py HALA eski yontemi kullaniyor, onlara dokunulmadi. !!!

# Ayni fonksiyon (hareket_algila) kullaniliyor, sadece x,y girdisi MUTLAK
# konum yerine bu goreli/olceklendirilmis konum - esik degerleri de buna
# gore YENIDEN kalibre edildi (birim artik "govde olcegi", 0.22 ~= omuz
# genisliginin %22'si kadar bir yer degistirme).
KOL_HAREKET_GORELI_ESIK = 0.22
KOL_HAREKET_GORELI_HISTEREZIS_ORANI = 0.55
KOL_HAREKET_GORELI_HIZLI_ORAN = 0.6
KOL_HAREKET_GORELI_YAVAS_ORAN = 0.035
KOL_HAREKET_GORELI_MIN_CIKIS_KARE = 8


KAFA_HAREKET_ESIK = 10.0
KAFA_HAREKET_HISTEREZIS_ORANI = 0.8
KAFA_HAREKET_HIZLI_ORAN = 0.6
KAFA_HAREKET_YAVAS_ORAN = 0.035
KAFA_HAREKET_MIN_CIKIS_KARE = 4


KAFA_KONUM_HAREKET_ESIK = 0.20
KAFA_KONUM_HAREKET_HISTEREZIS_ORANI = 0.8
KAFA_KONUM_HAREKET_HIZLI_ORAN = 0.6
KAFA_KONUM_HAREKET_YAVAS_ORAN = 0.035
KAFA_KONUM_HAREKET_MIN_CIKIS_KARE = 4

BACAK_HAREKET_GORELI_ESIK = 0.22
BACAK_HAREKET_GORELI_HISTEREZIS_ORANI = 0.55
BACAK_HAREKET_GORELI_HIZLI_ORAN = 0.6
BACAK_HAREKET_GORELI_YAVAS_ORAN = 0.035
BACAK_HAREKET_GORELI_MIN_CIKIS_KARE = 8

# --- GOVDE OLCEGI YUMUSATMA (bkz. gorsellik.govde_olcek_hesapla) ----------
# SORUN (gercek kullanici videosuyla bulundu, 17.08.2026): govde_olcek SADECE
# omuzlardan (2 nokta) hesaplaniyor ve KOL + BACAK'IN DORDU DE (sol/sag)
# AYNI govde_olcek'e BOLUNUYOR (bkz. govdeye_goreli_konum).
GOVDE_OLCEK_YUMUSATMA_ORANI = 0.2
GOVDE_OLCEK_MAKS_SICRAMA = 0.02  # tek karede govde_olcek'in degisebilecegi AZAMI miktar



GOVDE_OLCEK_KABUL_MIN_ORAN = 0.6  # ham okuma, yumusatilmisin bu oranindan KUCUKSE yoksay
GOVDE_OLCEK_KABUL_MAKS_ORAN = 1.6  # ham okuma, yumusatilmisin bu oranindan BUYUKSE yoksay

# --- Parmak hareketi (17.08.2026 eklendi, SADECE gaze_birlesik.py) --------
# AMAC: hastanin parmaklarini oynatip oynatmadigini (kaba kol/bacak
# hareketinden BAGIMSIZ, ince motor tepkisi) yakalamak - orn. "eli tut, elini
# sik" gibi bir emre parmak duzeyinde tepki var mi diye.
#

PARMAK_HIZ_ESIK = 0.015       # ardisik iki karedeki (yumusatilmis) konum degisimi bu kadar olursa "hareket" (panel genisligi/yuksekliginin orani, 0..1)


PARMAK_HIZ_ESIK_GORELI = 0.06
PARMAK_HIZ_HIZLI_ORAN = 0.6      # yumusatma orani (konumun HAM veriyi ne kadar yakindan takip ettigi)
PARMAK_YENIDEN_TETIK_MIN_KARE = 6  # iki ayri tetiklenme arasi ARKA ARKAYA en az kac kare gecmeli (debounce DEGIL, kisa bir refractory sure)

EL_OLCEK_MIN = 0.01  # govde_olcek_hesapla'daki min_olcek tabani (el, omuzdan COK KUCUK oldugu icin ayri/daha kucuk bir taban)


EL_OLCEK_YUMUSATMA_ORANI = 0.25
EL_OLCEK_MAKS_SICRAMA = 0.01     # tek karede el_olcegi'nin degisebilecegi AZAMI miktar
EL_OLCEK_KABUL_MIN_ORAN = 0.6    # ham okuma, yumusatilmisin bu oranindan KUCUKSE yoksay
EL_OLCEK_KABUL_MAKS_ORAN = 1.6   # ham okuma, yumusatilmisin bu oranindan BUYUKSE yoksay


EL_BILEK_ESLESTIRME_MAKS_MESAFE = 0.15

PARMAK_OLAY_KESITI_AKTIF = False
PARMAK_SOL_KLASORU = VIDEO_KLASORU / "sol_parmak"
PARMAK_SAG_KLASORU = VIDEO_KLASORU / "sag_parmak"


# BlazePose (PoseLandmarker), kol/govde kadraj DISINDA kalsa bile 33 noktanin
# HEPSI icin bir tahmin uretir - sadece "visibility" skoru dusuk olur. Bu
# yuzden kol sayilirken ilgili noktalarin yeterince "gorunur" olmasi sart
# kosuluyor (bkz. gorsellik.gorunur_mu).
GORUNURLUK_ESIK = 0.35

# EKRANA GORE SOL/SAG (bkz. gorsellik.ekran_sol_sag_ayikla): True ise KOL VE
# BACAK sayaclari icin MediaPipe'in ANATOMIK LEFT_*/RIGHT_* etiketi TAMAMEN
# YOK SAYILIR, bunun yerine iki bilek/ayak-bilegi noktasindan hangisi
# EKRANDA daha SOLDA ise "SOL" sayilir. YUKARIDAN/BASUCUNDAN bakan, sirtustu
# yatan hasta kamera acisinda MediaPipe'in anatomik sol/sag atamasi
# GUVENILMEZ/TUTARSIZ cikabildigi gercek kullanici videosuyla dogrulandi -
# bu ayar acikken SOL/SAG HER ZAMAN ekranda gorunenle tutarli olur. Kamera
# NORMAL/ON'den bakan bir acida kullanilirsa (MediaPipe'in kendi atamasinin
# zaten guvenilir oldugu durum) False yapip MediaPipe'in ANATOMIK etiketine
# geri donebilirsin.
EKRANA_GORE_SOL_SAG = True

# True ise (bkz. gorsellik.ekran_etiket_ciz), sayaçların GERÇEKTEN hangi
# noktayı "SOL"/"SAĞ" saydığı, o noktanın TAM ÜSTÜNE küçük bir daire+yazı
# ile çizilir - MediaPipe'in kendi govde_ciz çizimi (ham/anatomik renklere
# göre) ile SAYAÇLARIN (ekrana göre düzeltilmiş) SOL/SAĞ tanımı FARKLI
# olabildiği için (bkz. EKRANA_GORE_SOL_SAG), bu ikisi karıştırılmasın diye.
EKRANA_GORE_ETIKET_GOSTER = True

ESIK_BLINK = 0.5

# --- Goz kirpintisi (OpenVINO gaze modeli icin) --------------------------------
# Sol/sag goz landmark noktalarindan cikarilan kutunun kac kat genisletilecegi
# - model, goz kapaklarinin biraz otesini de (kas/yanak baglami) gormek uzere
# egitilmis, bu yuzden tam goz aralıgindan biraz daha genis kesiyoruz.
GOZ_KIRPINTI_MARJI = 2.2

# --- OLAY KESITI (event clip) -------------------------------------------------
# SOL KOL / SAG KOL / KIRPMA / BAKIS sayaclarindan biri her artinca, o andan
# ONCEKI ve SONRAKI birkac saniyeyi birlikte tek bir MP4 olarak otomatik
# kaydeder (guvenlik kamerasi mantigi).
OLAY_ONCE_SANIYE = 2.0   # tetiklenmeden ONCEKI kac saniye dahil edilsin
OLAY_SONRA_SANIYE = 2.0  # tetiklendikten SONRAKI kac saniye dahil edilsin

# --- Yumusatma (EMA) ----------------------------------------------------------
# Yuz uzaklastikca/kuculdukce hem landmark hem gaze vektoru tahmini daha
# gurultulu oluyor, bu da oku titrek gosteriyordu. YUMUSATMA_ORANI
# KUCULTULURSE daha az titrer ama tepki gecikir; BUYUTULURSE daha hizli tepki
# verir ama daha titrek olur. 1.0 = yumusatma YOK.
YUMUSATMA_ORANI = 0.35

# SICRAMA (outlier) SINIRLAMASI: yuz kucukken/uzaktayken gaze vektorunun
# x/y bilesenlerinde bazen tek bir karede COK sapan bir tahmin ("sicrama")
# olusabiliyor - EMA yumusatma bunu yavaslatir ama tek basina yeterli degil.
# Bu yuzden EMA'dan ONCE, bir karede vektor bileseninin en fazla bu kadar
# degismesine izin veriyoruz - daha buyuk bir fark gelirse kirpip sadece bu
# kadarini kabul ediyoruz.
MAKS_BAKIS_SICRAMA = 0.35  # boyutsuz, vektor bileseni (-1..1) icin karede izin verilen maks degisim

VIDEO_FPS_VARSAYILAN = 15.0
# --- UZAK KAMERA: SABIT BOLGE ZOOM (17.08.2026 eklendi) --------------------

# NOT: bu SADECE gaze_birlesik_uzak.py tarafindan kullanilir - normal
# webcam'de calisan gaze_birlesik.py'ye (ve oradaki takip_yakinlastir/
# DIJITAL_YAKINLASTIRMA mantigina) HICBIR sekilde dokunulmadi/etkilenmedi.
BOLGE_NOKTALARI_DOSYASI = BURASI / "zoom_noktalari.json"

# Bolge basina VARSAYILAN zoom orani (nokta_sec.py'de +/- ile bolge bazinda
# degistirilip JSON'a kaydedilir) - takip_yakinlastir/DIJITAL_YAKINLASTIRMA
# ile AYNI anlamda: kirpma alani = kare boyutu / oran (kucuk kirpma alani =
# fazla zoom). Kamera ne kadar UZAKSA bu deger o kadar BUYUK olmali.
BOLGE_ZOOM_ORANI_VARSAYILAN = 6.0

# Bolunmus ekranda (izgara) her panelin piksel boyutu - kucultursen
# pencere daha az yer kaplar (goruntu kalitesini ETKILEMEZ, tespit YINE DE
# bolge_kirp'in kirptigi HAM cozunurluk uzerinden yapilir, bu SADECE
# EKRANDA GOSTERME/kayit boyutu).
BOLGE_PANEL_GENISLIK = 320
BOLGE_PANEL_YUKSEKLIK = 320





# CIHAZ SECIMI: bu makinede BIRDEN FAZLA mikrofon var (orn. "python -c
# "import sounddevice as sd; print(sd.query_devices())"" ile listelenebilir) -
# kamera USB'sindeki mikrofon (hastaya yakinsa DOGRU secim, "ALGILAMA" icin)
# ile bilgisayarin KENDI dahili mikrofonu (kullaniciya yakinsa "interkom
# konusma" icin daha DOGRU olabilir) FARKLI cihazlar olabilir. None = sistem
# varsayilani - fiziksel kurulumunuza gore SES_GIRIS_CIHAZI'ni ilgili cihazin
# INDEKSINE (yukaridaki listeden) ayarlayin.
SES_GIRIS_CIHAZI = None   # mikrofon (None = sistem varsayilani)
SES_CIKIS_CIHAZI = None   # hoparlor (None = sistem varsayilani)

SES_ORNEKLEME_HIZI = 16000  # Hz - konusma icin yeterli, dusuk CPU/gecikme
SES_BLOK_BOYUTU = 512       # kac ornekte bir callback tetiklenir (~32ms @16kHz)

# Ses seviyesi (RMS, 0..1 araligina yakin float32) esikleri - KOL/BACAK/KAFA
# ile AYNI giris/cikis-histerezis FIKRI (bkz. gorsellik.hareket_algila), YENI
# ("kare" degil zaman/saniye tabanli, cunku ses callback'i video karesinden
# BAGIMSIZ/farkli bir hizda calisir) bir debounce ile: "SES ALGILANDI" olayi
# gercek ses/konusma BASLADIGINDA bir kez sayilir, ayni ses SURERKEN tekrar
# tekrar sayilmaz. HENUZ gercek mikrofon kaydiyla KALIBRE EDILMEDI - cok
# hassassa (sessizlikte bile sayiyorsa) YUKSELT, kaciriyorsa DUSUR (ses.py
# gelecekte bir TANI/debug satirinda canli RMS degerini gosterebilir).
SES_ALGILAMA_ESIK = 0.02
SES_ALGILAMA_HISTEREZIS_ORANI = 0.5
SES_ALGILAMA_MIN_CIKIS_SANIYE = 0.3

# passthrough (mikrofon->hoparlor canli gecis) CIKISI bu RMS'in USTUNDEYKEN
# (yani kullanicinin sesi hoparlorden CALINIYORKEN) SES ALGILAMA o anlik
# olarak DURDURULUR - hoparlorden cikan kendi sesimizin mikrofona sizip
# "yatan kisi ses cikardi" sanilmasini (akustik geri besleme/yanlis pozitif)
# onlemek icin. Passthrough KAPALIYKEN bu kontrolun hicbir etkisi yok
# (cikis her zaman sessiz).
SES_PASSTHROUGH_ESIK = 0.01