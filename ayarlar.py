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
#
# YOGUN BAKIM SENARYOSU: hastanin bacaginda olusan HERHANGI bir hareketin
# (kucuk/kismi olsa bile) ATLANMAMASI, gec algilanmaktan cok daha onemli -
# bu yuzden asagidaki degerler bilincli olarak DUYARLILIK (recall) lehine,
# yanlis-pozitif (gereksiz sayim/klip) riskini goze alarak ayarlandi. 30
# gurultu senaryosu x 300 karelik simulasyonla dogrulandi: normal landmark
# titremesinde (~0.008-0.01 genlik) pratikte yanlis pozitif yok denecek kadar
# az (9000 karede ~5), gercek bir hareket (ani sicrama VEYA yavas gecis) 1-8
# kare (bir kac yuzde saniye) icinde yakalaniyor.
#
# GERI ALINDI (0.018 -> 0.035): telefon kamerasiyla (dusuk isik, farkli/daha
# gurultulu goruntu, daha dusuk fps) gercek testte esik=0.018 SAATLERCE
# HAREKETSIZ dururken bile SOL/SAG BACAK'i durmadan artiriyordu (gercek
# kullanici videosuyla dogrulandi - kisi hic kipirdamiyor ama sayaç
# artiyordu). Simulasyonla olculdu: 0.018 esikte, landmark titremesi
# 0.02 genlige ciktiginda (bu kameranin/isik seviyesinin urettigi gibi
# gorunuyor) 4500 karenin ~4300'unde yanlis-pozitif olusuyordu (%96!).
# esik=0.035 + hizli_oran=0.6 (eskiden 0.75) ile ayni testte 0.02 genlikte
# yanlis pozitif SIFIR, 0.025 genlikte bile cok az (~%1.6) cikti - gercek
# hareketleri (5cm+ hareket) hala 1-8 kare icinde yakaliyor. NOT: bu, "kolu
# bacak saniyor" seklinde YANLIS ANLASILABILIR - aslinda KOL VE BACAK
# BIRBIRINDEN BAGIMSIZ calisiyor (kod kontrol edildi, capraz baglanti YOK),
# sadece HER IKISI DE bu kamerada/isikta bagimsiz olarak surekli yanlis
# pozitif uretiyordu, kol oynatilinca TESADUFEN ayni ana denk geliyordu.
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
GOVDE_CIZIMI_GOSTER = False
EL_CIZIMI_GOSTER = False
YUZ_CIZIMI_GOSTER = True

# FaceLandmarker'in yuzu "yuz" olarak kabul etmesi icin gereken minimum
# guven skoru (0-1). Varsayilan MediaPipe degeri 0.5'tir, biz zaten 0.3'e
# indirmistik. TEPEDEN/DIREK KAMERA gibi asiri acili kurulumlarda (yuz
# normalden COK farkli/sikismis gorunuyor) bunu daha da dusurmek tespiti
# kolaylastirir - ama COK dusurursen yuz OLMAYAN seyleri de "yuz" sanma
# riski artar (yanlis pozitif). 0.1-0.15 arasi genelde makul bir sinir.
YUZ_TESPIT_ESIK = 0.15

# DIJITAL YAKINLASTIRMA: kare kirpilip eski boyutuna geri buyutulur - yuz
# kucuk/uzakta kaliyorsa (orn. direk/tepeden kamera) bunu artirmak
# MediaPipe'e daha buyuk/net bir yuz verir, tespiti kolaylastirir. 1.0 =
# yakinlastirma YOK (tam goruntu). 2.0 = YARISI (genislik VE yukseklikte)
# alinip tam kareye buyutulur - yani goruntu alani (FOV) 4'te bire duser.
#
# TAKIP EDEN yakinlastirma (bkz. gorsellik.takip_yakinlastir, uc birlesik
# dosyada da kullaniliyor): kirpma alani ARTIK SABIT merkezde degil, kilitli
# kisinin SON bilinen konumu etrafinda - kisi kadrajda nereye giderse
# gitsin (merkezde olmasa bile) yakinlastirilmis/detayli goruntude kalir.
# Kisi hic bulunamadigi/kaybedildigi surece (henuz kilit yok VEYA kimlik
# kilidi tamamen birakildi) yakinlastirma OTOMATIK olarak 1x'e (TAM
# GORUNTU, hic kirpma yok) doner - boylece sistem kisiyi ARAMAK icin dar/
# yakinlastirilmis bir alana degil, kameranin gordugu HER SEYE bakar.
# Kilitlenir kilitlenmez asagidaki oran devreye girer.
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

# ESKI (ARTIK KULLANILMIYOR - bkz. asagidaki KOL_HAREKET_* ve proje gecmisi):
# "kol kalkik mi" YONE OZGU kontrolu icin gorsellik.kol_aktif_mi hala
# TANIMLI (referans/geriye-uyum icin, dijital_yakinlastir gibi) ama artik
# HICBIR birlesik dosya bunu CAGIRMIYOR - SIRTUSTU YATAN bir hastada kol
# govde uzerinde/yana dogru hareket ederken bilek omuzden "yukari kalkmadigi"
# icin bu yontem gercek kol hareketlerinin COGUNU KACIRIYORDU (gercek
# videoyla dogrulandi - ayni sorunun bacakta da yasandigi, "yatarken bacak
# kaldiramama" gerekcesiyle daha once BACAK icin coz ulmustu, simdi KOL da
# AYNI coz ume tasindi).
KOL_Y_ESIK = 0.08       # bilek-omuz y farki bu degerden KUCUKSE "ayni hizada" sayilir
DIRSEK_ACI_ESIK = 90.0  # derece - dirsek acisi bu degerin ALTINDAYSA kol "kivrik" sayilir
KOL_HISTEREZIS_Y = 0.04
KOL_HISTEREZIS_ACI = 15.0
KOL_MIN_CIKIS_KARE = 8

# --- YENI: Kol hareketi (bkz. gorsellik.hareket_algila) -------------------
# BACAK_HAREKET_* ile TAM AYNI mantik/parametreler, ayni fonksiyon - "kol
# kalkik mi" gibi YONE OZGU bir kural DEGIL, bilek konumundaki HERHANGI BIR
# yone dogru (ani sicrama VEYA yavas/surekli hareket) yeterince buyuk bir
# degisikligi "hareket" sayar. Boylece hasta SIRTUSTU yatip kolunu govdesi
# uzerinde/yana dogru hareket ettirse bile (bilek hicbir zaman omuzden
# "yukari kalkmasa" bile) yakalanir - gercek kullanici videosuyla dogrulandi
# (eski yone-ozgu yontem bu senaryoda kol hareketlerinin COGUNU kaciriyordu).
# GERI ALINDI - BACAK_HAREKET_ESIK ile AYNI gerekce (bkz. yukarisi): telefon
# kamerasiyla dusuk isikta 0.018 hareketsizken bile surekli yanlis pozitif
# uretiyordu (KOL VE BACAK BAGIMSIZ calisiyor - bkz. BACAK_HAREKET_ESIK
# aciklamasi, "kol bacagi karistiriyor" gibi gorunen sey aslinda ikisinin de
# ayni kamerada bagimsiz olarak asiri duyarli olmasiydi).
KOL_HAREKET_ESIK = 0.035
KOL_HAREKET_HISTEREZIS_ORANI = 0.55
KOL_HAREKET_HIZLI_ORAN = 0.6
KOL_HAREKET_YAVAS_ORAN = 0.035
KOL_HAREKET_MIN_CIKIS_KARE = 8

# --- YENI (SADECE gaze_birlesik.py): GOVDEYE GORELI kol/bacak hareketi ---
# !!! ESKI YONTEMIN (yukaridaki KOL_HAREKET_*/BACAK_HAREKET_*, MUTLAK/kare-
# ici konum kullanan) YERINE gaze_birlesik.py'de KULLANILIYOR - l2cs_birlesik.py
# ve uniface_birlesik.py HALA eski yontemi kullaniyor, onlara dokunulmadi. !!!
#
# SORUN: MUTLAK konum kullanildiginda, kol VE bacak sayaçlari KOD OLARAK
# birbirinden bagimsiz olsa bile, govde/yatak/kamera EN UFAK sekilde
# kaydiginda (orn. kolunu guclu oynatan bir hastanin govdesi hafifce
# yaslaniyor/kayiyor, ya da telefon kamerasi titriyor) TUM landmarklar
# (bilek DE ayak bilegi DE) AYNI YONDE kayiyor - "kolu oynatinca bacak
# sayaci da artiyor, bacagi oynatinca kol sayaci da artiyor" seklinde
# CAPRAZ yanlis tetiklenmeye yol aciyordu (gercek kullanici geri
# bildirimiyle bulundu).
#
# COZUM: bkz. gorsellik.govdeye_goreli_konum - bilek/ayak bilegi, MUTLAK
# konumu yerine AYNI TARAF omuz/kalcaya GORE, govde olcegine (omuz
# genisligi) BOLUNMUS konumuyla izleniyor. Govde-geneli kaymalar boylece
# MATEMATIKSEL OLARAK iptal olur (hem uzuv hem referans ayni miktarda
# kayar, aralarindaki fark degismez) - SADECE uzvun govdeye GORE gercekten
# hareket etmesi sinyal uretir. 4 senaryoyla (govde kaymasi TEK BASINA,
# sadece kol hareketi, sadece bacak hareketi, kol hareketi+govde kaymasi
# BIRLIKTE - ASIL SORUN SENARYOSU) simulasyonla dogrulandi: hepsinde
# BEKLENEN sayaç (ve SADECE o sayaç) artti, capraz tetiklenme SIFIRA
# indi. Ayni gurultu/titreme/debounce testleri de tekrarlandi (bkz.
# hareket_algila docstring'i) - ayni saglamlik korunuyor.
#
# Ayni fonksiyon (hareket_algila) kullaniliyor, sadece x,y girdisi MUTLAK
# konum yerine bu goreli/olceklendirilmis konum - esik degerleri de buna
# gore YENIDEN kalibre edildi (birim artik "govde olcegi", 0.22 ~= omuz
# genisliginin %22'si kadar bir yer degistirme).
KOL_HAREKET_GORELI_ESIK = 0.22
KOL_HAREKET_GORELI_HISTEREZIS_ORANI = 0.55
KOL_HAREKET_GORELI_HIZLI_ORAN = 0.6
KOL_HAREKET_GORELI_YAVAS_ORAN = 0.035
KOL_HAREKET_GORELI_MIN_CIKIS_KARE = 8

BACAK_HAREKET_GORELI_ESIK = 0.22
BACAK_HAREKET_GORELI_HISTEREZIS_ORANI = 0.55
BACAK_HAREKET_GORELI_HIZLI_ORAN = 0.6
BACAK_HAREKET_GORELI_YAVAS_ORAN = 0.035
BACAK_HAREKET_GORELI_MIN_CIKIS_KARE = 8

# --- GOVDE OLCEGI YUMUSATMA (bkz. gorsellik.govde_olcek_hesapla) ----------
# SORUN (gercek kullanici videosuyla bulundu, 17.08.2026): govde_olcek SADECE
# omuzlardan (2 nokta) hesaplaniyor ve KOL + BACAK'IN DORDU DE (sol/sag)
# AYNI govde_olcek'e BOLUNUYOR (bkz. govdeye_goreli_konum). Omuz landmark'i
# TEK bir karede (hareket bulanikligi/kisa sureli hatali tahmin gibi
# nedenlerle) HAM/gurultulu cikinca govde_olcek ANLIK KUCULUYOR - bu da
# PAYDA ortak oldugu icin KOL SOL + KOL SAG + BACAK SOL + BACAK SAG
# mesafelerinin HEPSININ AYNI ANDA (gercek hicbir uzuv hareket etmemisken
# bile) esigin COK UZERINE FIRLAMASINA yol aciyordu - kullaniciya "kolumu
# oynatinca bacak sayiyor, bacagi oynatinca kol sayiyor" gibi bir CAPRAZ
# tetiklenme YANILSAMASI olarak gorunuyor, ama aslinda kol/bacak degiskenleri
# KARISMIYOR - ikisi de ayni (o an gurultulu) ORTAK PAYDAYI paylastigi icin
# BIRLIKTE sicriyorlar. COZUM: govde_olcek'i (diger noktalar gibi) EMA ile
# yumusat + tek karedeki degisimi sinirla (bkz. gorsellik.yumusat) - boylece
# tek bir gurultulu omuz karesi ANINDA payda'yi carpitip DORT sayaci BIRDEN
# yanlis tetiklemez.
GOVDE_OLCEK_YUMUSATMA_ORANI = 0.2
GOVDE_OLCEK_MAKS_SICRAMA = 0.02  # tek karede govde_olcek'in degisebilecegi AZAMI miktar

# EK KORUMA (17.08.2026, ikinci bulgu): kolu YANA kaldirinca dogru sayiliyor
# ama YUKARI (govdeden kalkip KAMERAYA DOGRU / tepeye dogru) kaldirinca BACAK
# sayaci artiyordu. Sebep: bu hareket omuz/bilek landmark'ini TEK kareden
# COK DAHA UZUN (defalarca kare, ~yarim-bir saniye) SURELI bozuyor - kamera
# TAM TEPEDEN baktigi icin "kol dogrudan kameraya/tavana dogru kalkmasi" bu
# modelin egitim dagiliminda neredeyse hic yok, MediaPipe bu poz icin
# gerceκci olmayan/oynak bir tahmin verebiliyor. Yukaridaki MAKS_SICRAMA
# (tek karelik sinir) boyle UZUN SURELI bir bozulmaya karsi yetersiz kaliyor
# (adim adim BIRIKEREK yine de gercek degerden uzaklasabiliyor). Bu yuzden
# AYRICA bir "makul aralik disi ise bu okumayi TAMAMEN YOKSAY (hic
# harmanlama, hic adim atma)" filtresi eklendi - bkz. gaze_birlesik.py.
GOVDE_OLCEK_KABUL_MIN_ORAN = 0.6  # ham okuma, yumusatilmisin bu oranindan KUCUKSE yoksay
GOVDE_OLCEK_KABUL_MAKS_ORAN = 1.6  # ham okuma, yumusatilmisin bu oranindan BUYUKSE yoksay

# --- Parmak hareketi (17.08.2026 eklendi, SADECE gaze_birlesik.py) --------
# AMAC: hastanin parmaklarini oynatip oynatmadigini (kaba kol/bacak
# hareketinden BAGIMSIZ, ince motor tepkisi) yakalamak - orn. "eli tut, elini
# sik" gibi bir emre parmak duzeyinde tepki var mi diye.
#
# TASARIM (kol/bacak'taki AYNI derslerle): 5 parmak ucunun (basparmak,
# isaret, orta, yuzuk, serce) ORTALAMA konumu, o ELIN KENDI BILEGINE GORE
# (govdeye_goreli_konum ile AYNI fonksiyon, burada "govde" yerine "el" icin
# kullaniliyor - matematiksel olarak birebir ayni islem) ve el buyuklugune
# (bilek-orta parmak kok mesafesi) OLCEKLENEREK izlenir. BOYLECE:
#   - Kol/govde hareketinden BAGIMSIZDIR (bilege GORE oldugu icin kolu
#     oynatmak parmak sayacini ARTIRMAZ - kol/bacak'ta yasanan CAPRAZ
#     tetiklenme dersi buraya da uygulandi).
#   - Kisinin kameraya uzakligindan BAGIMSIZDIR (el buyuklugune bolundugu
#     icin).
# HandLandmarker'in kendi Sol/Sag (handedness) etiketine GUVENILMIYOR (pose
# icin ekran_sol_sag_ayikla'da bulunan AYNI sorun burada da olasi) - bkz.
# gaze_birlesik.py'deki el-govde eslestirme mantigi.
#
# TEK PARMAK DUYARLILIGI (17.08.2026 eklendi): ilk tasarimda 5 parmak
# ucunun ORTALAMASI tek bir sinyale indirgeniyordu - SADECE BIR parmak
# oynadiginda bu hareket ortalamaya girip 1/5'e "sulanip" esigin altinda
# kalabiliyordu (kullanicinin acik istegi: "parmaklardan biri bile hafif
# oynayinca algilayabilelim"). COZUM: artik HER PARMAK UCU KENDI BAGIMSIZ
# durumuyla AYRI AYRI izleniyor (bkz. gaze_birlesik.py parmak_durum), el
# HERHANGI BIR ucun esigi GECMESIYLE "hareketli" sayilir (5'in OR'u,
# ortalamasi DEGIL).
#
# "ASAGI INIP TEKRAR KALKMASINA GEREK YOK" (17.08.2026, ikinci istek):
# yukaridaki hareket_algila (KOL/BACAK'ta kullanilan, iki-EMA HIZLI/YAVAS
# farki) parmak icin YANLIS ARAC oldugu ortaya cikti - o fonksiyon "hareketli"
# durumdan cikip YENIDEN tetiklenebilmek icin mesafenin cikis esiginin
# ALTINA DUSMESINI sart kosuyor. El HAVADA/kalkik TUTULURKEN (govdeye/bilege
# GORE "yeni" bir taban konumda) art arda gelen KUCUK ek kipirdanmalar bu
# YAVAS EMA'nin o yeni tabana dogru surunmesi yuzunden esigi bir daha HIC
# GECEMEYEBILIYORDU (simulasyonla dogrulandi: varsayilan ayarlarla el kalkik
# tutulurken sadece ILK kalkis sayiliyor, sonraki 7 ayri hafif kipirdanmanin
# HICBIRI sayilmiyordu). COZUM: parmak icin FARKLI bir teknige gecildi - bkz.
# gorsellik.parmak_hareket_algila. Bu fonksiyon "yavas referansa GORE mesafe"
# yerine "yumusatilmis konumun KENDI ANLIK degisim HIZINA" bakar - bu deger
# hareket duruncа birkac karede SIFIRA doner (asagi inmesi/eski konuma
# donmesi GEREKMEZ), sadece KISA bir yeniden-tetiklenme bekleme suresi
# (PARMAK_YENIDEN_TETIK_MIN_KARE) vardir. 250 karelik bir simulasyonla
# dogrulandi: el kalkik tutulurken 7 ayri hafif kipirdanmanin HEPSI ayri ayri
# yakalaniyor (once: sadece 1/8), parmaklar TAMAMEN sabitken (80 senaryo x
# 300 kare, gercekci landmark titremesi) SIFIR yanlis pozitif.
PARMAK_HIZ_ESIK = 0.06           # ardisik iki karedeki (yumusatilmis) konum degisimi bu kadar olursa "hareket"
PARMAK_HIZ_HIZLI_ORAN = 0.6      # yumusatma orani (konumun HAM veriyi ne kadar yakindan takip ettigi)
PARMAK_YENIDEN_TETIK_MIN_KARE = 6  # iki ayri tetiklenme arasi ARKA ARKAYA en az kac kare gecmeli (debounce DEGIL, kisa bir refractory sure)

EL_OLCEK_MIN = 0.01  # govde_olcek_hesapla'daki min_olcek tabani (el, omuzdan COK KUCUK oldugu icin ayri/daha kucuk bir taban)

# Bir eldeki tek bilek, HANGI govde tarafina (sol_bilek/sag_bilek, POSE'dan)
# ait sayilsin diye eslestirilirken izin verilen AZAMI ekran mesafesi
# (normalize, 0..1) - bundan UZAKSA eslestirme GUVENILMEZ sayilip o el o
# karede ATLANIR (yanlis sol/sag atamasi yapmaktansa o kareyi kacirmak
# tercih edildi - proje boyunca benimsenen "supheliyse atla" ilkesi).
EL_BILEK_ESLESTIRME_MAKS_MESAFE = 0.15

PARMAK_OLAY_KESITI_AKTIF = False
PARMAK_SOL_KLASORU = VIDEO_KLASORU / "sol_parmak"
PARMAK_SAG_KLASORU = VIDEO_KLASORU / "sag_parmak"

# --- GCS Motor Tepkisi Testi (M2-M5) --------------------------------------
# !!! SEZGISEL/HEURISTIC - KESIN TIBBI OLCUM DEGIL !!! Bkz. gorsellik.
# gcs_kol_tepkisini_sinifla docstring'i - sonuc bir klinisyenin KENDI
# GOZLEMIYLE dogrulamasi gereken bir ON-ONERI, otomatik/nihai skor DEGIL.
#
# 'g' tusuyla baslatilir - MERKEZI (sternal ovma/supraorbital baski tarzi)
# bir agrili uyaran verildigini VARSAYAR (uyarani klinisyen KENDISI
# uygular, 'g' sadece "uyaran SIMDI verildi" anini isaretler). Ardindan
# asagidaki sure boyunca HER IKI kolun tepkisi ayri ayri gozlenir.
GCS_PENCERE_SANIYE = 4.0
# Bu kadar (normalize, 0..1) toplam bilek yer degistirmesi OLMAZSA
# "hareketsiz" (M1 adayi) sayilir.
GCS_HAREKETSIZ_ESIK = 0.05
# Bilek, pencere basindaki omuz (~kopruckkemigi) seviyesinin bu KADAR
# YUKARISINA (normalize y, kucuk deger = ekranda yukari) cikarsa "uyarana
# ulasmaya calisiyor" (lokalize ediyor, M5) sayilir.
GCS_LOKALIZE_PAY = 0.03
# Dirsek acisi baslangica gore EN AZ bu kadar (derece) KUCULURSE (asiri
# bukulme) VE bilek klavikula seviyesine ulasamadiysa -> M3 (anormal
# fleksiyon / dekortike postur).
GCS_DEKORTIKE_DEGISIM_ESIK = 35.0
# Dirsek acisi baslangica gore EN AZ bu kadar (derece) BUYURSE (daha da
# gerilme/acilma) -> M2 (anormal ekstansiyon / deserebre postur).
GCS_DESEREBRE_DEGISIM_ESIK = 25.0
# Test bitince sonuc ekranda kac saniye gosterilsin.
GCS_SONUC_GOSTERIM_SANIYE = 10.0

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
# SENARYO: webcam yerine hastadan UZAKTA/sabit duran (orn. tavana/koseye
# monteli) bir kamera kullaniliyor - govde/kol/bacak genis acidan hala
# gorulebiliyor ama YUZ ve PARMAK gibi INCE detaylar kadrajda cok kucuk
# kaliyor, MediaPipe bunlari guvenilir tespit edemiyor.
#
# COZUM: bu bolgelerin EKRANDAKI konumu SABIT/degismiyor (kamera da hasta da
# hareket ETMIYOR - takip_yakinlastir'daki gibi bir kisiyi "takip etmeye"
# GEREK YOK) - bu yuzden konumlarini BIR KEZ, elle (nokta_sec.py ile fare
# tiklayarak) isaretleyip ayarlar.BOLGE_NOKTALARI_DOSYASI'na kaydediyoruz.
# gaze_birlesik_uzak.py (bkz. o dosyanin basindaki aciklama) HER karede bu
# SABIT noktalarin etrafini kirpip buyutur (zoom yapar) ve tespiti DOGRUDAN
# bu buyutulmus panel uzerinde calistirir - kucuk/uzak goruntude kaybolan
# yuz/parmak hareketleri boylece yakalanir. Panel basina AYRI bir pencerede
# ("bolunmus ekran") gosterilir; parmak icin ayri, yuz icin ayri panel.
#
# NOT: bu SADECE gaze_birlesik_uzak.py tarafindan kullanilir - normal
# webcam'de calisan gaze_birlesik.py'ye (ve oradaki takip_yakinlastir/
# DIJITAL_YAKINLASTIRMA mantigina) HICBIR sekilde dokunulmadi/etkilenmedi.
BOLGE_NOKTALARI_DOSYASI = BURASI / "zoom_noktalari.json"

# Bolge basina VARSAYILAN zoom orani (nokta_sec.py'de +/- ile bolge bazinda
# degistirilip JSON'a kaydedilir) - takip_yakinlastir/DIJITAL_YAKINLASTIRMA
# ile AYNI anlamda: kirpma alani = kare boyutu / oran (kucuk kirpma alani =
# fazla zoom). Kamera ne kadar UZAKSA bu deger o kadar BUYUK olmali.
BOLGE_ZOOM_ORANI_VARSAYILAN = 7.0

# Bolunmus ekranda (izgara) her panelin piksel boyutu - kucultursen
# pencere daha az yer kaplar (goruntu kalitesini ETKILEMEZ, tespit YINE DE
# bolge_kirp'in kirptigi HAM cozunurluk uzerinden yapilir, bu SADECE
# EKRANDA GOSTERME/kayit boyutu).
BOLGE_PANEL_GENISLIK = 320
BOLGE_PANEL_YUKSEKLIK = 320

# --- EL BILEGI YUMUSATMA (SADECE gaze_birlesik_uzak.py bolge modu, ---------
# 18.08.2026 eklendi, gercek kullanici videosuyla teshis edildi) -----------
# SORUN: "el" bolgesi COK SIKI kirpilirsa (yuksek BOLGE_ZOOM_ORANI/oran),
# BILEK (HandLandmark 0) bazen kirpma alaninin DISINDA/kenarinda kalir -
# MediaPipe yine de bir tahmin URETIR ama bu tahmin kareden kareye COK
# oynak/guvenilmez olabilir. govdeye_goreli_konum HEM referans (bilek) HEM
# olcek (bilek - orta parmak kok mesafesi, el_olcegi) icin bu noktayi
# kullandigindan, bilek tahmini oynadikca 5 parmak ucunun TUMU (parmaklar
# GERCEKTE hic kipirdamamis olsa bile) AYNI ANDA sicriyor - ekranda "HIZ"
# degeri esigin COK UZERINE (orn. 0.3-1.3, esik 0.12 iken) FIRLADIGI
# gercek kullanici videosuyla dogrulandi (bkz. proje sohbet gecmisi,
# 18.08.2026). Bu TAM OLARAK govde_olcek'te (bkz. yukaridaki GOVDE_OLCEK_
# YUMUSATMA_ORANI aciklamasi) daha once cozulen "ortak/gurultulu payda TUM
# uzuvlari BIRDEN yanlis tetikliyor" sorununun EL surumu.
#
# COZUM: bilek konumu (referans noktasi) da govde_olcek gibi EMA ile
# yumusatiliyor (bkz. gaze_birlesik_uzak.py) - hem el_olcegi hem her parmak
# ucunun goreli konumu artik HAM/tek-karelik bilek tahmini yerine bu
# YUMUSATILMIS bilege gore hesaplaniyor.
#
# ONEMLI: bu sadece bir GUVENLIK AGI - ASIL/kalici cozum nokta_sec.py'de
# bolgeyi bilek DE goruntude/kirpma alaninda kalacak sekilde isaretlemek
# (gerekirse o bolge icin '-' ile zoom oranini biraz azaltip bilegi de
# kadraja almak).
EL_OLCEK_YUMUSATMA_ORANI = 0.1   # GOVDE_OLCEK_YUMUSATMA_ORANI'ndan (0.2) DAHA agresif - deneme
EL_OLCEK_MAKS_SICRAMA = 0.008    # tek karede bilek konumunun (normalize, panel-ici) degisebilecegi AZAMI miktar - deneme