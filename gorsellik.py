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


def ekran_sol_sag_ayikla(sol_aday, sag_aday):
    """MediaPipe'in LEFT_*/RIGHT_* (anatomik) etiketine KOR KORUNE guvenmek
    yerine, iki noktadan (orn. LEFT_ANKLE/RIGHT_ANKLE veya LEFT_WRIST/
    RIGHT_WRIST) hangisi EKRANDA/goruntude daha SOLDA (x kucuk) ise onu
    "sol", digerini "sag" olarak dondurur.

    NEDEN: MediaPipe'in "sol/sag" atamasi, ANATOMIK (kisinin KENDI sol/sagi)
    olacak sekilde tahmin ediliyor - bu, modelin cogunlukla egitildigi
    "kisi kameraya ON'den bakiyor, ayakta/oturuyor" senaryosunda genelde
    guvenilir. Ama BU projede kullanilan YUKARIDAN/BASUCUNDAN bakan,
    SIRTUSTU yatan hasta kamera acisi bu egitim dagilimindan COK farkli -
    gercek kullanici videosuyla (kare kare, sayaç artislarini gorsel
    hareketle karsilastirarak) dogrulandi ki bu acida modelin sol/sag
    atamasi GUVENILMEZ VE TUTARSIZ olabiliyor (ayni kayit icinde bile
    degisebiliyor) - bu bir MediaPipe/proje sinirliligi, bizim sayma
    mantigimizin hatasi degil.

    COZUM: MediaPipe'in etiketini TAMAMEN YOK SAYIP sadece EKRANDAKI (x
    koordinati) konuma gore ata - boylece "SOL BACAK"/"SAG BACAK" HER ZAMAN
    ekranda gorunenle (bakan kisinin sezgisiyle) tutarli olur, modelin ic
    anatomik-etiket tutarsizligindan ETKILENMEZ. Bkz. ayarlar.
    EKRANA_GORE_SOL_SAG (True/False ile ac/kapa).

    Donus: (ekran_solu, ekran_sagi).
    """
    if sol_aday.x <= sag_aday.x:
        return sol_aday, sag_aday
    return sag_aday, sol_aday


def ekran_etiket_ciz(kare, nokta, etiket, renk, w, h):
    """SAYAÇLARIN (ekrana_gore_sol_sag_ayikla SONRASI) GERCEKTEN hangi
    noktayi "SOL"/"SAG" saydigini kucuk bir daire + yazi ile o noktanin
    ustune ciziyor.

    NEDEN GEREKLI: govde_ciz() / MediaPipe'in KENDI cizim fonksiyonu, HAM
    (anatomik, bu kamera acisinda GUVENILMEZ olabilen - bkz.
    ekran_sol_sag_ayikla docstring'i) LEFT_*/RIGHT_* kimligine gore
    renklendiriyor - ekrana-gore duzeltmeyi YANSITMIYOR. Yani cizilen
    govde SOL/SAG rengiyle SAYAÇLARIN SOL/SAG'i FARKLI SEYLERE dayanabilir,
    bu da ekrana bakan kisiye "landmark'lar yer degistirmis" gibi
    gorunebilir (gercek kullanici geri bildirimiyle bulundu). Bu fonksiyon
    o karisikligi gidermek icin SAYACIN KULLANDIGI (duzeltilmis) noktanin
    TAM USTUNE acik bir "SOL"/"SAG" yazisi koyar.
    """
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
    """Bir "agrili uyaran" penceresi (bkz. ayarlar.GCS_PENCERE_SANIYE) boyunca
    TEK bir kol icin toplanan (dirsek_acisi, bilek_x, bilek_y) orneklerinden
    Glasgow Koma Skalasi motor tepkisini (M2-M5) TAHMIN ETMEYE calisan bir
    SEZGISEL (heuristic) siniflandirici.

    !!! ONEMLI/GUVENLIK UYARISI !!!: Bu KESIN bir tibbi olcum ARACI DEGIL.
    Tek kameradan, 2D landmark konumlarina dayanan bir yaklasim - dokunsal
    geri bildirim, kas tonusu, hastanin genel klinik durumu gibi gercek
    muayenenin icerdigi hicbir seyi degerlendiremiyor. Sonuc, bir
    klinisyenin KENDI GOZLEMIYLE DOGRULAMASI/DUZELTMESI gereken bir
    ON-ONERI olarak dusunulmeli - otomatik/nihai bir GCS-M skoru olarak
    ASLA kullanilmamali.Ozellikle M1 (hic tepki yok) sonucu, "hasta
    gercekten tepkisiz" ile "kamera/gorunurluk kolu izleyemedi" arasinda
    AYRIM YAPAMAZ - bu yuzden cagiran kod "sebep" alanini da rapor eder.

    Kullanilan sezgi (MERKEZI/sternal-tarzi bir uyaran varsayilir - bkz.
    dosya/proje notlari, klinikte de yaygin kullanilan "el kopruckkemigi
    seviyesinin ustune cikiyor mu" testi):
      - Bilek, PENCERE BASINDAKI omuz (~kopruckkemigi) seviyesinin EN AZ
        GCS_LOKALIZE_PAY kadar USTUNE cikarsa (uyarana "ulasmaya" calisiyor)
        -> M5 (lokalize ediyor).
      - Dirsek, baslangica gore EN AZ GCS_DEKORTIKE_DEGISIM_ESIK kadar
        KUCULURSE (asiri/anormal fleksiyon) ama bilek o seviyeye
        ulasamadiysa -> M3 (anormal fleksiyon / dekortike postur).
      - Dirsek, baslangica gore EN AZ GCS_DESEREBRE_DEGISIM_ESIK kadar
        BUYURSE (daha da gerilme/acilma) -> M2 (anormal ekstansiyon /
        deserebre postur).
      - Bunlarin hicbiri degil ama GCS_HAREKETSIZ_ESIK'i asan belirgin bir
        hareket VARSA -> M4 (cekiyor / withdrawal).
      - Belirgin hareket yoksa -> None, sebep="hareketsiz" (M1 ADAYI).
      - Kol bu pencerede hic yeterince gorunmediyse -> None, sebep="izlenemedi".

    Donus: (etiket, detaylar) - etiket "M5"/"M4"/"M3"/"M2"/None, detaylar
    sozlugu ham olculen degerleri (ve varsa "sebep") icerir - EKRANDA/
    konsolda gosterilip klinisyenin kendi degerlendirmesine seffaf bir
    dayanak sunmasi icin.
    """
    if not ornekler or baslangic_dirsek_acisi is None or omuz_baslangic_y is None:
        return None, {"sebep": "izlenemedi"}

    acilar = [o[0] for o in ornekler]
    xler = [o[1] for o in ornekler]
    yler = [o[2] for o in ornekler]

    min_acisi = min(acilar)
    maks_acisi = max(acilar)
    min_bilek_y = min(yler)
    yer_degistirme = math.hypot(max(xler) - min(xler), max(yler) - min(yler))
    # Dirsek acisindaki EN BUYUK degisim (ister fleksiyon ister ekstansiyon
    # yonunde) - SADECE bilek yer degistirmesine bakarsak, dirsek govdeye
    # yakin bir eksende bukulup bilek pek fazla mutlak konum degistirmeden
    # de asiri fleksiyon/ekstansiyon (M3/M2) olusabilir - bu yuzden
    # "hareketsiz" kararini ASAGIDA HEM yer_degistirme HEM aci_degisimine
    # bakarak veriyoruz (biri bile yeterince buyukse hareketsiz DEGILDIR).
    aci_degisimi = max(baslangic_dirsek_acisi - min_acisi, maks_acisi - baslangic_dirsek_acisi)

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
    """Kol "kalkik" ya da "kivrik" mi - HISTEREZISLI (Schmitt trigger).

    BACAK'taki hareket_algila ile AYNI DEBOUNCE fikri (bkz. o fonksiyonun
    docstring'i - bacagi kaldirip HAVADA TUTARKEN kas titremesi, dar bir
    histerezis bandiyla birlesince AYNI hareketi birden fazla kez
    saydirabiliyordu): kol da kaldirilip TUTULURKEN benzer bir titreme riski
    var, o yuzden AYNI korumayi burada da uyguluyoruz - onceden AKTIF olan
    kol, "aktif degil" durumuna gecmek icin TEK karede degil, ARKA ARKAYA en
    az min_cikis_kare kare boyunca kosulun disinda kalmasi gerekiyor (araya
    bir titreme karesi girerse sayac sifirlanir). min_cikis_kare=0 (varsayilan)
    ile eski (debounce'suz) davranisla TAM AYNI calisir - geriye donuk uyumlu.

    Donus: (aktif, yeni_cikis_kare_sayaci).
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
    """Bir noktanin (orn. ayak bilegi) HAREKET halinde olup olmadigini IKI
    FARKLI HIZDA EMA (HIZLI ve YAVAS) arasindaki farktan bulur - borsa
    grafiklerindeki MACD (iki hareketli ortalama farki) ile AYNI fikir.
    KOL'daki "yukari kalkti mi" gibi YONE OZGU bir kural DEGIL - herhangi bir
    YONE dogru (ani SICRAMA veya yavas/surekli kayma FARK ETMEZ) yeterince
    buyuk bir konum degisikligini yakalar, bu yuzden yatan/oturan hastada da
    calisir.

    UCUNCU SORUN (ilk ikisi yukarida anlatiliyor) - YOGUN BAKIM icin
    DUYARLILIGI artirdiktan (esik dusuruldu, histerezis bandi daraltildi)
    SONRA ORTAYA CIKTI: bacak KALDIRILIP HAVADA TUTULURKEN dogal kas
    titremesi/dengeleme, artik DAR olan giris/cikis esik bandini birkac kez
    ileri-geri gecip AYNI TEK hareketi 2-3 kez saydiriyordu (gercek kullanici
    testiyle dogrulandi). Cozum KLASIK bir DEBOUNCE: "hareketli" durumundan
    "hareketsiz"e gecmek icin mesafenin TEK bir karede degil, ARKA ARKAYA
    EN AZ min_cikis_kare kare boyunca cikis esiginin ALTINDA KALMASI sart
    kosuluyor (cikis_kare_sayaci bunu sayar, araya bir tremor karesi girerse
    sifirlanir). Boylece kisa sureli titremeler "hareket bitti" saymiyor,
    AYNI hareket tek sayiliyor - ama gercekten AYRI bir sonraki hareket
    (birkac saniye sonra) yine normal sekilde yeni bir sayim olarak
    algilaniyor (min_cikis_kare, giris esigine gore KISA tutuluyor).

    NEDEN IKI EMA (TEK EMA DENENDI, IKI AYRI SORUNA YOL ACTI):
      1) "Durgun referans SADECE hareketsizken guncellenir" (ilk tasarim):
         bir hareket olunca referans ESKI konumda SONSUZA DEK kalir (cunku
         "hareketli" bir daha hic False olamiyor), sonraki hareketler HIC
         sayilmiyor.
      2) "Durgun referans HER ZAMAN guncellenir" (ilk duzeltme): bu sefer
         YAVAS/SUREKLI bir hareket sirasinda (orn. bacagini 1-2 saniyede
         yavasca cekmek) referans hareketi neredeyse ANINDA "takip edip"
         mesafeyi hicbir zaman esigi asacak kadar buyutemiyordu - gercek
         bacak hareketleri (ani sicrama degil, kademeli hareket) SAYILMIYORDU
         (bkz. proje gecmisi, gercek videoyla dogrulandi).
      Cozum: HIZLI EMA (soke yakin, HAM sinyali neredeyse birebir takip eder,
      sadece tek karelik landmark titremesini yumusatir) ile YAVAS EMA
      (uzun sureli "dinlenme" konumunu temsil eder, ~1-2 saniyede yakalar)
      arasindaki fark olculur. Ani sicramada HIZLI hemen zipliyor, YAVAS
      geride kaliyor -> fark buyuyor. Yavas/surekli hareket de HIZLI
      HAM sinyale yakin gittigi icin ayni mantikla fark yine buyuyor -
      HER IKI durumda da yakalanir. Hareket durunca HIZLI sabitlenir, YAVAS
      birkac saniyede ona yetisir, fark tekrar sifira duser -> bir sonraki
      hareket icin otomatik "yeniden kurulur" (SONSUZA DEK KILITLENME YOK).

    - hizli_x/y, yavas_x/y: BIR ONCEKI karenin HIZLI ve YAVAS EMA konumlari
      (hizli_x=None ise ilk kare, henuz referans yok).
    - x, y: BU karenin HAM (yumusatilmamis) konumu.
    - onceki_hareketli: bir onceki karede HAREKETLI miydi (histerezis icin,
      Schmitt trigger - giris esigi > cikis esigi - ayni hareketin ART ARDA
      birden fazla kez sayilmasini onler).
    - esik: giris esigi (HIZLI-YAVAS farki bu kadar olunca "hareket" baslar).
    - histerezis_orani: cikis esigi = esik * bu oran (KUCUK, fark bu kadara
      dusunce "hareket" biter) - giris esiginden KUCUK olmali.
    - hizli_oran / yavas_oran: iki EMA'nin HAM konuma yaklasma hizlari (EMA
      orani) - hizli_oran >> yavas_oran olmali, aradaki fark ne kadar
      buyukse ayrisma o kadar belirgin/hizli olusur.
    - cikis_kare_sayaci: BIR ONCEKI karenin "arka arkaya cikis esigi altinda
      kalinan kare sayisi" durumu (debounce icin, ilk kare icin 0 verilir).
    - min_cikis_kare: "hareketli"den "hareketsiz"e gecmek icin mesafenin
      ARKA ARKAYA en az kac kare cikis esiginin altinda kalmasi gerektigi -
      buyutursen tek bir hareketin tremor/titremeyle birden fazla kez
      sayilmasi ihtimali azalir ama ardisik AYRI iki hareket arasindaki
      minimum bosluk da uzar (asiri buyutursen gercekten ayri iki hareket
      tek sayilmaya baslar).

    Donus: (hareketli, yeni_hizli_x, yeni_hizli_y, yeni_yavas_x, yeni_yavas_y,
    yeni_cikis_kare_sayaci).
    """
    if hizli_x is None:
        return False, x, y, x, y, 0
    yeni_hizli_x = hizli_oran * x + (1 - hizli_oran) * hizli_x
    yeni_hizli_y = hizli_oran * y + (1 - hizli_oran) * hizli_y
    yeni_yavas_x = yavas_oran * x + (1 - yavas_oran) * yavas_x
    yeni_yavas_y = yavas_oran * y + (1 - yavas_oran) * yavas_y
    mesafe = math.hypot(yeni_hizli_x - yeni_yavas_x, yeni_hizli_y - yeni_yavas_y)
    esik_kullan = esik * histerezis_orani if onceki_hareketli else esik
    if onceki_hareketli:
        # DEBOUNCE: cikis esiginin altina TEK karede degil, ARKA ARKAYA
        # min_cikis_kare kare boyunca dusmesi gerekiyor - araya bir tremor
        # karesi (mesafe tekrar esigin ustune cikarsa) girerse sayac sifirlanir.
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


def parmak_hareket_algila(hizli_x, hizli_y, x, y, esik, hizli_oran=0.6,
                           son_tetik_uzerinden_kare=9999, min_yeniden_tetik_kare=5):
    """Parmak ucu icin hareket_algila'DAN FARKLI, DAHA TEPKISEL bir tetikleyici
    (17.08.2026, kullanici istegi: "parmak biraz oynadi mi hemen algilasin,
    asagi inip tekrar kalkmasina gerek olmasin").

    NEDEN hareket_algila (KOL/BACAK'ta kullanilan) BURAYA UYMUYOR: o
    fonksiyon HIZLI ve YAVAS iki EMA arasindaki farka bakar - YAVAS EMA
    (yavas_oran=0.035) bir harekete ~1 saniyede "yetisir". Bu, "hareketli"
    durumundan cikip YENIDEN tetiklenebilmek icin mesafenin cikis esiginin
    altina DUSMESINI gerektirir - yani parmak KALKIK/degismis konumda
    TUTULURKEN yeni bir kipirdanma olsa bile, YAVAS EMA henuz yetismediyse
    (hala "hareketli" sayiliyorsa) bu YENI kipirdanma SAYILMAZ. KOL/BACAK
    icin bu DOGRU (ayni hareketi tekrar saymamak istiyoruz), ama PARMAK icin
    istenen TAM TERSI: el havada dururken bile HER YENI kipirdanma ayri ayri
    yakalanmali.

    YONTEM: TEK bir (kisa/HIZLI, hizli_oran=0.6) EMA'nin KENDI ARDISIK IKI
    KARE arasindaki degisimi (yani "yumusatilmis konumun hizi") olculur -
    bu, YAVAS bir referansa GORE degil, konumun KENDI ANLIK degisim ORANINA
    bakar. Hareket DURUNCA bu hiz DEGERI birkac kare icinde (hizli_oran'in
    kisa zaman sabitiyle) SIFIRA doner - yani "asagi inip tekrar kalkma"
    beklenmez, sadece KISA bir "ayni titremeyi birden fazla kez saymayi
    onleyen" yeniden-tetiklenme bekleme suresi (min_yeniden_tetik_kare, kac
    KARE) vardir.

    - hizli_x/y: bir onceki karenin yumusatilmis (HIZLI EMA) konumu (None
      ise ilk kare).
    - x, y: bu karenin HAM (goreli) konumu.
    - esik: ardisik iki karedeki yumusatilmis konum degisimi bu kadar
      olursa "hareket algilandi" sayilir.
    - son_tetik_uzerinden_kare: en son tetiklenmeden BU YANA kac kare
      gectigi (baslangicta buyuk bir sayi ver, ilk hareket hemen sayilsin).
    - min_yeniden_tetik_kare: iki ayri tetiklenme arasinda ARKA ARKAYA en
      az kac kare gecmesi gerektigi (tek bir kipirdanmanin birden fazla
      ardisik karede sayilmasini onler - "debounce" degil, kisa bir
      "yeniden silahlanma/refractory" suresi).

    Donus: (tetiklendi_mi, yeni_hizli_x, yeni_hizli_y, yeni_son_tetik_uzerinden_kare, hiz)
    - hiz SADECE tani/debug amacli (esikle karsilastirmak icin ekrana yazdirilabilir).
    """
    if hizli_x is None:
        return False, x, y, son_tetik_uzerinden_kare + 1, 0.0
    yeni_hizli_x = hizli_oran * x + (1 - hizli_oran) * hizli_x
    yeni_hizli_y = hizli_oran * y + (1 - hizli_oran) * hizli_y
    hiz = math.hypot(yeni_hizli_x - hizli_x, yeni_hizli_y - hizli_y)
    tetiklendi = hiz > esik and son_tetik_uzerinden_kare >= min_yeniden_tetik_kare
    yeni_sayac = 0 if tetiklendi else son_tetik_uzerinden_kare + 1
    return tetiklendi, yeni_hizli_x, yeni_hizli_y, yeni_sayac, hiz


def omuz_genisligi_piksel(sol_omuz, sag_omuz, w, h):
    """Iki omuz landmark'i arasindaki piksel mesafesi - kimlik kilidinde
    govdenin "buyuklugu" (sicrama esigini olceklemek) icin kullanilir."""
    dx = (sol_omuz.x - sag_omuz.x) * w
    dy = (sol_omuz.y - sag_omuz.y) * h
    return math.hypot(dx, dy)


def govde_olcek_hesapla(sol_omuz, sag_omuz, min_olcek=0.03):
    """Omuzlar arasi NORMALIZE (0..1, kare-ici) mesafe - govdenin
    "buyuklugunu" (kisi kameraya ne kadar yakin/uzak) temsil eder. Kol/bacak
    hareketini bu olcege GORE normalize etmek icin kullanilir (bkz.
    govdeye_goreli_konum) - boylece ayni FIZIKSEL hareket, kisi kameraya
    yakinken/uzakken (govde ekranda buyuk/kucuk gorunse de) benzer bir
    sinyal uretir. min_olcek: bolme-sifira-karsi taban deger (omuzlar
    gecici olarak neredeyse ust uste geldiyse/yanlis algilandiysa bile
    guvenli kalinsin diye)."""
    olcek = math.hypot(sol_omuz.x - sag_omuz.x, sol_omuz.y - sag_omuz.y)
    return max(olcek, min_olcek)


def govdeye_goreli_konum(nokta, referans, olcek):
    """Bir uzuv noktasinin (bilek/ayak bilegi) GOVDEYE AIT bir referans
    noktaya (ayni taraf omuz/kalca) GORE, govde OLCEGINE (bkz.
    govde_olcek_hesapla) BOLUNMUS konumu.

    NEDEN BU GEREKLI - KOL/BACAK CAPRAZ TETIKLENME SORUNU: hareket_algila'yi
    HAM/MUTLAK (kare-ici) x,y konumuyla besledigimizde, kol ve bacak
    hareketleri TEORIDE bagimsiz degildi: govde/yatak/kamera EN UFAK
    sekilde kaydiginda (orn. kolunu guclu oynatan bir hastanin govdesi de
    hafifce yaslaniyor/kayiyor, ya da telefon kamerasi hafifce titriyor)
    TUM landmark'lar (bilek DE ayak bilegi DE) AYNI YONDE birlikte kayiyor -
    bu da "kolu oynatinca bacak sayaci da artiyor, bacagi oynatinca kol
    sayaci da artiyor" seklinde CAPRAZ YANLIS TETIKLENMEYE yol aciyordu
    (gercek kullanici geri bildirimiyle bulundu - kod capraz baglanti
    ICERMIYORDU, ama HER IKI sayaç da MUTLAK konum kullandigi icin GOVDE
    GENELINDEKI ORTAK bir kaymaya AYNI ANDA tepki veriyorlardi).

    COZUM: uzvun MUTLAK konumu yerine, GOVDEYE GORE (ayni taraf omuz/kalca
    referans alinarak) konumunu kullanmak. Boylece TUM govde birlikte
    kayarsa (referans nokta DA ayni miktarda kayar) ARADAKI FARK
    DEGISMEZ -> ortak/govde-geneli kaymalar MATEMATIKSEL OLARAK iptal olur.
    SADECE uzvun govdeye GORE gercekten (bagimsiz olarak) hareket etmesi
    sinyal uretir. Govde olcegine bolmek de kisinin kameraya yakinligindan
    bagimsiz, tutarli bir esik kullanilabilmesini saglar.

    Donus: (goreli_x, goreli_y) - boyutsuz (govde olcegi cinsinden, orn.
    0.5 = omuz genisligi kadar bir yer degistirme).
    """
    return (nokta.x - referans.x) / olcek, (nokta.y - referans.y) / olcek


def dijital_yakinlastir(kare, oran):
    """Kareyi TAM ORTASINDAN kirpip eski boyutuna geri buyutur (bkz.
    ayarlar.DIJITAL_YAKINLASTIRMA). oran<=1.0 ise kareyi OLDUGU GIBI
    dondurur (kopyasiz, maliyetsiz).

    NOT: Bu SABIT/merkez-odakli eski davranis - hala baska bir yerde
    kullanilmasi gerekirse diye korunuyor. Uc birlesik dosya (gaze_/l2cs_/
    uniface_birlesik.py) artik asagidaki takip_yakinlastir'i kullaniyor
    (kisi kadrajda nerede olursa olsun ONU takip eder)."""
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
    """dijital_yakinlastir ile AYNI mantik ama kirpma alani SABIT merkez
    yerine 'takip_merkezi' (piksel, BU karenin kendi w x h koordinat
    sisteminde) etrafinda konumlanir - kisi kadrajin neresinde olursa
    olsun (merkezde olmasa bile) yakinlastirilmis/detayli goruntude kalir.

    takip_merkezi=None ise (henuz kimse kilitlenmedi / kilit tamamen
    birakildi) TAM ORTADAN kirpar - yani ilk karelerde/kisiyi kaybedince
    otomatik olarak genis (arama) goruntusune doner.

    Donus: (yakinlastirilmis_kare, kirpma_dikdortgeni). kirpma_dikdortgeni
    = (x1, y1, kirp_w, kirp_h) - bu KAREDE hangi ham bolgenin kirpildigini
    tutar; bir SONRAKI karede tespit edilen bir konumu HAM koordinatlara
    geri cevirmek (bkz. raw_konuma_cevir) ve takip_merkezi'ni guncellemek
    icin gerekli.
    """
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
    """takip_yakinlastir'in urettigi yakinlastirilmis karedeki (x, y) piksel
    konumunu, o karenin turetildigi HAM (kirpilmamis kameradan gelen)
    karenin koordinatlarina geri cevirir - bir sonraki karenin
    takip_merkezi'ni guncellemek icin kullanilir."""
    x1, y1, kirp_w, kirp_h = kirpma_dikdortgeni
    raw_x = x1 + (x / w) * kirp_w
    raw_y = y1 + (y / h) * kirp_h
    return raw_x, raw_y


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

# --- UZAK KAMERA: SABIT BOLGE ZOOM (bkz. ayarlar.BOLGE_* aciklamasi, ---------
# nokta_sec.py, gaze_birlesik_uzak.py) - takip_yakinlastir'DAN farki: kirpma
# merkezi HER KAREDE yeniden hesaplanan bir "takip" noktasi DEGIL, nokta_sec.py
# ile BIR KEZ elle isaretlenmis SABIT (normalize, 0..1) bir nokta. Kamera da
# hasta da sabit durdugu icin "takip etmeye" gerek yok - bu yuzden ayri/daha
# basit bir fonksiyon (takip_yakinlastir'a hicbir sekilde dokunulmadi).
def bolge_kirp(kare_ham, nx, ny, oran, panel_genislik, panel_yukseklik):
    """HAM (kirpilmamis) kameradan gelen karede, normalize (nx,ny) noktasi
    etrafinda oran'a gore kirpip (panel_genislik x panel_yukseklik) boyutuna
    buyutur. nokta_sec.py'nin kaydettigi noktalar da AYNI HAM kare uzerinde
    (nx,ny) olarak tanimlandigi icin koordinat sistemleri birebir tutarlidir.

    Donus: (panel, kirpma_dikdortgeni) - kirpma_dikdortgeni = HAM karedeki
    (x1, y1, kirp_w, kirp_h), debug/gorsel dogrulama (bkz. gaze_birlesik_uzak.py
    - genis-aci penceresinde her bolgenin kirpma alanini kutu olarak cizmek
    icin) amacli donduruluyor.
    """
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
    """Ayni boyuttaki panel goruntularini (liste) TEK bir izgara/"bolunmus
    ekran" goruntusune yan yana/alt alta dizer (bkz. gaze_birlesik_uzak.py).
    Panel sayisi verilen sutun_sayisi'na sigmiyorsa otomatik ikinci satira
    gecer, EKSIK kalan hucreler siyahla doldurulur (np.vstack genislikleri
    esit istedigi icin).

    sutun_sayisi=None ise TUM panelleri TEK SATIRDA yan yana dizer (bu
    projede tipik kullanim: en fazla 3 bolge - yuz/sol el/sag el).
    """
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