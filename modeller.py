"""OpenVINO gaze-estimation-adas-0002 (bakis) ve MediaPipe Tasks (yuz/govde/el)
modellerinin yuklenmesi.

gaze-estimation-adas-0002.xml/.bin, MediaPipe .task dosyalari gibi ilk
calistirmada BURAYA (ayarlar.BURASI) otomatik indirilir (internet gerekir,
sadece ilk sefer). Indirme basarisiz olursa ayarlar.py'deki URL'lerden elle
indirip ayni klasore koyabilirsin.
"""
import time
import urllib.request

import mediapipe as mp

import ayarlar as A


def _indir_gerekirse(url, yol, ad):
    if not yol.exists():
        print(f"{ad} indiriliyor (bir kereye mahsus, internet gerekir)...")
        urllib.request.urlretrieve(url, yol)


def gaze_pipeline_yukle():
    """OpenVINO gaze-estimation-adas-0002 modelini yukler, calistirilabilir
    (compiled) modeli dondurur.

    NOT: openvino import'u BILEREK burada, fonksiyon icinde (lazy) -
    ayarlar.AKTIF_GAZE=False iken bu fonksiyon hic cagrilmiyor, yani bu
    import'un suresi de HIC harcanmiyor.
    """
    import openvino as ov

    _indir_gerekirse(A.GAZE_MODEL_XML_URL, A.GAZE_MODEL_XML, "gaze-estimation-adas-0002.xml")
    _indir_gerekirse(A.GAZE_MODEL_BIN_URL, A.GAZE_MODEL_BIN, "gaze-estimation-adas-0002.bin")

    if not A.GAZE_MODEL_XML.exists() or not A.GAZE_MODEL_BIN.exists():
        raise SystemExit(
            f"gaze-estimation-adas-0002.xml VE .bin dosyalarini indiremedim. "
            f"Elle indirip {A.BURASI} icine koy (ayarlar.py'deki GAZE_MODEL_*_URL "
            "degerlerine bak)."
        )

    _t0 = time.time()
    core = ov.Core()
    model = core.read_model(str(A.GAZE_MODEL_XML))
    compiled = core.compile_model(model, A.GAZE_CIHAZ)
    print(f"[zaman] OpenVINO gaze modeli ({A.GAZE_CIHAZ}) yuklendi: {time.time() - _t0:.1f}s")
    return compiled.create_infer_request()


def mediapipe_landmarker_lari_yukle():
    """FaceLandmarker'i (her zaman), PoseLandmarker/HandLandmarker'i (ayarlar.
    AKTIF_POSE/AKTIF_EL True ise) yukler. (face, pose, hand) dondurur - pose/
    hand kapaliysa None olur.

    FaceLandmarker HEM blendshape (kirpma icin) HEM facial transformation
    matrix (gaze icin head-pose) HEM 478 nokta landmark (gaze icin goz
    kirpintisi) uretecek sekilde ayarlanir - L2CS surumunden farkli olarak
    ayrica bir RetinaFace/yuz tespiti CALISMAZ, tek dedektor hepsine yeter.
    """
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    _indir_gerekirse(A.FACE_TASK_URL, A.FACE_TASK_YOLU, "face_landmarker.task")
    if A.AKTIF_POSE:
        _indir_gerekirse(A.POSE_TASK_URL, A.POSE_TASK_YOLU, "pose_landmarker_lite.task")
    if A.AKTIF_EL:
        _indir_gerekirse(A.HAND_TASK_URL, A.HAND_TASK_YOLU, "hand_landmarker.task")

    # KIMLIK_KILIDI_AKTIF ise FaceLandmarker/PoseLandmarker BIRDEN FAZLA aday
    # dondurur (asagidaki *_ADAY_SAYISI kadar) - bu, gaze_birlesik.py'nin
    # "kilitli kisiyi digerlerinden ayirt edebilmesi" icin gerekli (bkz.
    # ayarlar.py). Bu, MediaPipe'in KENDI (hafif) tespitini biraz daha
    # calistirir ama pahali OpenVINO bakis modeli SADECE secilen tek aday
    # icin cagrildigindan performansi onemli olcude etkilemez.
    yuz_aday_sayisi = A.YUZ_ADAY_SAYISI if A.KIMLIK_KILIDI_AKTIF else 1
    govde_aday_sayisi = A.GOVDE_ADAY_SAYISI if A.KIMLIK_KILIDI_AKTIF else 1

    _t3 = time.time()
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(A.FACE_TASK_YOLU)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=yuz_aday_sayisi,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=A.YUZ_TESPIT_ESIK,
            min_face_presence_confidence=A.YUZ_TESPIT_ESIK,
            min_tracking_confidence=A.YUZ_TESPIT_ESIK,
        )
    )
    print(f"[zaman] FaceLandmarker yuklendi: {time.time() - _t3:.1f}s")

    pose_landmarker = None
    if A.AKTIF_POSE:
        _t4 = time.time()
        pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(A.POSE_TASK_YOLU)),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=govde_aday_sayisi,
            )
        )
        print(f"[zaman] PoseLandmarker yuklendi: {time.time() - _t4:.1f}s")

    hand_landmarker = None
    if A.AKTIF_EL:
        _t5 = time.time()
        hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(A.HAND_TASK_YOLU)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )
        )
        print(f"[zaman] HandLandmarker yuklendi: {time.time() - _t5:.1f}s")

    return face_landmarker, pose_landmarker, hand_landmarker

def pose_landmarker_yukle():
    """SADECE PoseLandmarker'i yukler (FaceLandmarker'SIZ) - AYRICA/YENI bir
    fonksiyon, mediapipe_landmarker_lari_yukle'ye DOKUNMADAN eklendi.

    NEDEN AYRI: mediapipe_landmarker_lari_yukle HER ZAMAN (docstring'inde de
    belirtildigi gibi) bir FaceLandmarker yukler - gaze_birlesik.py bunu
    hem kirpma/bakis HEM kimlik-kilidi-adaylari icin kullaniyor. Ama
    gaze_birlesik_uzak.py'de (bkz. o dosyanin basindaki aciklama) yuz/el
    ARTIK genis kareden degil, nokta_sec.py ile isaretlenmis SABIT bolge
    panellerinden (bkz. bolge_landmarklarini_yukle) okunuyor - genis kare
    icin SADECE govde/kol/bacak takibi amacli PoseLandmarker gerekiyor.
    Kullanilmayacak bir FaceLandmarker'i yine de yuklemek gereksiz
    baslangic suresi/RAM harcar, bu yuzden bu YALIN alternatif eklendi.
    """
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    _indir_gerekirse(A.POSE_TASK_URL, A.POSE_TASK_YOLU, "pose_landmarker_lite.task")
    govde_aday_sayisi = A.GOVDE_ADAY_SAYISI if A.KIMLIK_KILIDI_AKTIF else 1
    _t = time.time()
    pose_landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(A.POSE_TASK_YOLU)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=govde_aday_sayisi,
        )
    )
    print(f"[zaman] PoseLandmarker yuklendi (bolge modu, genis-aci kol/bacak icin): {time.time() - _t:.1f}s")
    return pose_landmarker


def bolge_landmarklarini_yukle(bolge_turleri):
    """gaze_birlesik_uzak.py'nin SABIT BOLGE panelleri (bkz. ayarlar.BOLGE_*,
    nokta_sec.py) icin AYRI/kendi FaceLandmarker + HandLandmarker'ini yukler
    - genis-kare PoseLandmarker'indan (bkz. pose_landmarker_yukle) VE
    normal gaze_birlesik.py'nin kimlik-kilitli/coklu-aday landmarker'larindan
    (bkz. mediapipe_landmarker_lari_yukle) TAMAMEN BAGIMSIZDIR - kendi
    zaman damgasi sayaci kullanilir (bkz. gaze_birlesik_uzak.py), TEK hedef
    (num_faces=1) beklenir (bolge zaten TEK yuzu/eli kapsayacak sekilde elle
    isaretlendigi icin coklu-aday kimlik kilidi mantigina GEREK YOK).

    bolge_turleri: {"yuz", "el"} kumesinin bir alt kumesi (nokta_sec.py'de
    tanimlanmis bolgelerin turlerinden turetilir) - SADECE GEREKEN modeli
    yukler, gereksiz baslangic suresi harcamamak icin.

    Donus: (bolge_face_landmarker, bolge_hand_landmarker) - tanimli degilse
    (ilgili tur bolge_turleri'nde yoksa) None.
    """
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    bolge_face = None
    if "yuz" in bolge_turleri:
        _indir_gerekirse(A.FACE_TASK_URL, A.FACE_TASK_YOLU, "face_landmarker.task (bolge)")
        _t = time.time()
        bolge_face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(A.FACE_TASK_YOLU)),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=A.YUZ_TESPIT_ESIK,
                min_face_presence_confidence=A.YUZ_TESPIT_ESIK,
                min_tracking_confidence=A.YUZ_TESPIT_ESIK,
            )
        )
        print(f"[zaman] FaceLandmarker yuklendi (yuz bolgesi icin): {time.time() - _t:.1f}s")

    bolge_hand = None
    if "el" in bolge_turleri:
        _indir_gerekirse(A.HAND_TASK_URL, A.HAND_TASK_YOLU, "hand_landmarker.task (bolge)")
        _t = time.time()
        bolge_hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(A.HAND_TASK_YOLU)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )
        )
        print(f"[zaman] HandLandmarker yuklendi (el bolgeleri icin): {time.time() - _t:.1f}s")

    return bolge_face, bolge_hand