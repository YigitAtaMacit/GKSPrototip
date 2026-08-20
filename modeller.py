"""OpenVINO gaze-estimation-adas-0002 ve MediaPipe Tasks modellerinin yuklenmesi - dosyalar ilk calistirmada ayarlar.BURASI'na otomatik indirilir."""
import time
import urllib.request

import mediapipe as mp

import ayarlar as A


def _indir_gerekirse(url, yol, ad):
    if not yol.exists():
        print(f"{ad} indiriliyor (bir kereye mahsus, internet gerekir)...")
        urllib.request.urlretrieve(url, yol)


def gaze_pipeline_yukle():
    """OpenVINO gaze modelini yukler, compiled modeli dondurur - openvino import'u lazy (AKTIF_GAZE=False'ta hic yuklenmez)."""
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
    """FaceLandmarker (her zaman) + PoseLandmarker/HandLandmarker (AKTIF_POSE/AKTIF_EL ise) yukler, (face,pose,hand) dondurur - kapali olanlar None."""
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

    yuz_aday_sayisi = A.YUZ_ADAY_SAYISI if A.KIMLIK_KILIDI_AKTIF else 1  # KIMLIK_KILIDI_AKTIF ise birden fazla aday donsun ki kilitli kisi ayirt edilebilsin
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
                min_hand_detection_confidence=A.EL_TESPIT_ESIK,
                min_hand_presence_confidence=A.EL_IZLEME_ESIK,
                min_tracking_confidence=A.EL_IZLEME_ESIK,
            )
        )
        print(f"[zaman] HandLandmarker yuklendi: {time.time() - _t5:.1f}s")

    return face_landmarker, pose_landmarker, hand_landmarker

def pose_landmarker_yukle():
    """SADECE PoseLandmarker'i yukler (FaceLandmarker'siz) - gaze_birlesik_uzak.py'de yuz/el SABIT bolgelerden okundugu icin genis karede gereksiz FaceLandmarker yukune gerek yok."""
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


def bolge_landmarklarini_yukle(bolge_turleri, etiket="bolge", el_bolge_adlari=None):
    """SABIT BOLGE panelleri icin ayri FaceLandmarker+HandLandmarker yukler; el_bolge_adlari verilirse (SABIT BOLGE modu) her ad icin AYRI HandLandmarker doner ({ad: HandLandmarker}) - aksi halde (ANA KAMERA modu) TEK paylasimli landmarker (num_hands=2) yeterli, cunku iki el ayni tek karede birlikte islenir. AYRI landmarker sarti: TEK paylasimli instance'ta sol_el/sag_el ayni "onceki ROI" hafizasini paylasip birbirinin takibini bozuyordu (19.08.2026 bulundu). Donus: (bolge_face, bolge_hand_landmarker_veya_sozlugu)."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    bolge_face = None
    if "yuz" in bolge_turleri:
        _indir_gerekirse(A.FACE_TASK_URL, A.FACE_TASK_YOLU, f"face_landmarker.task ({etiket})")
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
        print(f"[zaman] FaceLandmarker yuklendi ({etiket}): {time.time() - _t:.1f}s")

    bolge_hand = None
    if "el" in bolge_turleri:
        _indir_gerekirse(A.HAND_TASK_URL, A.HAND_TASK_YOLU, f"hand_landmarker.task ({etiket})")
        if el_bolge_adlari:
            bolge_hand = {}
            for _ad in el_bolge_adlari:
                _t = time.time()
                bolge_hand[_ad] = HandLandmarker.create_from_options(
                    HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(A.HAND_TASK_YOLU)),
                        running_mode=VisionRunningMode.VIDEO,
                        num_hands=1,
                        min_hand_detection_confidence=A.EL_TESPIT_ESIK,
                        min_hand_presence_confidence=A.EL_IZLEME_ESIK,
                        min_tracking_confidence=A.EL_IZLEME_ESIK,
                    )
                )
                print(f"[zaman] HandLandmarker yuklendi ({etiket}, {_ad}): {time.time() - _t:.1f}s")
        else:
            _t = time.time()
            bolge_hand = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(A.HAND_TASK_YOLU)),
                    running_mode=VisionRunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=A.EL_TESPIT_ESIK,
                    min_hand_presence_confidence=A.EL_IZLEME_ESIK,
                    min_tracking_confidence=A.EL_IZLEME_ESIK,
                )
            )
            print(f"[zaman] HandLandmarker yuklendi ({etiket}): {time.time() - _t:.1f}s")

    return bolge_face, bolge_hand