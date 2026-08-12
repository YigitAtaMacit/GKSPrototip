"""L2CS-Net (bakis) ve MediaPipe Tasks (yuz/govde/el) modellerinin yuklenmesi.

Agirlik dosyasi (L2CSNet_gaze360.pkl) ELLE indirilip ayarlar.BURASI'na
konmali (bkz. l2csgaze.py docstring'i). MediaPipe .task dosyalari ise ilk
calistirmada BURAYA otomatik indirilir (internet gerekir, sadece ilk sefer).
"""
import time
import urllib.request

import mediapipe as mp
import torch
from l2cs import Pipeline

import ayarlar as A


def gaze_pipeline_yukle():
    """L2CS-Net (ResNet50 + RetinaFace) pipeline'ini yukler, gaze nesnesini dondurur."""
    if not A.AGIRLIK.exists():
        raise SystemExit(
            f"Once L2CSNet_gaze360.pkl dosyasini indirip {A.BURASI} icine koy "
            "(l2csgaze.py docstring'ine bak)."
        )

    _t0 = time.time()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[zaman] torch/cuda hazir: {time.time() - _t0:.1f}s")

    _t1 = time.time()
    gaze = Pipeline(weights=str(A.AGIRLIK), arch="ResNet50", device=device)
    print(f"[zaman] L2CS Pipeline (ResNet50 + RetinaFace) yuklendi: {time.time() - _t1:.1f}s")
    return gaze


def _indir_gerekirse(url, yol, ad):
    if not yol.exists():
        print(f"{ad} indiriliyor (bir kereye mahsus, internet gerekir)...")
        urllib.request.urlretrieve(url, yol)


def mediapipe_landmarker_lari_yukle():
    """FaceLandmarker'i (her zaman), PoseLandmarker/HandLandmarker'i (ayarlar.
    AKTIF_POSE/AKTIF_EL True ise) yukler. (face, pose, hand) dondurur - pose/
    hand kapaliysa None olur."""
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

    _t3 = time.time()
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(A.FACE_TASK_YOLU)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
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
                num_poses=1,
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