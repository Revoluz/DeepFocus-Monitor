"""
core/face_analyzer.py - Analisis Biometrik Wajah

File ini berisi kelas FaceAnalyzer yang bertanggung jawab untuk:
1. Mendeteksi landmark wajah menggunakan MediaPipe Face Landmarker
2. Menghitung Eye Aspect Ratio (EAR) untuk deteksi mata tertutup
3. Menghitung Lip Distance untuk deteksi menguap
4. Mengestimasi head pose (yaw, pitch, roll) menggunakan solvePnP

Teknologi:
- MediaPipe Face Landmarker: Model AI untuk deteksi 468 titik wajah
- OpenCV solvePnP: Algoritma untuk estimasi orientasi 3D dari titik 2D

Alur Kerja:
1. Load model Face Landmarker (auto-download jika belum ada)
2. Untuk setiap frame:
   a. Deteksi wajah dan ekstrak 468 landmarks
   b. Hitung EAR dari 6 titik per mata (kiri + kanan)
   c. Hitung Lip Distance dari titik bibir atas dan bawah
   d. Estimasi head pose menggunakan 6 titik wajah + solvePnP
3. Return dictionary dengan semua hasil analisis
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.helpers import euclidean_distance


# =============================================================================
# KONFIGURASI MODEL MEDIAPIPE
# =============================================================================

# URL download model Face Landmarker (auto-download jika belum ada)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"  # Path lokal untuk menyimpan model

# =============================================================================
# INDEKS LANDMARK MATA (MediaPipe Face Mesh 468 titik)
# =============================================================================
# Setiap mata menggunakan 6 titik untuk menghitung EAR:
# p1 = sudut dalam, p4 = sudut luar
# p2, p3 = kelopak atas, p5, p6 = kelopak bawah
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# =============================================================================
# INDEKS LANDMARK BIBIR
# =============================================================================
UPPER_LIP_INDEX = 13  # Titik tengah bibir atas
LOWER_LIP_INDEX = 14  # Titik tengah bibir bawah

# =============================================================================
# TITIK 3D UNTUK HEAD POSE ESTIMATION (solvePnP)
# =============================================================================
# 6 titik wajah dengan koordinat 3D standar (dalam mm):
# [0] Nose tip, [1] Chin, [2] Left eye corner, [3] Right eye corner,
# [4] Left mouth corner, [5] Right mouth corner
HEAD_POSE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -330.0, -65.0],      # Chin
    [-225.0, 170.0, -135.0],   # Left eye corner
    [225.0, 170.0, -135.0],    # Right eye corner
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [150.0, -150.0, -125.0],   # Right mouth corner
], dtype=np.double)

# Indeks landmark 2D yang sesuai dengan titik 3D di atas
HEAD_POSE_2D_INDICES = [1, 9, 33, 263, 61, 291]


def ensure_face_landmarker_model(model_path: str) -> None:
    """
    Memastikan model Face Landmarker tersedia di path lokal.
    Jika belum ada, model akan diunduh otomatis dari Google Storage.

    Args:
        model_path: Path lokal untuk menyimpan model (.task file)
    """
    if os.path.exists(model_path):
        return
    print("Mengunduh model Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Model tersimpan di: {model_path}")


class FaceAnalyzer:
    """
    Kelas untuk analisis biometrik wajah menggunakan MediaPipe Face Landmarker.

    Fungsi utama:
    - calculate_ear: Menghitung Eye Aspect Ratio (EAR) untuk deteksi mata tertutup
    - calculate_lip_distance: Menghitung jarak bibir untuk deteksi menguap
    - estimate_head_pose: Mengestimasi orientasi kepala (yaw, pitch, roll)
    - analyze: Pipeline lengkap untuk analisis satu frame

    Cara kerja EAR (Eye Aspect Ratio):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Dimana:
    - p1, p4 = sudut dalam dan luar mata (horizontal)
    - p2, p3 = kelopak mata atas
    - p5, p6 = kelopak mata bawah

    EAR menurun saat mata tertutup, meningkat saat mata terbuka.
    Threshold EAR ditentukan melalui kalibrasi dinamis per pengguna.
    """

    def __init__(self, fps=30):
        """
        Inisialisasi FaceAnalyzer dengan MediaPipe Face Landmarker.

        Args:
            fps: Frame per second untuk timing detection
        """
        # Pastikan model tersedia
        ensure_face_landmarker_model(MODEL_PATH)

        # Setup MediaPipe Face Landmarker dengan parameter confidence
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,  # Hanya deteksi 1 wajah
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.fps = fps
        self.camera_matrix = None
        self.dist_coeffs = None
        self._init_camera_matrix()

    def _init_camera_matrix(self):
        """
        Inisialisasi matriks kamera untuk head pose estimation (solvePnP).

        Matriks kamera berisi parameter intrinsik kamera:
        - Focal length (fx, fy)
        - Optical center (cx, cy)

        Matriks ini diperlukan untuk mengkonversi titik 2D ke 3D
        dalam estimasi orientasi kepala.
        """
        frame_width, frame_height = 640, 480
        focal_length = frame_width
        self.camera_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1],
        ], dtype=np.double)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.double)

    def calculate_ear(self, landmarks, side='left'):
        """
        Menghitung Eye Aspect Ratio (EAR) untuk satu mata.

        EAR adalah rasio antara jarak vertikal dan horizontal kelopak mata.
        Nilai EAR menurun saat mata tertutup, meningkat saat mata terbuka.

        Rumus:
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            landmarks: List 468 titik wajah dari MediaPipe
            side: 'left' untuk mata kiri, 'right' untuk mata kanan

        Returns:
            float: Nilai EAR (0.0 = tertutup, 0.2-0.35 = terbuka normal)

        Referensi:
            Soukupova & Cech (2016) - Real-Time Eye Blink Detection
        """
        if side == 'left':
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in LEFT_EYE_INDICES]
        else:
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in RIGHT_EYE_INDICES]

        # Jarak vertikal (kelopak atas ke bawah)
        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)

        # Jarak horizontal (sudut dalam ke luar)
        horizontal = euclidean_distance(p1, p4)

        # Hindari division by zero
        if horizontal == 0:
            return 0.0

        # Hitung EAR
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def calculate_ear_average(self, landmarks):
        """
        Menghitung rata-rata EAR dari kedua mata.

        Menggunakan rata-rata untuk mengurangi noise dan meningkatkan
        akurasi deteksi mata tertutup.

        Args:
            landmarks: List 468 titik wajah dari MediaPipe

        Returns:
            float: Rata-rata EAR dari mata kiri dan kanan
        """
        ear_left = self.calculate_ear(landmarks, 'left')
        ear_right = self.calculate_ear(landmarks, 'right')
        return (ear_left + ear_right) / 2.0

    def calculate_lip_distance(self, landmarks):
        """
        Menghitung jarak absolut antara bibir atas dan bawah.

        Digunakan untuk mendeteksi aktivitas menguap.
        Jarak meningkat saat mulut terbuka lebar (menguap).

        Args:
            landmarks: List 468 titik wajah dari MediaPipe

        Returns:
            float: Jarak Euclidean antara bibir atas dan bawah (normalized)
        """
        upper_lip = landmarks[UPPER_LIP_INDEX]
        lower_lip = landmarks[LOWER_LIP_INDEX]
        return euclidean_distance(upper_lip, lower_lip)

    def estimate_head_pose(self, landmarks, frame_shape):
        """
        Mengestimasi orientasi kepala (yaw, pitch, roll) menggunakan solvePnP.

        solvePnP (Perspective-n-Point) adalah algoritma yang mengestimasi
        pose objek 3D dari titik-titik 2D yang terdeteksi di gambar.

        Args:
            landmarks: List 468 titik wajah dari MediaPipe
            frame_shape: Dimensi frame (height, width, channels)

        Returns:
            dict: {'yaw': float, 'pitch': float, 'roll': float} dalam derajat

        Penjelasan sudut:
            - yaw: Rotasi kiri-kanan (menoleh)
            - pitch: Rotasi atas-bawah (mengangguk/menunduk)
            - roll: Rotasi miring (kepala miring ke samping)
        """
        h, w = frame_shape[:2]

        # Konversi landmark 2D normalized ke pixel coordinates
        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in HEAD_POSE_2D_INDICES
        ], dtype=np.double)

        # SolvePnP: estimasi rotation & translation vector
        success, rotation_vector, translation_vector = cv2.solvePnP(
            HEAD_POSE_3D_POINTS,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

        # Convert rotation vector ke rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose projection matrix untuk dapat Euler angles
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

        # Extract yaw, pitch, roll dari Euler angles
        yaw = float(euler_angles[1][0])
        pitch = float(euler_angles[0][0])
        roll = float(euler_angles[2][0])

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    def analyze(self, frame):
        """
        Pipeline lengkap untuk analisis satu frame.

        Memproses frame melalui MediaPipe Face Landmarker dan
        menghitung semua metrik biometrik (EAR, Lip Distance, Head Pose).

        Args:
            frame: Frame BGR dari kamera (numpy array)

        Returns:
            dict: Hasil analisis dengan keys:
                - face_detected (bool): Apakah wajah terdeteksi
                - ear (float/None): Eye Aspect Ratio rata-rata
                - lip_distance (float/None): Jarak bibir
                - head_pose (dict/None): {'yaw', 'pitch', 'roll'}
                - landmarks (list/None): 468 titik wajah
        """
        # Convert BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.monotonic() * 1000)

        # Deteksi wajah dan landmarks
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        # Jika tidak ada wajah terdeteksi
        if not results.face_landmarks:
            return {
                'face_detected': False,
                'ear': None,
                'lip_distance': None,
                'head_pose': None,
                'landmarks': None,
            }

        # Ekstrak landmarks wajah pertama
        landmarks = results.face_landmarks[0]

        # Hitung semua metrik
        ear = self.calculate_ear_average(landmarks)
        lip_distance = self.calculate_lip_distance(landmarks)
        head_pose = self.estimate_head_pose(landmarks, frame.shape)

        return {
            'face_detected': True,
            'ear': ear,
            'lip_distance': lip_distance,
            'head_pose': head_pose,
            'landmarks': landmarks,
        }
