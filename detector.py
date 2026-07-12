"""
detector.py - Semua Logic Deteksi DeepFocus Monitor

File ini menggabungkan semua modul deteksi menjadi satu file:
1. FaceAnalyzer: Analisis biometrik wajah (EAR, Lip, Head Pose)
2. DynamicCalibration: Kalibrasi otomatis threshold per pengguna
3. MicrosleepClassifier: State machine untuk klasifikasi kantuk
4. StateManager: Koordinator status sistem
5. SessionTracker: Tracking statistik sesi
6. Alarm: Sistem alarm suara (TTS)
"""

import math
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def euclidean_distance(p1, p2):
    """
    Menghitung jarak Euclidean antara dua titik 2D.
    
    RUMUS:
        distance = √((x1 - x2)² + (y1 - y2)²)
    
    Digunakan untuk:
    - Menghitung jarak antar landmark mata (EAR calculation)
    - Menghitung jarak bibir atas-bawah (Lip Distance)
    
    Args:
        p1: Titik pertama dengan atribut .x dan .y (MediaPipe landmark)
        p2: Titik kedua dengan atribut .x dan .y (MediaPipe landmark)
    
    Returns:
        float: Jarak Euclidean antara dua titik
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# =============================================================================
# FACE ANALYZER
# =============================================================================

# URL download model Face Landmarker (auto-download jika belum ada)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# =============================================================================
# INDEKS LANDMARK MATA (MediaPipe Face Mesh 468 titik)
# =============================================================================
# Setiap mata menggunakan 6 titik untuk menghitung EAR:
#   p1 = sudut dalam (inner corner)
#   p2, p3 = kelopak mata atas (upper eyelid)
#   p4 = sudut luar (outer corner)
#   p5, p6 = kelopak mata bawah (lower eyelid)
#
# Mata Kiri:  [33, 160, 158, 133, 153, 144]
# Mata Kanan: [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Titik landmark bibir: 13 = bibir atas, 14 = bibir bawah
LIP_POINTS = [13, 14]

# Titik 3D untuk head pose estimation (solvePnP)
# 6 titik wajah dengan koordinat 3D standar (dalam mm)
HEAD_POSE_3D = np.array([
    [0.0, 0.0, 0.0],           # [0] Nose tip
    [0.0, -330.0, -65.0],      # [1] Chin
    [-225.0, 170.0, -135.0],   # [2] Left eye corner
    [225.0, 170.0, -135.0],    # [3] Right eye corner
    [-150.0, -150.0, -125.0],  # [4] Left mouth corner
    [150.0, -150.0, -125.0],   # [5] Right mouth corner
], dtype=np.double)
HEAD_POSE_2D = [1, 9, 33, 263, 61, 291]


def ensure_model(path):
    """Memastikan model Face Landmarker tersedia di path lokal."""
    if not os.path.exists(path):
        print("Mengunduh model Face Landmarker...")
        urllib.request.urlretrieve(MODEL_URL, path)


class FaceAnalyzer:
    """
    Analisis biometrik wajah: EAR, Lip Distance, Head Pose.
    
    Menggunakan MediaPipe Face Landmarker untuk mendeteksi 468 titik wajah,
    kemudian menghitung metrik biometrik untuk deteksi kantuk.
    """

    def __init__(self, fps=30):
        """Inisialisasi Face Landmarker detector dan camera matrix."""
        ensure_model(MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, running_mode=vision.RunningMode.VIDEO,
            num_faces=1, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.fps = fps
        # Camera matrix untuk head pose estimation (solvePnP)
        self.cam_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.double)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.double)

    def calculate_ear(self, landmarks, side='left'):
        """
        Menghitung Eye Aspect Ratio (EAR) untuk satu mata.
        
        RUMUS MATEMATIS (Soukupova & Cech, 2016):
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Dimana:
            p1 = sudut dalam mata (inner corner)
            p4 = sudut luar mata (outer corner)
            p2, p3 = kelopak mata atas
            p5, p6 = kelopak mata bawah
            ||a-b|| = jarak Euclidean antara titik a dan b
        
        INTERPRETASI:
            - EAR tinggi (0.25-0.35) = mata terbuka lebar
            - EAR sedang (0.15-0.25) = mata normal
            - EAR rendah (< 0.15) = mata setengah tertutup
            - EAR sangat rendah (< threshold) = mata tertutup (microsleep)
        
        Args:
            landmarks: List 468 titik wajah dari MediaPipe
            side: 'left' untuk mata kiri, 'right' untuk mata kanan
        
        Returns:
            float: Nilai EAR (0.0 = tertutup, 0.2-0.35 = terbuka normal)
        """
        indices = LEFT_EYE if side == 'left' else RIGHT_EYE
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
        
        # Jarak vertikal (kelopak atas ke bawah)
        v1 = euclidean_distance(p2, p6)  # Vertikal 1
        v2 = euclidean_distance(p3, p5)  # Vertikal 2
        
        # Jarak horizontal (sudut dalam ke luar)
        h = euclidean_distance(p1, p4)
        
        # Hindari division by zero
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def calculate_ear_avg(self, landmarks):
        """
        Menghitung rata-rata EAR dari kedua mata.
        
        Menggunakan rata-rata untuk mengurangi noise dan meningkatkan
        akurasi deteksi mata tertutup.
        
        Args:
            landmarks: List 468 titik wajah dari MediaPipe
        
        Returns:
            float: Rata-rata EAR dari mata kiri dan kanan
        """
        return (self.calculate_ear(landmarks, 'left') + self.calculate_ear(landmarks, 'right')) / 2.0

    def calculate_lip_dist(self, landmarks):
        """
        Menghitung jarak absolut antara bibir atas dan bawah.
        
        RUMUS:
            Lip Distance = ||landmark_13 - landmark_14||
                         = √((x13-x14)² + (y13-y14)²)
        
        INTERPRETASI:
            - < 0.02 = mulut tertutup (normal)
            - 0.02-0.05 = mulut sedikit terbuka (bicara normal)
            - > 0.05 = mulut terbuka lebar (menguap)
        
        Args:
            landmarks: List 468 titik wajah dari MediaPipe
        
        Returns:
            float: Jarak Euclidean antara bibir atas dan bawah (normalized)
        """
        return euclidean_distance(landmarks[LIP_POINTS[0]], landmarks[LIP_POINTS[1]])

    def estimate_head_pose(self, landmarks, frame_shape):
        """
        Estimasi orientasi kepala (yaw, pitch, roll) menggunakan solvePnP.
        
        solvePnP (Perspective-n-Point) mengestimasi pose objek 3D dari
        titik-titik 2D yang terdeteksi di gambar.
        
        Args:
            landmarks: List 468 titik wajah dari MediaPipe
            frame_shape: Dimensi frame (height, width, channels)
        
        Returns:
            dict: {'yaw': float, 'pitch': float, 'roll': float} dalam derajat
        """
        h, w = frame_shape[:2]
        # Konversi landmark 2D normalized ke pixel coordinates
        img_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_POSE_2D], dtype=np.double)
        
        # SolvePnP: estimasi rotation & translation vector
        success, rot_vec, trans_vec = cv2.solvePnP(HEAD_POSE_3D, img_pts, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # Convert rotation vector ke rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        # Decompose projection matrix untuk dapat Euler angles
        proj_mat = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        
        return {'yaw': float(euler[1][0]), 'pitch': float(euler[0][0]), 'roll': float(euler[2][0])}

    def analyze(self, frame):
        """
        Pipeline lengkap analisis satu frame.
        
        Args:
            frame: Frame BGR dari kamera (numpy array)
        
        Returns:
            dict: {
                'face_detected': bool,
                'ear': float/None,
                'lip_distance': float/None,
                'head_pose': dict/None,
                'landmarks': list/None
            }
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        results = self.detector.detect_for_video(mp_img, ts)

        if not results.face_landmarks:
            return {'face_detected': False, 'ear': None, 'lip_distance': None, 'head_pose': None, 'landmarks': None}

        lm = results.face_landmarks[0]
        return {
            'face_detected': True,
            'ear': self.calculate_ear_avg(lm),
            'lip_distance': self.calculate_lip_dist(lm),
            'head_pose': self.estimate_head_pose(lm, frame.shape),
            'landmarks': lm,
        }


# =============================================================================
# CALIBRATION & CLASSIFIER
# =============================================================================

class DynamicCalibration:
    """
    Kalibrasi otomatis threshold per pengguna (5 detik).
    
    ALGORITMA KALIBRASI:
    1. Kumpulkan 150 sampel EAR & Lip Distance (5 detik @ 30fps)
       - User diminta: mata terbuka normal, mulut tertutup, jangan bicara/menguap
    
    2. Hitung statistik dari sampel:
       - ear_mean = mean(ear_samples)      → baseline EAR normal
       - ear_std  = std(ear_samples)       → variasi EAR normal
       - lip_mean = mean(lip_samples)      → baseline lip normal
       - lip_std  = std(lip_samples)       → variasi lip normal
    
    3. Tentukan threshold:
       - ear_threshold = MAX(mean - 2*std, mean - 0.05)
       - lip_threshold = MAX(mean + 2*std, mean + 0.01)
    
    MENGAPA 2 * std?
    - 1 * std → mencakup ~68% data normal → terlalu sensitif
    - 2 * std → mencakup ~95% data normal → balance (yang dipakai)
    - 3 * std → mencakup ~99.7% data normal → terlalu longgar
    
    MENGAPA ADA FLOOR MINIMUM (mean - 0.05)?
    - Jika std sangat kecil (user sangat stabil), threshold bisa terlalu dekat ke mean
    - Floor minimum mencegah false trigger saat std kecil
    - ear_threshold minimal = mean - 0.05
    - lip_threshold minimal = mean + 0.01
    """

    def __init__(self, duration=config.CALIBRATION_DURATION, fps=config.FPS):
        """
        Inisialisasi kalibrasi.
        
        Args:
            duration: Durasi kalibrasi dalam detik (default: 5)
            fps: Frame per second (default: 30)
        """
        self.total_frames = duration * fps  # 150 frame untuk 5 detik
        self.ear_samples = []
        self.lip_samples = []
        self.current_frame = 0
        self.ear_threshold = None
        self.lip_threshold = None

    def add_sample(self, ear, lip):
        """
        Menambahkan sampel EAR dan Lip Distance ke dalam koleksi.
        
        Dipanggil setiap frame selama proses kalibrasi (150 kali).
        
        Args:
            ear: Nilai Eye Aspect Ratio dari frame saat ini
            lip: Jarak bibir dari frame saat ini
        
        Returns:
            bool: True jika kalibrasi selesai, False jika belum
        """
        if self.current_frame >= self.total_frames:
            return True
        
        self.ear_samples.append(ear)
        self.lip_samples.append(lip)
        self.current_frame += 1
        
        if self.current_frame >= self.total_frames:
            # =========================================================
            # PERHITUNGAN THRESHOLD SETELAH 150 SAMPEL TERKUMPUL
            # =========================================================
            
            # 1. Hitung mean (rata-rata) dari semua sampel
            ear_mean = np.mean(self.ear_samples)
            lip_mean = np.mean(self.lip_samples)
            
            # 2. Hitung std (standar deviasi) dari semua sampel
            ear_std = np.std(self.ear_samples)
            lip_std = np.std(self.lip_samples)
            
            # 3. Hitung threshold dengan floor minimum
            #    EAR threshold: batas bawah untuk deteksi mata tertutup
            #    Lip threshold: batas atas untuk deteksi menguap
            self.ear_threshold = max(
                ear_mean - (config.THRESHOLD_STD_MULTIPLIER * ear_std),  # Statistical: mean - 2*std
                ear_mean - 0.05  # Floor minimum: mean - 0.05
            )
            self.lip_threshold = max(
                lip_mean + (config.THRESHOLD_STD_MULTIPLIER * lip_std),  # Statistical: mean + 2*std
                lip_mean + 0.01  # Floor minimum: mean + 0.01
            )
            return True
        
        return False

    def progress(self):
        """Progress kalibrasi (0.0 - 1.0)."""
        return self.current_frame / self.total_frames

    def remaining(self, fps):
        """Sisa waktu kalibrasi dalam detik."""
        return (self.total_frames - self.current_frame) / fps

    def is_complete(self):
        """Apakah kalibrasi sudah selesai."""
        return self.current_frame >= self.total_frames


class MicrosleepClassifier:
    """
    State machine untuk klasifikasi status kelelahan.
    
    STATE FLOW:
        NORMAL → YAWNING → DROWSY → MICROSLEEP
          ↑         ↑         ↑         ↑
          └─────────┴─────────┴─────────┘
              (Reset jika kondisi normal kembali)
    
    LOGIKA KLASIFIKASI (per frame):
    1. Jika EAR < ear_threshold → ear_cnt += 1 (mata tertutup)
       Jika tidak → ear_cnt = 0 (mata terbuka, reset counter)
    
    2. Jika Lip > lip_threshold → lip_cnt += 1, catat timestamp (menguap)
       Jika tidak → lip_cnt = 0, in_yawn = False
    
    3. Hitung jumlah yawning dalam window 10 detik terakhir
    
    4. Klasifikasi berdasarkan prioritas (dari yang paling kritis):
       - ear_cnt >= 30 frame (1 detik) → MICROSLEEP
       - yawning_count >= 3 dalam 10s → DROWSY
       - Lip > lip_threshold → YAWNING
       - else → NORMAL
    
    ANTI-FALSE POSITIVE:
    - Kedipan normal: < 9 frame (0.3 detik) → diabaikan
    - Microsleep: >= 30 frame (1 detik) → trigger
    - Menguap valid: >= 1 frame dengan Lip > threshold → dihitung
    """

    NORMAL, YAWNING, DROWSY, MICROSLEEP = 'NORMAL', 'YAWNING', 'DROWSY', 'MICROSLEEP'

    def __init__(self, ear_thr, lip_thr, fps=config.FPS):
        """
        Inisialisasi classifier dengan threshold dari kalibrasi.
        
        Args:
            ear_thr: Batas EAR untuk deteksi mata tertutup
            lip_thr: Batas lip distance untuk deteksi menguap
            fps: Frame per second
        """
        self.ear_thr = ear_thr
        self.lip_thr = lip_thr
        self.ear_cnt = 0  # Counter frame EAR < threshold
        self.lip_cnt = 0  # Counter frame Lip > threshold
        self.yawn_ts = []  # Timestamp setiap yawning terdeteksi
        self.in_yawn = False  # Flag: sedang dalam state menguap
        self.state = self.NORMAL

    def update(self, ear, lip, face_ok=True):
        """
        Update state machine dengan data frame terbaru.
        
        Args:
            ear: Nilai EAR saat ini
            lip: Jarak bibir saat ini
            face_ok: Apakah wajah terdeteksi
        
        Returns:
            str: Status saat ini (NORMAL, YAWNING, DROWSY, MICROSLEEP, FACE_LOST)
        """
        if not face_ok:
            self.reset()
            return 'FACE_LOST'

        # =========================================================
        # 1. UPDATE EAR COUNTER (deteksi mata tertutup)
        # =========================================================
        # Jika EAR < threshold → mata tertutup → increment counter
        # Jika EAR >= threshold → mata terbuka → reset counter
        self.ear_cnt = self.ear_cnt + 1 if ear < self.ear_thr else 0

        # =========================================================
        # 2. UPDATE LIP COUNTER (deteksi menguap)
        # =========================================================
        if lip > self.lip_thr:
            if not self.in_yawn:
                # Pertama kali terdeteksi menguap, catat timestamp
                self.in_yawn = True
                self.yawn_ts.append(time.time())
            self.lip_cnt += 1
        else:
            self.in_yawn = False
            self.lip_cnt = 0

        # =========================================================
        # 3. HITUNG YAWNING DALAM WINDOW 10 DETIK
        # =========================================================
        now = time.time()
        # Filter timestamp yang masih dalam window 10 detik
        self.yawn_ts = [t for t in self.yawn_ts if now - t <= config.YAWN_WINDOW_SECONDS]

        # =========================================================
        # 4. KLASIFIKASI STATE (prioritas dari yang paling kritis)
        # =========================================================
        if self.ear_cnt >= config.MICROSLEEP_MIN_FRAMES:
            # Mata terpejam >= 30 frame (1 detik) → MICROSLEEP
            self.state = self.MICROSLEEP
        elif len(self.yawn_ts) >= config.YAWN_SEQUENCE_THRESHOLD:
            # 3x menguap dalam 10 detik → DROWSY
            self.state = self.DROWSY
        elif lip > self.lip_thr:
            # Sedang menguap (1x) → YAWNING
            self.state = self.YAWNING
        else:
            # Tidak ada tanda kantuk → NORMAL
            self.state = self.NORMAL
        
        return self.state

    def reset(self):
        """Reset semua counters dan state ke kondisi awal."""
        self.ear_cnt = 0
        self.lip_cnt = 0
        self.yawn_ts = []
        self.in_yawn = False
        self.state = self.NORMAL


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """
    Koordinator utama untuk manajemen status sistem.
    
    Menghubungkan:
    - DynamicCalibration: Untuk kalibrasi threshold
    - MicrosleepClassifier: Untuk klasifikasi status kantuk
    - Phone counter: Untuk deteksi distraksi HP
    
    Status yang dikelola:
    - NORMAL, YAWNING, DROWSY, MICROSLEEP (dari classifier)
    - DISTRACTED: HP terdeteksi > 5 detik
    - PHONE_ALERT: HP terdeteksi > 10 detik + alarm
    - FACE_LOST: Wajah tidak terdeteksi
    """

    def __init__(self, fps=config.FPS):
        self.calibration = DynamicCalibration(fps=fps)
        self.classifier = None
        self.phone_cnt = 0
        self.fps = fps
        self.break_triggered = False
        self.last_break = time.time()

    def calibrate(self, ear, lip):
        """Menjalankan proses kalibrasi dengan menambahkan sampel."""
        done = self.calibration.add_sample(ear, lip)
        if done and not self.calibration.is_complete():
            return False, self.calibration.progress(), self.calibration.remaining(self.fps)
        if done and self.classifier is None:
            self.classifier = MicrosleepClassifier(self.calibration.ear_threshold, self.calibration.lip_threshold, self.fps)
            return True, 1.0, 0
        if self.calibration.is_complete():
            return True, 1.0, 0
        return False, self.calibration.progress(), self.calibration.remaining(self.fps)

    def reset_calibration(self):
        """Reset kalibrasi untuk kalibrasi ulang."""
        self.calibration = DynamicCalibration(fps=self.fps)
        self.classifier = None
        self.phone_cnt = 0
        self.break_triggered = False

    def update(self, ear, lip, head_pose, phone_ok, face_ok=True):
        """
        Update status utama sistem dengan data frame terbaru.
        
        Args:
            ear: Nilai EAR saat ini
            lip: Jarak bibir saat ini
            head_pose: Dictionary {'yaw', 'pitch', 'roll'}
            phone_ok: bool, apakah HP terdeteksi
            face_ok: bool, apakah wajah terdeteksi
        
        Returns:
            str: Status saat ini
        """
        if not face_ok:
            self.phone_cnt = 0
            return 'FACE_LOST'

        # Cek deteksi HP
        if phone_ok:
            self.phone_cnt += 1
            if self.phone_cnt >= config.PHONE_LIMIT_FRAMES:
                return 'PHONE_ALERT'  # HP > 10 detik → alarm
            return 'DISTRACTED'  # HP terdeteksi → warning

        self.phone_cnt = 0
        state = self.classifier.update(ear, lip, face_ok)
        return state

    def check_break(self, tracker):
        """Mengecek apakah perlu menampilkan break reminder."""
        if time.time() - self.last_break > 1800:
            self.break_triggered = False
        if not self.break_triggered:
            msg = tracker.check_break_reminder()
            if msg:
                self.break_triggered = True
                self.last_break = time.time()
                return msg
        return None


# =============================================================================
# SESSION TRACKER
# =============================================================================

class SessionTracker:
    """Tracking statistik sesi belajar."""

    def __init__(self, fps=config.FPS):
        self.start = time.time()
        self.counts = {'NORMAL': 0, 'YAWNING': 0, 'DROWSY': 0, 'DISTRACTED': 0, 'MICROSLEEP': 0, 'PHONE_ALERT': 0, 'FACE_LOST': 0}
        self.microsleep_frames = 0
        self.yawning_frames = 0
        self.fps = fps
        self.ear_hist = []
        self.lip_hist = []

    def update(self, status, ear, lip):
        """Menambahkan data frame terbaru ke statistik."""
        self.counts[status] += 1
        if ear is not None:
            self.ear_hist.append(ear)
        if lip is not None:
            self.lip_hist.append(lip)
        if status == 'MICROSLEEP':
            self.microsleep_frames += 1
        elif status in ['YAWNING', 'DROWSY']:
            self.yawning_frames += 1

    def focus_score(self):
        """Persentase waktu fokus."""
        total = sum(self.counts.values())
        return (self.counts['NORMAL'] / total * 100) if total > 0 else 100.0

    def microsleep_seconds(self):
        """Total durasi microsleep dalam detik."""
        return self.microsleep_frames / self.fps

    def yawning_seconds(self):
        """Total durasi yawning dalam detik."""
        return self.yawning_frames / self.fps

    def check_break_reminder(self, ms_thr=3, yawn_thr=5):
        """Mengecek apakah perlu break reminder."""
        if self.microsleep_seconds() >= ms_thr:
            return f"Anda sudah {self.microsleep_seconds():.1f}s microsleep. Disarankan istirahat 5 menit!"
        elif self.yawning_seconds() >= yawn_thr:
            return f"Anda sudah {self.yawning_seconds():.1f}s menguap. Mungkin perlu istirahat?"
        return None

    def duration(self):
        """Durasi sesi dalam format HH:MM:SS."""
        elapsed = time.time() - self.start
        return f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"

    def reset(self):
        """Reset semua data untuk sesi baru."""
        self.start = time.time()
        self.counts = {k: 0 for k in self.counts}
        self.microsleep_frames = 0
        self.yawning_frames = 0
        self.ear_hist = []
        self.lip_hist = []


# =============================================================================
# ALARM (TTS)
# =============================================================================

MESSAGES = {
    'MICROSLEEP': 'Perhatian! Anda terdeteksi microsleep. Segera istirahat!',
    'PHONE_ALERT': 'Perhatian! Anda bermain HP terlalu lama. Fokus kembali belajar!',
}


class Alarm:
    """Sistem alarm suara menggunakan Text-to-Speech (pyttsx3)."""

    def __init__(self):
        self.playing = False
        self.stop_evt = threading.Event()
        self.thread = None
        self.engine = None
        self.lock = threading.Lock()

    def _init_engine(self):
        """Inisialisasi pyttsx3 engine dengan pengaturan suara."""
        if self.engine is None:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)

    def play(self, alarm_type='MICROSLEEP'):
        """Memulai alarm di thread terpisah."""
        if self.playing and self.thread and self.thread.is_alive():
            return
        self.stop_evt = threading.Event()
        self.playing = True
        self.thread = threading.Thread(target=self._speak, args=(alarm_type,), daemon=True)
        self.thread.start()

    def _speak(self, alarm_type):
        """Loop Text-to-Speech message sampai di-stop."""
        self._init_engine()
        msg = MESSAGES.get(alarm_type, MESSAGES['MICROSLEEP'])
        try:
            while not self.stop_evt.is_set():
                with self.lock:
                    self.engine.say(msg)
                    self.engine.runAndWait()
                if self.stop_evt.wait(2):
                    break
        finally:
            self.playing = False

    def stop(self):
        """Menghentikan alarm dan cleanup thread."""
        if not self.playing and not (self.thread and self.thread.is_alive()):
            return
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.playing = False
        if self.engine:
            self.engine.stop()
