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
import csv
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
    """Menghitung jarak Euclidean antara dua titik 2D."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# =============================================================================
# FACE ANALYZER
# =============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIP_POINTS = [13, 14]
HEAD_POSE_3D = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.double)
HEAD_POSE_2D = [1, 9, 33, 263, 61, 291]


def ensure_model(path):
    if not os.path.exists(path):
        print("Mengunduh model Face Landmarker...")
        urllib.request.urlretrieve(MODEL_URL, path)


class FaceAnalyzer:
    """Analisis biometrik wajah: EAR, Lip Distance, Head Pose."""

    def __init__(self, fps=30):
        ensure_model(MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, running_mode=vision.RunningMode.VIDEO,
            num_faces=1, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.fps = fps
        self.cam_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.double)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.double)

    def calculate_ear(self, landmarks, side='left'):
        """Hitung Eye Aspect Ratio (EAR) untuk satu mata."""
        indices = LEFT_EYE if side == 'left' else RIGHT_EYE
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
        v1 = euclidean_distance(p2, p6)
        v2 = euclidean_distance(p3, p5)
        h = euclidean_distance(p1, p4)
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def calculate_ear_avg(self, landmarks):
        """Rata-rata EAR dari kedua mata."""
        return (self.calculate_ear(landmarks, 'left') + self.calculate_ear(landmarks, 'right')) / 2.0

    def calculate_lip_dist(self, landmarks):
        """Jarak absolut antara bibir atas dan bawah."""
        return euclidean_distance(landmarks[LIP_POINTS[0]], landmarks[LIP_POINTS[1]])

    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimasi orientasi kepala (yaw, pitch, roll) menggunakan solvePnP."""
        h, w = frame_shape[:2]
        img_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_POSE_2D], dtype=np.double)
        success, rot_vec, trans_vec = cv2.solvePnP(HEAD_POSE_3D, img_pts, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        return {'yaw': float(euler[1][0]), 'pitch': float(euler[0][0]), 'roll': float(euler[2][0])}

    def analyze(self, frame):
        """Pipeline lengkap analisis satu frame."""
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
    """Kalibrasi otomatis threshold per pengguna (5 detik)."""

    def __init__(self, duration=config.CALIBRATION_DURATION, fps=config.FPS):
        self.total_frames = duration * fps
        self.ear_samples = []
        self.lip_samples = []
        self.current_frame = 0
        self.ear_threshold = None
        self.lip_threshold = None

    def add_sample(self, ear, lip):
        if self.current_frame >= self.total_frames:
            return True
        self.ear_samples.append(ear)
        self.lip_samples.append(lip)
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.ear_threshold = np.mean(self.ear_samples) - (config.THRESHOLD_STD_MULTIPLIER * np.std(self.ear_samples))
            self.lip_threshold = np.mean(self.lip_samples) + (config.THRESHOLD_STD_MULTIPLIER * np.std(self.lip_samples))
            return True
        return False

    def progress(self):
        return self.current_frame / self.total_frames

    def remaining(self, fps):
        return (self.total_frames - self.current_frame) / fps

    def is_complete(self):
        return self.current_frame >= self.total_frames


class MicrosleepClassifier:
    """State machine: NORMAL → YAWNING → DROWSY → MICROSLEEP."""

    NORMAL, YAWNING, DROWSY, MICROSLEEP = 'NORMAL', 'YAWNING', 'DROWSY', 'MICROSLEEP'

    def __init__(self, ear_thr, lip_thr, fps=config.FPS):
        self.ear_thr = ear_thr
        self.lip_thr = lip_thr
        self.ear_cnt = 0
        self.lip_cnt = 0
        self.yawn_ts = []
        self.state = self.NORMAL

    def update(self, ear, lip, face_ok=True):
        if not face_ok:
            self.reset()
            return 'FACE_LOST'

        self.ear_cnt = self.ear_cnt + 1 if ear < self.ear_thr else 0

        if lip > self.lip_thr:
            if self.lip_cnt == 0:
                self.yawn_ts.append(time.time())
            self.lip_cnt += 1
        else:
            self.lip_cnt = 0

        recent = [t for t in self.yawn_ts if time.time() - t <= config.YAWN_WINDOW_SECONDS]
        self.yawn_ts = recent

        if self.ear_cnt >= config.MICROSLEEP_MIN_FRAMES:
            self.state = self.MICROSLEEP
        elif len(recent) >= config.YAWN_SEQUENCE_THRESHOLD:
            self.state = self.DROWSY
        elif lip > self.lip_thr:
            self.state = self.YAWNING
        else:
            self.state = self.NORMAL
        return self.state

    def reset(self):
        self.ear_cnt = 0
        self.lip_cnt = 0
        self.yawn_ts = []
        self.state = self.NORMAL


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """Koordinator status: kalibrasi, classifier, phone counter, break reminder."""

    def __init__(self, fps=config.FPS):
        self.calibration = DynamicCalibration(fps=fps)
        self.classifier = None
        self.phone_cnt = 0
        self.fps = fps
        self.break_triggered = False
        self.last_break = time.time()

    def calibrate(self, ear, lip):
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
        self.calibration = DynamicCalibration(fps=self.fps)
        self.classifier = None
        self.phone_cnt = 0
        self.break_triggered = False

    def update(self, ear, lip, head_pose, phone_ok, face_ok=True):
        if not face_ok:
            self.phone_cnt = 0
            return 'FACE_LOST'
        self.phone_cnt = self.phone_cnt + 1 if phone_ok else 0
        state = self.classifier.update(ear, lip, face_ok)
        if phone_ok:
            if self.phone_cnt >= config.PHONE_LIMIT_FRAMES:
                return 'PHONE_ALERT'
            elif self.phone_cnt >= config.PHONE_DISTRACTED_FRAMES:
                return 'DISTRACTED'
        return state

    def check_break(self, tracker):
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

    def __init__(self):
        self.start = time.time()
        self.counts = {'NORMAL': 0, 'YAWNING': 0, 'DROWSY': 0, 'DISTRACTED': 0, 'MICROSLEEP': 0, 'PHONE_ALERT': 0, 'FACE_LOST': 0}
        self.microsleep_total = 0
        self.yawning_total = 0
        self.ear_hist = []
        self.lip_hist = []

    def update(self, status, ear, lip):
        self.counts[status] += 1
        if ear is not None:
            self.ear_hist.append(ear)
        if lip is not None:
            self.lip_hist.append(lip)
        if status == 'MICROSLEEP':
            self.microsleep_total += 1
        elif status in ['YAWNING', 'DROWSY']:
            self.yawning_total += 1

    def focus_score(self):
        total = sum(self.counts.values())
        return (self.counts['NORMAL'] / total * 100) if total > 0 else 100.0

    def check_break_reminder(self, ms_thr=3, yawn_thr=5):
        if self.microsleep_total >= ms_thr:
            return f"Anda sudah {self.microsleep_total}x microsleep. Disarankan istirahat 5 menit!"
        elif self.yawning_total >= yawn_thr:
            return f"Anda sudah {self.yawning_total}x menguap. Mungkin perlu istirahat?"
        return None

    def duration(self):
        elapsed = time.time() - self.start
        return f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"

    def reset(self):
        self.start = time.time()
        self.counts = {k: 0 for k in self.counts}
        self.microsleep_total = 0
        self.yawning_total = 0
        self.ear_hist = []
        self.lip_hist = []

    def export_csv(self, filename='session_log.csv'):
        if not self.ear_hist:
            return False
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'ear', 'lip_distance'])
            for i, (e, l) in enumerate(zip(self.ear_hist, self.lip_hist)):
                w.writerow([i, e, l])
        return True


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

    def _init_engine(self):
        if self.engine is None:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)

    def play(self, alarm_type='MICROSLEEP'):
        if self.playing:
            return
        self.stop_evt.clear()
        self.playing = True
        self.thread = threading.Thread(target=self._speak, args=(alarm_type,), daemon=True)
        self.thread.start()

    def _speak(self, alarm_type):
        self._init_engine()
        msg = MESSAGES.get(alarm_type, MESSAGES['MICROSLEEP'])
        while not self.stop_evt.is_set():
            self.engine.say(msg)
            self.engine.runAndWait()
            self.stop_evt.wait(2)

    def stop(self):
        if not self.playing:
            return
        self.stop_evt.set()
        self.playing = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.engine:
            self.engine.stop()
