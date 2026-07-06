"""
ui/overlay.py - Visualisasi Dinamis pada Frame OpenCV

File ini berisi kelas Overlay yang bertanggung jawab untuk:
1. Menggambar border berwarna di tepi frame sesuai status
2. Menampilkan teks status (FOKUS, PERINGATAN, BAHAYA)
3. Menampilkan statistik real-time (EAR, LIP, YAW, detections)
4. Menampilkan titik-titik landmark mata dan bibir

Warna border:
- Hijau (0, 255, 0): NORMAL/FOKUS
- Oranye (0, 165, 255): YAWNING, DROWSY
- Oranye tua (0, 140, 255): DISTRACTED
- Merah (0, 0, 255): MICROSLEEP, PHONE_ALERT
- Putih (255, 255, 255): FACE_LOST
"""

import cv2
import numpy as np

import config


class Overlay:
    """
    Kelas untuk menggambar overlay dinamis pada frame OpenCV.

    Semua method adalah class method (tidak perlu instance)
    untuk memudahkan pemanggilan langsung dari main loop.

    Elemen overlay:
    1. Border: Garis tebal di tepi frame (warna sesuai status)
    2. Status text: Teks besar di pojok kiri atas
    3. Stats: Statistik real-time di sisi kiri
    4. Landmarks: Titik-titik mata (merah) dan bibir (biru)
    """

    BORDER_THICKNESS = 8  # Ketebalan border (pixel)
    TEXT_SCALE = 1.0  # Skala font untuk status text
    TEXT_THICKNESS = 2  # Ketebalan font untuk status text
    STATS_SCALE = 0.6  # Skala font untuk stats
    STATS_THICKNESS = 1  # Ketebalan font untuk stats

    @classmethod
    def draw(cls, frame, status, info):
        """
        Main entry point — menggambar semua overlay pada frame.

        Args:
            frame: Frame BGR dari kamera (numpy array)
            status: Status saat ini (NORMAL, YAWNING, dll)
            info: Dictionary dengan data untuk ditampilkan
                  (ear, lip_distance, head_pose, landmarks, detections)

        Returns:
            numpy array: Frame yang sudah dimodifikasi dengan overlay
        """
        color = config.STATUS_COLORS.get(status, (255, 255, 255))

        cls._draw_border(frame, color)
        cls._draw_status_text(frame, status, color)
        cls._draw_stats(frame, info)
        cls._draw_landmarks(frame, info)

        return frame

    @classmethod
    def _draw_border(cls, frame, color):
        """
        Menggambar border berwarna di tepi frame.

        Args:
            frame: Frame BGR
            color: Warna BGR untuk border
        """
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), color, cls.BORDER_THICKNESS)

    @classmethod
    def _draw_status_text(cls, frame, status, color):
        """
        Menggambar teks status di pojok kiri atas frame.

        Mapping status ke label:
        - NORMAL → "FOKUS"
        - YAWNING → "PERINGATAN: MENGUAP"
        - DROWSY → "PERINGATAN: MENGANTUK"
        - DISTRACTED → "TERDISTRASI: MEMEGANG HP"
        - MICROSLEEP → "BAHAYA: MICROSLEEP!"
        - PHONE_ALERT → "BAHAYA: HP TERLALU LAMA!"
        - FACE_LOST → "WAJAH TIDAK TERDETEKSI"

        Args:
            frame: Frame BGR
            status: Status saat ini
            color: Warna BGR untuk teks
        """
        status_labels = {
            'NORMAL': 'FOKUS',
            'YAWNING': 'PERINGATAN: MENGUAP',
            'DROWSY': 'PERINGATAN: MENGANTUK',
            'DISTRACTED': 'TERDISTRASI: MEMEGANG HP',
            'MICROSLEEP': 'BAHAYA: MICROSLEEP!',
            'PHONE_ALERT': 'BAHAYA: HP TERLALU LAMA!',
            'FACE_LOST': 'WAJAH TIDAK TERDETEKSI',
        }

        label = status_labels.get(status, status)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, cls.TEXT_SCALE, cls.TEXT_THICKNESS)[0]

        x = 20
        y = 40
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, cls.TEXT_SCALE, color, cls.TEXT_THICKNESS)

    @classmethod
    def _draw_stats(cls, frame, info):
        """
        Menggambar statistik real-time di sisi kiri frame.

        Stats yang ditampilkan:
        - EAR: Eye Aspect Ratio saat ini
        - LIP: Lip Distance saat ini
        - YAW: Sudut yaw (orientasi kepala kiri-kanan)
        - Detections: Objek yang terdeteksi YOLO (person, phone, book)
        - STATE: Status saat ini

        Args:
            frame: Frame BGR
            info: Dictionary dengan data stats
        """
        y_offset = 80  # Posisi Y awal untuk stats
        line_height = 25  # Jarak antar baris
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = cls.STATS_SCALE
        thickness = cls.STATS_THICKNESS
        color = (255, 255, 255)  # Putih

        stats = []

        # Tambahkan EAR jika tersedia
        if info.get('ear') is not None:
            stats.append(f"EAR: {info['ear']:.4f}")

        # Tambahkan Lip Distance jika tersedia
        if info.get('lip_distance') is not None:
            stats.append(f"LIP: {info['lip_distance']:.4f}")

        # Tambahkan Yaw angle
        if info.get('head_pose'):
            yaw = info['head_pose'].get('yaw', 0)
            stats.append(f"YAW: {yaw:.1f}")

        # Tambahkan detections dari YOLO
        if info.get('detections'):
            for det in info['detections']:
                cls_name = det.get('class', '')
                conf = det.get('conf', 0)
                stats.append(f"{cls_name}: {conf:.2f}")

        # Tambahkan status saat ini
        if info.get('state'):
            stats.append(f"STATE: {info['state']}")

        # Gambar semua stats
        for i, stat in enumerate(stats):
            y = y_offset + (i * line_height)
            cv2.putText(frame, stat, (20, y), font, scale, color, thickness)

    @classmethod
    def _draw_landmarks(cls, frame, info):
        """
        Menggambar titik-titik landmark mata dan bibir.

        Landmarks yang ditampilkan:
        - Mata kiri: 6 titik (merah)
        - Mata kanan: 6 titik (merah)
        - Bibir: 2 titik (biru)

        Args:
            frame: Frame BGR
            info: Dictionary dengan landmarks
        """
        if not info.get('landmarks'):
            return

        h, w = frame.shape[:2]
        landmarks = info['landmarks']

        # Indeks landmark mata (sama dengan di face_analyzer.py)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        lip_indices = [13, 14]  # Bibir atas dan bawah

        # Gambar titik mata (merah)
        for idx in left_eye_indices + right_eye_indices:
            pt = landmarks[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Gambar titik bibir (biru)
        for idx in lip_indices:
            pt = landmarks[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
