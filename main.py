"""
main.py - Entry Point Aplikasi DeepFocus Monitor

File ini adalah titik masuk utama aplikasi yang mengintegrasikan semua modul:
1. FaceAnalyzer: Analisis biometrik wajah (EAR, Lip, Head Pose)
2. StateManager: Manajemen status dan kalibrasi
3. SessionTracker: Tracking statistik sesi
4. Alarm: Sistem alarm suara (TTS)
5. Overlay: Visualisasi dinamis pada frame
6. Dashboard: GUI monitoring real-time

Arsitektur Threading:
┌─────────────────────────────────────────────────────────┐
│ Main Thread (Dashboard GUI)                             │
│   - tkinter mainloop                                    │
│   - Update widget dengan data terbaru                   │
├─────────────────────────────────────────────────────────┤
│ CameraThread (Background)                               │
│   - Capture frame dari webcam                           │
│   - Thread-safe frame access dengan lock                │
├─────────────────────────────────────────────────────────┤
│ show_thread (Background)                                │
│   - Process frame (FaceAnalyzer + YOLO)                 │
│   - Update status (StateManager)                        │
│   - Draw overlay pada frame                             │
│   - Show frame di OpenCV window                         │
├─────────────────────────────────────────────────────────┤
│ Alarm Thread (Background, spawned saat perlu)           │
│   - Text-to-Speech alarm (pyttsx3)                      │
└─────────────────────────────────────────────────────────┘

Alur Eksekusi:
1. START → Inisialisasi semua modul
2. KALIBRASI (5 detik) → Kumpulkan 150 sampel EAR & Lip
3. MAIN LOOP → Process frame → Update status → Update dashboard
4. EXIT (tekan 'q') → Cleanup dan exit
"""

import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

from core.face_analyzer import FaceAnalyzer
from core.state_manager import StateManager
from core.session_tracker import SessionTracker
from core.alarm import Alarm
from ui.overlay import Overlay
from ui.dashboard import SimpleDashboard
import config


class CameraThread(threading.Thread):
    """
    Thread untuk capture frame dari webcam secara background.

    Cara kerja:
    1. Buka koneksi kamera dengan cv2.VideoCapture
    2. Loop terus-menerus: baca frame → copy ke self.frame → sleep
    3. Thread-safe access ke frame menggunakan threading.Lock

    Keuntungan:
    - Kamera selalu aktif, tidak blocking saat processing
    - Frame terbaru selalu tersedia untuk diproses
    - Bisa di-stop dengan aman saat exit
    """

    def __init__(self, camera_index, width=640, height=480):
        """
        Inisialisasi CameraThread.

        Args:
            camera_index: Indeks webcam (dari config.CAMERA_INDEX)
            width: Lebar frame (default: 640)
            height: Tinggi frame (default: 480)
        """
        super().__init__()
        # Buka koneksi kamera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None  # Frame terbaru
        self.running = True  # Flag untuk kontrol loop
        self.lock = threading.Lock()  # Lock untuk thread-safe access

    def run(self):
        """
        Main loop thread: baca frame dari kamera terus-menerus.

        Loop akan berjalan sampai self.running = False (saat stop dipanggil).
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()  # Copy frame untuk thread safety
            time.sleep(1 / 30)  # Sleep ~33ms untuk ~30 FPS

    def get_frame(self):
        """
        Mengambil frame terbaru dengan thread-safe access.

        Returns:
            numpy array/None: Frame terbaru atau None jika belum ada
        """
        with self.lock:
            return self.frame

    def stop(self):
        """
        Menghentikan thread dan release kamera.
        """
        self.running = False
        self.cap.release()


def main():
    """
    Fungsi utama aplikasi DeepFocus Monitor.

    Alur eksekusi:
    1. Inisialisasi semua modul (FaceAnalyzer, StateManager, dll)
    2. Start CameraThread untuk capture frame
    3. Jalankan Dashboard GUI di main thread
    4. Process frame di show_thread (background)
    5. Loop sampai user menekan 'q'
    6. Cleanup dan exit
    """
    # =====================================================================
    # INISIALISASI
    # =====================================================================

    # Start camera thread di background
    camera_thread = CameraThread(config.CAMERA_INDEX)
    camera_thread.start()

    # Inisialisasi semua modul
    face_analyzer = FaceAnalyzer(fps=config.FPS)
    state_manager = StateManager(fps=config.FPS)
    session_tracker = SessionTracker()
    alarm = Alarm()
    yolo_model = YOLO('yolov8n.pt')  # Load YOLOv8 pre-trained model

    # Buat dashboard GUI (akan dijalankan di main thread)
    dashboard = SimpleDashboard(session_tracker, state_manager, alarm)

    # Print instruksi startup
    print("Sistem dimulai. Kalibrasi otomatis dalam 5 detik...")
    print("Instruksi: Hadapkan wajah ke kamera, mata terbuka, mulut tertutup.")
    print("Jangan bicara atau menguap selama kalibrasi.")

    # Flag untuk tracking state kalibrasi
    calibrating = True

    # =====================================================================
    # FUNGSI PROCESS FRAME
    # =====================================================================

    def process_frame():
        """
        Memproses satu frame: analisis wajah, YOLO, update status.

        Returns:
            tuple: (frame, face_data)
                - frame: Frame yang sudah dimodifikasi dengan overlay
                - face_data: Dictionary hasil analisis wajah
        """
        nonlocal calibrating

        # Ambil frame terbaru dari camera thread
        frame = camera_thread.get_frame()
        if frame is None:
            return None, None

        # Cek apakah user meminta kalibrasi ulang dari dashboard
        if dashboard.recalibrating:
            calibrating = True
            dashboard.recalibrating = False

        # =================================================================
        # FASE KALIBRASI
        # =================================================================
        if calibrating:
            face_data = face_analyzer.analyze(frame)

            if face_data['face_detected']:
                # Tambahkan sampel ke kalibrasi
                done, progress, remaining = state_manager.calibrate(
                    face_data['ear'],
                    face_data['lip_distance']
                )

                if done:
                    # Kalibrasi selesai
                    calibrating = False
                    dashboard.paused = False
                    print(f"Kalibrasi selesai!")
                    print(f"  EAR threshold: {state_manager.classifier.ear_threshold:.4f}")
                    print(f"  LIP threshold: {state_manager.classifier.lip_threshold:.4f}")
                else:
                    # Tampilkan UI kalibrasi di frame
                    h, w = frame.shape[:2]
                    text = f"Kalibrasi... {int(remaining + 1)}s"
                    instruction1 = "Mata terbuka, mulut tertutup"
                    instruction2 = "Jangan bicara atau menguap"

                    cv2.putText(frame, text, (w // 2 - 120, h // 2 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, instruction1, (w // 2 - 180, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, instruction2, (w // 2 - 160, h // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Progress bar
                    progress_bar_width = 200
                    progress_width = int(progress * progress_bar_width)
                    cv2.rectangle(frame, (w // 2 - 100, h // 2 + 60),
                                  (w // 2 - 100 + progress_bar_width, h // 2 + 80),
                                  (255, 255, 255), 2)
                    cv2.rectangle(frame, (w // 2 - 100, h // 2 + 60),
                                  (w // 2 - 100 + progress_width, h // 2 + 80),
                                  (0, 255, 0), -1)

            return frame, face_data

        # =================================================================
        # FASE PAUSED
        # =================================================================
        elif dashboard.paused:
            # Tampilkan teks PAUSED di frame
            cv2.putText(frame, "PAUSED - Tekan tombol Resume di Dashboard",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return frame, None

        # =================================================================
        # FASE MONITORING NORMAL
        # =================================================================
        else:
            # Analisis wajah
            face_data = face_analyzer.analyze(frame)

            # Deteksi objek dengan YOLOv8
            yolo_results = yolo_model(frame, classes=config.YOLO_CLASSES, conf=config.YOLO_CONFIDENCE, verbose=False)
            annotated_frame = yolo_results[0].plot()

            # Ekstrak detections
            detections = []
            phone_detected = False

            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]

                    detections.append({'class': cls_name, 'conf': conf})

                    # Cek apakah terdeteksi cell phone (class 67)
                    if cls_id == 67:
                        phone_detected = True

            # Update status sistem
            status = state_manager.update(
                ear=face_data['ear'],
                lip_distance=face_data['lip_distance'],
                head_pose=face_data['head_pose'],
                phone_detected=phone_detected,
                face_detected=face_data['face_detected']
            )

            # Update session tracker
            session_tracker.update(status, face_data['ear'], face_data['lip_distance'])

            # Kontrol alarm (hanya untuk MICROSLEEP dan PHONE_ALERT)
            if status in ['MICROSLEEP', 'PHONE_ALERT']:
                alarm.play(status)
            else:
                alarm.stop()

            # Siapkan info untuk overlay
            info = {
                'ear': face_data['ear'],
                'lip_distance': face_data['lip_distance'],
                'head_pose': face_data['head_pose'],
                'landmarks': face_data['landmarks'],
                'detections': detections,
                'state': status,
            }

            # Gambar overlay pada frame
            annotated_frame = Overlay.draw(annotated_frame, status, info)

            # Update dashboard GUI
            dashboard.update(status, face_data['ear'], face_data['lip_distance'], face_data['head_pose'])

            return annotated_frame, face_data

    # =====================================================================
    # THREAD UNTUK SHOW FRAME (OPENCV WINDOW)
    # =====================================================================

    def show_frame():
        """
        Loop untuk menampilkan frame di OpenCV window.

        Berjalan di thread terpisah agar tidak blocking dashboard GUI.
        Exit saat user menekan 'q'.
        """
        while True:
            frame, _ = process_frame()
            if frame is not None:
                cv2.imshow('Drowsiness & Study Distraction Detection', frame)

            # Cek tombol 'q' untuk exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Start show thread
    show_thread = threading.Thread(target=show_frame, daemon=True)
    show_thread.start()

    # =====================================================================
    # JALANKAN DASHBOARD DI MAIN THREAD
    # =====================================================================
    # Dashboard harus di main thread karena requirement tkinter
    dashboard.run()

    # =====================================================================
    # CLEANUP SAAT EXIT
    # =====================================================================
    alarm.stop()
    camera_thread.stop()
    show_thread.join(timeout=1)
    cv2.destroyAllWindows()
    print("Sistem dihentikan.")


if __name__ == '__main__':
    main()
