"""
main.py - Entry Point DeepFocus Monitor

File utama yang mengintegrasikan semua modul:
- CameraThread: Capture frame dari webcam
- Overlay: Visualisasi dinamis pada frame
- main(): Inisialisasi, kalibrasi, main loop, cleanup
"""

import cv2
import threading
from ultralytics import YOLO

from config import *
from detector import FaceAnalyzer, StateManager, SessionTracker, Alarm
from dashboard import SimpleDashboard


class CameraThread(threading.Thread):
    """Thread untuk capture frame dari webcam."""

    def __init__(self, idx, w=640, h=480):
        super().__init__()
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = f.copy()
            import time; time.sleep(1/30)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


class Overlay:
    """Visualisasi dinamis pada frame OpenCV."""

    @staticmethod
    def draw(frame, status, info):
        color = STATUS_COLORS.get(status, (255, 255, 255))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), color, 8)

        labels = {'NORMAL': 'FOKUS', 'YAWNING': 'PERINGATAN: MENGUAP', 'DROWSY': 'PERINGATAN: MENGANTUK', 'DISTRACTED': 'TERDISTRASI: MEMEGANG HP', 'MICROSLEEP': 'BAHAYA: MICROSLEEP!', 'PHONE_ALERT': 'BAHAYA: HP TERLALU LAMA!', 'FACE_LOST': 'WAJAH TIDAK TERDETEKSI'}
        cv2.putText(frame, labels.get(status, status), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        y = 80
        for txt in [f"EAR: {info.get('ear', 0):.4f}" if info.get('ear') else None, f"LIP: {info.get('lip_distance', 0):.4f}" if info.get('lip_distance') else None, f"YAW: {info.get('head_pose', {}).get('yaw', 0):.1f}" if info.get('head_pose') else None]:
            if txt:
                cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 25

        if info.get('landmarks'):
            lm = info['landmarks']
            for idx in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (0, 0, 255), -1)
            for idx in [13, 14]:
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (255, 0, 0), -1)

        return frame


def main():
    """
    Fungsi utama DeepFocus Monitor.
    
    ALUR EKSEKUSI:
    1. START → Inisialisasi semua modul
    2. CameraThread → Capture frame dari webcam (background)
    3. KALIBRASI (5 detik) → Kumpulkan 150 sampel EAR & Lip
    4. MAIN LOOP → Process frame → Update status → Update dashboard
    5. EXIT (tekan 'q') → Cleanup dan exit
    
    THREADING:
    - Main Thread: Dashboard GUI (tkinter harus di main thread)
    - CameraThread: Capture frame dari webcam
    - show_thread: Process frame + show OpenCV window
    - Alarm Thread: TTS alarm (pyttsx3, spawned saat perlu)
    """
    # =========================================================
    # 1. INISIALISASI SEMUA MODUL
    # =========================================================
    cam = CameraThread(CAMERA_INDEX)  # Thread capture webcam
    cam.start()

    face = FaceAnalyzer(fps=FPS)      # Analisis wajah (EAR, Lip, Head Pose)
    state = StateManager(fps=FPS)     # Manajemen status & kalibrasi
    tracker = SessionTracker(fps=FPS) # Tracking statistik sesi
    alarm = Alarm()                   # Alarm suara (TTS)
    yolo = YOLO('yolov8n.pt')         # Deteksi objek (person, phone, book)
    dashboard = SimpleDashboard(tracker, state, alarm)  # GUI monitoring

    print("Sistem dimulai. Kalibrasi otomatis dalam 5 detik...")
    print("Instruksi: Hadapkan wajah ke kamera, mata terbuka, mulut tertutup.")

    calibrating = True
    last_status = None

    # =========================================================
    # 2. FUNGSI PROCESS FRAME (dipanggil setiap frame)
    # =========================================================
    def process():
        """
        Memproses satu frame melalui pipeline lengkap.
        
        ALUR PER FRAME:
        1. Jika kalibrasi → kumpulkan sampel, tampilkan countdown
        2. Jika paused → tampilkan "PAUSED", skip analisis
        3. Jika normal → analisis wajah + YOLO + update status
        4. Update dashboard & overlay
        5. Return frame yang sudah dimodifikasi
        """
        nonlocal calibrating, last_status
        frame = cam.get_frame()
        if frame is None:
            return None

        if dashboard.recalibrating:
            calibrating = True
            dashboard.recalibrating = False

        # =====================================================
        # FASE 1: KALIBRASI (5 detik pertama)
        # =====================================================
        if calibrating:
            fd = face.analyze(frame)
            if fd['face_detected']:
                done, prog, rem = state.calibrate(fd['ear'], fd['lip_distance'])
                if done:
                    calibrating = False
                    dashboard.paused = False
                    print(f"Kalibrasi selesai! EAR: {state.classifier.ear_thr:.4f}, LIP: {state.classifier.lip_thr:.4f}")
                else:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"Kalibrasi... {int(rem+1)}s", (w//2-120, h//2-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(frame, "Mata terbuka, mulut tertutup", (w//2-180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, "Jangan bicara atau menguap", (w//2-160, h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    pw = int(prog * 200)
                    cv2.rectangle(frame, (w//2-100, h//2+60), (w//2+100, h//2+80), (255,255,255), 2)
                    cv2.rectangle(frame, (w//2-100, h//2+60), (w//2-100+pw, h//2+80), (0,255,0), -1)
            return frame

        # =====================================================
        # FASE 2: PAUSED (user pause monitoring)
        # =====================================================
        if dashboard.paused:
            cv2.putText(frame, "PAUSED - Tekan Resume di Dashboard", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            return frame

        # =====================================================
        # FASE 3: MONITORING NORMAL
        # =====================================================
        
        # 3a. Analisis wajah (EAR, Lip Distance, Head Pose)
        fd = face.analyze(frame)
        
        # 3b. Deteksi objek dengan YOLOv8 (person, phone, book)
        res = yolo(frame, classes=YOLO_CLASSES, conf=YOLO_CONFIDENCE, verbose=False)
        ann = res[0].plot()

        # 3c. Ekstrak detections & cek apakah HP terdeteksi
        dets, phone = [], False
        for r in res:
            for box in r.boxes:
                cid, conf, name = int(box.cls[0]), float(box.conf[0]), r.names[int(box.cls[0])]
                dets.append({'class': name, 'conf': conf})
                if cid == 67:  # 67 = cell phone (COCO class)
                    phone = True

        # 3d. Update status sistem (NORMAL, YAWNING, DROWSY, MICROSLEEP, DISTRACTED, PHONE_ALERT)
        status = state.update(fd['ear'], fd['lip_distance'], fd['head_pose'], phone, fd['face_detected'])
        
        # 3e. Update statistik sesi
        tracker.update(status, fd['ear'], fd['lip_distance'])

        # 3f. Print status jika berubah
        if status != last_status:
            print(f"Status terdeteksi: {status}")
            last_status = status

        # 3g. Kontrol alarm (hanya untuk MICROSLEEP & PHONE_ALERT)
        if status in ['MICROSLEEP', 'PHONE_ALERT']:
            alarm.play(status)
        else:
            alarm.stop()

        # 3h. Update dashboard GUI & overlay frame
        info = {'ear': fd['ear'], 'lip_distance': fd['lip_distance'], 'head_pose': fd['head_pose'], 'landmarks': fd['landmarks'], 'detections': dets, 'state': status}
        dashboard.update(status, fd['ear'], fd['lip_distance'], fd['head_pose'])

        return Overlay.draw(ann, status, info)

    # =========================================================
    # 3. THREAD UNTUK SHOW FRAME (OPENCV WINDOW)
    # =========================================================
    def show_loop():
        """Loop menampilkan frame di OpenCV window. Exit saat user tekan 'q'."""
        while True:
            f = process()
            if f is not None:
                cv2.imshow('DeepFocus Monitor', f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    show = threading.Thread(target=show_loop, daemon=True)
    show.start()
    
    # =========================================================
    # 4. JALANKAN DASHBOARD DI MAIN THREAD (requirement tkinter)
    # =========================================================
    dashboard.run()

    # =========================================================
    # 5. CLEANUP SAAT EXIT
    # =========================================================
    alarm.stop()
    cam.stop()
    show.join(timeout=1)
    cv2.destroyAllWindows()
    print("Sistem dihentikan.")


if __name__ == '__main__':
    main()
