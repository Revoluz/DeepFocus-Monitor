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
    def __init__(self, camera_index, width=640, height=480):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(1 / 30)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


def main():
    camera_thread = CameraThread(config.CAMERA_INDEX)
    camera_thread.start()

    face_analyzer = FaceAnalyzer(fps=config.FPS)
    state_manager = StateManager(fps=config.FPS)
    session_tracker = SessionTracker()
    alarm = Alarm()
    yolo_model = YOLO('yolov8n.pt')

    dashboard = SimpleDashboard(session_tracker, state_manager, alarm)

    print("Sistem dimulai. Kalibrasi otomatis dalam 5 detik...")
    print("Instruksi: Hadapkan wajah ke kamera, mata terbuka, mulut tertutup.")
    print("Jangan bicara atau menguap selama kalibrasi.")

    calibrating = True

    def process_frame():
        nonlocal calibrating

        frame = camera_thread.get_frame()
        if frame is None:
            return None, None

        if dashboard.recalibrating:
            calibrating = True
            dashboard.recalibrating = False

        if calibrating:
            face_data = face_analyzer.analyze(frame)

            if face_data['face_detected']:
                done, progress, remaining = state_manager.calibrate(
                    face_data['ear'],
                    face_data['lip_distance']
                )

                if done:
                    calibrating = False
                    dashboard.paused = False
                    print(f"Kalibrasi selesai!")
                    print(f"  EAR threshold: {state_manager.classifier.ear_threshold:.4f}")
                    print(f"  LIP threshold: {state_manager.classifier.lip_threshold:.4f}")
                else:
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

                    progress_bar_width = 200
                    progress_width = int(progress * progress_bar_width)
                    cv2.rectangle(frame, (w // 2 - 100, h // 2 + 60),
                                  (w // 2 - 100 + progress_bar_width, h // 2 + 80),
                                  (255, 255, 255), 2)
                    cv2.rectangle(frame, (w // 2 - 100, h // 2 + 60),
                                  (w // 2 - 100 + progress_width, h // 2 + 80),
                                  (0, 255, 0), -1)

            return frame, face_data

        elif dashboard.paused:
            cv2.putText(frame, "PAUSED - Tekan tombol Resume di Dashboard",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return frame, None

        else:
            face_data = face_analyzer.analyze(frame)

            yolo_results = yolo_model(frame, classes=config.YOLO_CLASSES, conf=config.YOLO_CONFIDENCE, verbose=False)
            annotated_frame = yolo_results[0].plot()

            detections = []
            phone_detected = False

            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]

                    detections.append({'class': cls_name, 'conf': conf})

                    if cls_id == 67:
                        phone_detected = True

            status = state_manager.update(
                ear=face_data['ear'],
                lip_distance=face_data['lip_distance'],
                head_pose=face_data['head_pose'],
                phone_detected=phone_detected,
                face_detected=face_data['face_detected']
            )

            session_tracker.update(status, face_data['ear'], face_data['lip_distance'])

            if status in ['MICROSLEEP', 'PHONE_ALERT']:
                alarm.play(status)
            else:
                alarm.stop()

            info = {
                'ear': face_data['ear'],
                'lip_distance': face_data['lip_distance'],
                'head_pose': face_data['head_pose'],
                'landmarks': face_data['landmarks'],
                'detections': detections,
                'state': status,
            }

            annotated_frame = Overlay.draw(annotated_frame, status, info)

            dashboard.update(status, face_data['ear'], face_data['lip_distance'], face_data['head_pose'])

            return annotated_frame, face_data

    def show_frame():
        while True:
            frame, _ = process_frame()
            if frame is not None:
                cv2.imshow('Drowsiness & Study Distraction Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    show_thread = threading.Thread(target=show_frame, daemon=True)
    show_thread.start()

    dashboard.run()

    alarm.stop()
    camera_thread.stop()
    show_thread.join(timeout=1)
    cv2.destroyAllWindows()
    print("Sistem dihentikan.")


if __name__ == '__main__':
    main()
