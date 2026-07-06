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


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka kamera indeks {config.CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_analyzer = FaceAnalyzer(fps=config.FPS)
    state_manager = StateManager(fps=config.FPS)
    session_tracker = SessionTracker()
    alarm = Alarm()
    yolo_model = YOLO('yolov8n.pt')

    dashboard = SimpleDashboard(session_tracker, state_manager, alarm)
    dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
    dashboard_thread.start()

    print("Sistem dimulai. Kalibrasi otomatis dalam 5 detik...")
    print("Instruksi: Hadapkan wajah ke kamera, mata terbuka, mulut tertutup.")
    print("Jangan bicara atau menguap selama kalibrasi.")

    calibrating = True
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

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

        elif dashboard.paused:
            cv2.putText(frame, "PAUSED - Tekan tombol Resume di Dashboard",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Drowsiness & Study Distraction Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

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
            frame = annotated_frame

            dashboard.update(status, face_data['ear'], face_data['lip_distance'], face_data['head_pose'])

        cv2.imshow('Drowsiness & Study Distraction Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Sistem dihentikan.")


if __name__ == '__main__':
    main()
