import cv2
import mediapipe as mp
import os
import time
import urllib.request
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Inisialisasi YOLOv8 (Pre-trained)
model_yolo = YOLO('yolov8n.pt') 

# 2. Inisialisasi MediaPipe Face Landmarker (Tasks API)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"


def ensure_face_landmarker_model(model_path: str) -> None:
    if os.path.exists(model_path):
        return

    print("Mengunduh model Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Model tersimpan di: {model_path}")


ensure_face_landmarker_model(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(2)

with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- BAGIAN YOLOv8 (Deteksi HP) ---
        # Kelas 67 adalah 'cell phone' dalam dataset COCO
        yolo_results = model_yolo(frame, classes=[0,67], conf=0.5, verbose=False)
        annotated_frame = yolo_results[0].plot()

        # --- BAGIAN MEDIAPIPE TASKS (Face Landmarker) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.monotonic() * 1000)
        mesh_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if mesh_results.face_landmarks:
            h, w = annotated_frame.shape[:2]
            for face_landmarks in mesh_results.face_landmarks:
                # for landmark in face_landmarks:
                #     x = int(landmark.x * w)
                #     y = int(landmark.y * h)
                #     cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
                # Ambil titik mata (contoh mata kiri)
                p_top_left = face_landmarks[159]
                p_bottom_left = face_landmarks[145]
                
                p_top_right = face_landmarks[386]
                p_bottom_right = face_landmarks[374]

                # Hitung jarak vertikal sederhana
                ear_dist_left = abs(p_top_left.y - p_bottom_left.y)
                ear_dist_right = abs(p_top_right.y - p_bottom_right.y)

                # Visualisasi titik mata agar terlihat
                cv2.circle(annotated_frame, (int(p_top_left.x * w), int(p_top_left.y * h)), 2, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (int(p_bottom_left.x * w), int(p_bottom_left.y * h)), 2, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (int(p_top_right.x * w), int(p_top_right.y * h)), 2, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (int(p_bottom_right.x * w), int(p_bottom_right.y * h)), 2, (0, 0, 255), -1)

                # Ambang batas (threshold) biasanya sekitar 0.02 - 0.03 untuk Tasks API
                if ear_dist_left < 0.015 or ear_dist_right < 0.015:
                    cv2.putText(annotated_frame, "MENGANTUK!", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Di sini Anda bisa menambahkan logika EAR (Eye Aspect Ratio)
                # menggunakan koordinat landmarks mata untuk deteksi kantuk yang lebih akurat.
                
        # Tampilkan Hasil
        cv2.imshow('YOLOv8 + MediaPipe Face Mesh', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()