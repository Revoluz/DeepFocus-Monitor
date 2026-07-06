import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.helpers import euclidean_distance


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14

HEAD_POSE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.double)

HEAD_POSE_2D_INDICES = [1, 9, 33, 263, 61, 291]


def ensure_face_landmarker_model(model_path: str) -> None:
    if os.path.exists(model_path):
        return
    print("Mengunduh model Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Model tersimpan di: {model_path}")


class FaceAnalyzer:
    def __init__(self, fps=30):
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
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.fps = fps
        self.camera_matrix = None
        self.dist_coeffs = None
        self._init_camera_matrix()

    def _init_camera_matrix(self):
        frame_width, frame_height = 640, 480
        focal_length = frame_width
        self.camera_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1],
        ], dtype=np.double)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.double)

    def calculate_ear(self, landmarks, side='left'):
        if side == 'left':
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in LEFT_EYE_INDICES]
        else:
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in RIGHT_EYE_INDICES]

        vertical_1 = euclidean_distance(p2, p6)
        vertical_2 = euclidean_distance(p3, p5)
        horizontal = euclidean_distance(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def calculate_ear_average(self, landmarks):
        ear_left = self.calculate_ear(landmarks, 'left')
        ear_right = self.calculate_ear(landmarks, 'right')
        return (ear_left + ear_right) / 2.0

    def calculate_lip_distance(self, landmarks):
        upper_lip = landmarks[UPPER_LIP_INDEX]
        lower_lip = landmarks[LOWER_LIP_INDEX]
        return euclidean_distance(upper_lip, lower_lip)

    def estimate_head_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in HEAD_POSE_2D_INDICES
        ], dtype=np.double)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            HEAD_POSE_3D_POINTS,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

        yaw = float(euler_angles[1][0])
        pitch = float(euler_angles[0][0])
        roll = float(euler_angles[2][0])

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    def analyze(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.monotonic() * 1000)

        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return {
                'face_detected': False,
                'ear': None,
                'lip_distance': None,
                'head_pose': None,
                'landmarks': None,
            }

        landmarks = results.face_landmarks[0]
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
