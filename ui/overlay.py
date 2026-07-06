import cv2
import numpy as np

import config


class Overlay:
    BORDER_THICKNESS = 8
    TEXT_SCALE = 1.0
    TEXT_THICKNESS = 2
    STATS_SCALE = 0.6
    STATS_THICKNESS = 1

    @classmethod
    def draw(cls, frame, status, info):
        color = config.STATUS_COLORS.get(status, (255, 255, 255))

        cls._draw_border(frame, color)
        cls._draw_status_text(frame, status, color)
        cls._draw_stats(frame, info)
        cls._draw_landmarks(frame, info)

        return frame

    @classmethod
    def _draw_border(cls, frame, color):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), color, cls.BORDER_THICKNESS)

    @classmethod
    def _draw_status_text(cls, frame, status, color):
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
        y_offset = 80
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = cls.STATS_SCALE
        thickness = cls.STATS_THICKNESS
        color = (255, 255, 255)

        stats = []

        if info.get('ear') is not None:
            stats.append(f"EAR: {info['ear']:.4f}")

        if info.get('lip_distance') is not None:
            stats.append(f"LIP: {info['lip_distance']:.4f}")

        if info.get('head_pose'):
            yaw = info['head_pose'].get('yaw', 0)
            stats.append(f"YAW: {yaw:.1f}")

        if info.get('detections'):
            for det in info['detections']:
                cls_name = det.get('class', '')
                conf = det.get('conf', 0)
                stats.append(f"{cls_name}: {conf:.2f}")

        if info.get('state'):
            stats.append(f"STATE: {info['state']}")

        for i, stat in enumerate(stats):
            y = y_offset + (i * line_height)
            cv2.putText(frame, stat, (20, y), font, scale, color, thickness)

    @classmethod
    def _draw_landmarks(cls, frame, info):
        if not info.get('landmarks'):
            return

        h, w = frame.shape[:2]
        landmarks = info['landmarks']

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        lip_indices = [13, 14]

        for idx in left_eye_indices + right_eye_indices:
            pt = landmarks[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        for idx in lip_indices:
            pt = landmarks[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
