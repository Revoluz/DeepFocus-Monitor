import time
import numpy as np

import config


class DynamicCalibration:
    def __init__(self, duration=config.CALIBRATION_DURATION, fps=config.FPS):
        self.duration = duration
        self.fps = fps
        self.total_frames = duration * fps
        self.ear_samples = []
        self.lip_samples = []
        self.current_frame = 0

        self.baseline_ear = None
        self.baseline_lip = None
        self.std_ear = None
        self.std_lip = None
        self.ear_threshold = None
        self.lip_threshold = None

    def add_sample(self, ear, lip_distance):
        if self.current_frame >= self.total_frames:
            return True

        self.ear_samples.append(ear)
        self.lip_samples.append(lip_distance)
        self.current_frame += 1

        if self.current_frame >= self.total_frames:
            self._compute_thresholds()
            return True

        return False

    def _compute_thresholds(self):
        self.baseline_ear = np.mean(self.ear_samples)
        self.baseline_lip = np.mean(self.lip_samples)
        self.std_ear = np.std(self.ear_samples)
        self.std_lip = np.std(self.lip_samples)

        self.ear_threshold = self.baseline_ear - (config.THRESHOLD_STD_MULTIPLIER * self.std_ear)
        self.lip_threshold = self.baseline_lip + (config.THRESHOLD_STD_MULTIPLIER * self.std_lip)

    def get_progress(self):
        return self.current_frame / self.total_frames

    def get_remaining_seconds(self):
        remaining_frames = self.total_frames - self.current_frame
        return remaining_frames / self.fps

    def is_complete(self):
        return self.current_frame >= self.total_frames


class MicrosleepClassifier:
    NORMAL = 'NORMAL'
    YAWNING = 'YAWNING'
    DROWSY = 'DROWSY'
    MICROSLEEP = 'MICROSLEEP'

    def __init__(self, ear_threshold, lip_threshold, fps=config.FPS):
        self.ear_threshold = ear_threshold
        self.lip_threshold = lip_threshold
        self.fps = fps

        self.ear_low_counter = 0
        self.lip_high_counter = 0
        self.yawn_timestamps = []

        self.current_state = self.NORMAL

    def update(self, ear, lip_distance, face_detected=True):
        if not face_detected:
            self.reset()
            return 'FACE_LOST'

        if ear < self.ear_threshold:
            self.ear_low_counter += 1
        else:
            self.ear_low_counter = 0

        if lip_distance > self.lip_threshold:
            if self.lip_high_counter == 0:
                self.yawn_timestamps.append(time.time())
            self.lip_high_counter += 1
        else:
            self.lip_high_counter = 0

        recent_yawns = self._count_recent_yawns()

        if self.ear_low_counter >= config.MICROSLEEP_MIN_FRAMES:
            self.current_state = self.MICROSLEEP
        elif recent_yawns >= config.YAWN_SEQUENCE_THRESHOLD:
            self.current_state = self.DROWSY
        elif lip_distance > self.lip_threshold:
            self.current_state = self.YAWNING
        else:
            self.current_state = self.NORMAL

        return self.current_state

    def _count_recent_yawns(self):
        now = time.time()
        window = config.YAWN_WINDOW_SECONDS
        recent = [t for t in self.yawn_timestamps if (now - t) <= window]
        self.yawn_timestamps = recent
        return len(recent)

    def reset(self):
        self.ear_low_counter = 0
        self.lip_high_counter = 0
        self.yawn_timestamps = []
        self.current_state = self.NORMAL


class StateManager:
    def __init__(self, fps=config.FPS):
        self.calibration = DynamicCalibration(fps=fps)
        self.classifier = None
        self.phone_counter = 0
        self.fps = fps
        self.break_reminder_triggered = False
        self.last_break_time = time.time()

    def calibrate(self, ear, lip_distance):
        done = self.calibration.add_sample(ear, lip_distance)

        if done and not self.calibration.is_complete():
            return False, self.calibration.get_progress(), self.calibration.get_remaining_seconds()

        if done and self.classifier is None:
            self.classifier = MicrosleepClassifier(
                ear_threshold=self.calibration.ear_threshold,
                lip_threshold=self.calibration.lip_threshold,
                fps=self.fps
            )
            return True, 1.0, 0

        if self.calibration.is_complete():
            return True, 1.0, 0

        return False, self.calibration.get_progress(), self.calibration.get_remaining_seconds()

    def reset_calibration(self):
        self.calibration = DynamicCalibration(fps=self.fps)
        self.classifier = None
        self.phone_counter = 0
        self.break_reminder_triggered = False

    def update(self, ear, lip_distance, head_pose, phone_detected, face_detected=True):
        if not face_detected:
            self.phone_counter = 0
            return 'FACE_LOST'

        if phone_detected:
            self.phone_counter += 1
        else:
            self.phone_counter = 0

        yaw = head_pose.get('yaw', 0.0) if head_pose else 0.0
        state = self.classifier.update(ear, lip_distance, face_detected)

        if phone_detected:
            if self.phone_counter >= config.PHONE_LIMIT_FRAMES:
                return 'PHONE_ALERT'
            elif self.phone_counter >= config.PHONE_DISTRACTED_FRAMES:
                return 'DISTRACTED'

        return state

    def check_break_reminder(self, session_tracker):
        if time.time() - self.last_break_time > 1800:
            self.break_reminder_triggered = False

        if not self.break_reminder_triggered:
            message = session_tracker.check_break_reminder()
            if message:
                self.break_reminder_triggered = True
                self.last_break_time = time.time()
                return message
        return None
