import time
import csv


class SessionTracker:
    def __init__(self):
        self.start_time = time.time()
        self.status_counts = {
            'NORMAL': 0,
            'YAWNING': 0,
            'DROWSY': 0,
            'DISTRACTED': 0,
            'MICROSLEEP': 0,
            'PHONE_ALERT': 0,
            'FACE_LOST': 0,
        }
        self.microsleep_total = 0
        self.yawning_total = 0
        self.ear_history = []
        self.lip_history = []
        self.break_reminder_shown = False

    def update(self, status, ear, lip_distance):
        self.status_counts[status] += 1

        if ear is not None:
            self.ear_history.append(ear)
        if lip_distance is not None:
            self.lip_history.append(lip_distance)

        if status == 'MICROSLEEP':
            self.microsleep_total += 1
        elif status in ['YAWNING', 'DROWSY']:
            self.yawning_total += 1

    def get_focus_score(self):
        total = sum(self.status_counts.values())
        if total == 0:
            return 100.0
        return (self.status_counts['NORMAL'] / total) * 100.0

    def get_status_distribution(self):
        total = sum(self.status_counts.values())
        if total == 0:
            return {k: 0.0 for k in self.status_counts}
        return {k: (v / total) * 100.0 for k, v in self.status_counts.items()}

    def check_break_reminder(self, microsleep_threshold=3, yawning_threshold=5):
        if self.microsleep_total >= microsleep_threshold:
            return f"Anda sudah {self.microsleep_total}x microsleep. Disarankan istirahat 5 menit!"
        elif self.yawning_total >= yawning_threshold:
            return f"Anda sudah {self.yawning_total}x menguap. Mungkin perlu istirahat?"
        return None

    def get_session_duration(self):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def reset(self):
        self.start_time = time.time()
        self.status_counts = {k: 0 for k in self.status_counts}
        self.microsleep_total = 0
        self.yawning_total = 0
        self.ear_history = []
        self.lip_history = []
        self.break_reminder_shown = False

    def export_csv(self, filename='session_log.csv'):
        if not self.ear_history:
            return False

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ear', 'lip_distance'])
            for i, (ear, lip) in enumerate(zip(self.ear_history, self.lip_history)):
                writer.writerow([i, ear, lip])
        return True
