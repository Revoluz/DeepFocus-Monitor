"""
core/session_tracker.py - Tracking Statistik Sesi Belajar

File ini berisi kelas SessionTracker yang bertanggung jawab untuk:
1. Menghitung durasi sesi belajar
2. Menghitung focus score (persentase waktu fokus)
3. Menghitung distribusi status (NORMAL, YAWNING, DROWSY, dll)
4. Menghitung total microsleep dan yawning
5. Menyediakan data untuk break reminder
6. Export riwayat data ke file CSV

Data yang dilacak:
- status_counts: Jumlah frame per status
- microsleep_total: Total kejadian microsleep
- yawning_total: Total kejadian yawning
- ear_history: Riwayat nilai EAR (untuk chart)
- lip_history: Riwayat nilai Lip Distance (untuk chart)
"""

import time
import csv


class SessionTracker:
    """
    Kelas untuk tracking statistik sesi belajar.

    Fungsi utama:
    - update: Menambahkan data frame terbaru ke statistik
    - get_focus_score: Menghitung persentase waktu fokus
    - get_status_distribution: Menghitung distribusi semua status
    - check_break_reminder: Mengecek apakah perlu break reminder
    - get_session_duration: Menghitung durasi sesi
    - export_csv: Export riwayat data ke file CSV
    - reset: Reset semua data untuk sesi baru
    """

    def __init__(self):
        """Inisialisasi SessionTracker dengan data awal kosong."""
        self.start_time = time.time()  # Waktu mulai sesi
        self.status_counts = {
            'NORMAL': 0,
            'YAWNING': 0,
            'DROWSY': 0,
            'DISTRACTED': 0,
            'MICROSLEEP': 0,
            'PHONE_ALERT': 0,
            'FACE_LOST': 0,
        }
        self.microsleep_total = 0  # Total kejadian microsleep
        self.yawning_total = 0  # Total kejadian yawning
        self.ear_history = []  # Riwayat EAR untuk chart
        self.lip_history = []  # Riwayat Lip Distance untuk chart
        self.break_reminder_shown = False  # Flag break reminder

    def update(self, status, ear, lip_distance):
        """
        Menambahkan data frame terbaru ke statistik.

        Args:
            status: Status saat ini (NORMAL, YAWNING, dll)
            ear: Nilai EAR saat ini (bisa None jika tidak terdeteksi)
            lip_distance: Jarak bibir saat ini (bisa None)
        """
        # Increment counter untuk status saat ini
        self.status_counts[status] += 1

        # Simpan riwayat untuk chart (jika data tersedia)
        if ear is not None:
            self.ear_history.append(ear)
        if lip_distance is not None:
            self.lip_history.append(lip_distance)

        # Increment total kejadian untuk break reminder
        if status == 'MICROSLEEP':
            self.microsleep_total += 1
        elif status in ['YAWNING', 'DROWSY']:
            self.yawning_total += 1

    def get_focus_score(self):
        """
        Menghitung persentase waktu fokus.

        Rumus:
            focus_score = (count['NORMAL'] / total_frames) * 100

        Returns:
            float: Persentase waktu fokus (0-100)
        """
        total = sum(self.status_counts.values())
        if total == 0:
            return 100.0
        return (self.status_counts['NORMAL'] / total) * 100.0

    def get_status_distribution(self):
        """
        Menghitung distribusi persentase semua status.

        Returns:
            dict: {status: percentage} untuk semua status
        """
        total = sum(self.status_counts.values())
        if total == 0:
            return {k: 0.0 for k in self.status_counts}
        return {k: (v / total) * 100.0 for k, v in self.status_counts.items()}

    def check_break_reminder(self, microsleep_threshold=3, yawning_threshold=5):
        """
        Mengecek apakah perlu menampilkan break reminder.

        Trigger:
        - microsleep_total >= microsleep_threshold (default: 3) ATAU
        - yawning_total >= yawning_threshold (default: 5)

        Args:
            microsleep_threshold: Batas microsleep untuk trigger reminder
            yawning_threshold: Batas yawning untuk trigger reminder

        Returns:
            str/None: Pesan reminder atau None jika tidak perlu
        """
        if self.microsleep_total >= microsleep_threshold:
            return f"Anda sudah {self.microsleep_total}x microsleep. Disarankan istirahat 5 menit!"
        elif self.yawning_total >= yawning_threshold:
            return f"Anda sudah {self.yawning_total}x menguap. Mungkin perlu istirahat?"
        return None

    def get_session_duration(self):
        """
        Menghitung durasi sesi dalam format HH:MM:SS.

        Returns:
            str: Durasi sesi (contoh: "01:23:45")
        """
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def reset(self):
        """
        Reset semua data untuk sesi baru.
        Dipanggil saat user menekan tombol "Kalibrasi Ulang".
        """
        self.start_time = time.time()
        self.status_counts = {k: 0 for k in self.status_counts}
        self.microsleep_total = 0
        self.yawning_total = 0
        self.ear_history = []
        self.lip_history = []
        self.break_reminder_shown = False

    def export_csv(self, filename='session_log.csv'):
        """
        Export riwayat data EAR dan Lip Distance ke file CSV.

        Args:
            filename: Nama file output (default: 'session_log.csv')

        Returns:
            bool: True jika berhasil, False jika tidak ada data
        """
        if not self.ear_history:
            return False

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ear', 'lip_distance'])
            for i, (ear, lip) in enumerate(zip(self.ear_history, self.lip_history)):
                writer.writerow([i, ear, lip])
        return True
