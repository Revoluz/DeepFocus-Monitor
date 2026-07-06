"""
core/state_manager.py - Manajemen Status dan Kalibrasi

File ini berisi tiga kelas utama untuk mengelola status sistem:

1. DynamicCalibration: Kalibrasi otomatis threshold per pengguna
   - Mengumpulkan 150 sampel (5 detik @ 30fps) saat startup
   - Menghitung baseline EAR dan Lip Distance
   - Menentukan threshold dinamis menggunakan mean ± 2*std

2. MicrosleepClassifier: State machine untuk klasifikasi kantuk
   - NORMAL: Kondisi fokus ideal
   - YAWNING: Terdeteksi menguap (1x)
   - DROWSY: Mengantuk (3x menguap dalam 10 detik)
   - MICROSLEEP: Mata terpejam > 1 detik (30 frame)

3. StateManager: Koordinator utama yang menghubungkan kalibrasi,
   classifier, dan deteksi distraksi (HP)
   - Mengelola transisi antar status
   - Menghitung phone counter untuk trigger DISTRACTED/PHONE_ALERT
   - Menyediakan break reminder logic

Alur Kerja:
1. Startup → DynamicCalibration.collect_samples(150 frame)
2. Kalibrasi selesai → MicrosleepClassifier.init(thresholds)
3. Setiap frame → StateManager.update() → return status
4. Jika phone_detected > 5s → DISTRACTED
5. Jika phone_detected > 10s → PHONE_ALERT + alarm
"""

import time
import numpy as np

import config


class DynamicCalibration:
    """
    Kalibrasi otomatis untuk menentukan threshold unik per pengguna.

    Cara kerja:
    1. Kumpulkan 150 sampel EAR dan Lip Distance (5 detik @ 30fps)
    2. Hitung mean dan standar deviasi dari sampel
    3. Tentukan threshold:
       - ear_threshold = mean - (2 * std)  → mata tertutup
       - lip_threshold = mean + (2 * std)  → mulut terbuka (menguap)

    Mengapa 2 * std?
    - Mencakup ~95% data normal (confidence interval 95%)
    - Balance antara sensitivitas dan akurasi
    - Mengurangi false positive dari kedipan/bicara normal
    """

    def __init__(self, duration=config.CALIBRATION_DURATION, fps=config.FPS):
        """
        Inisialisasi kalibrasi dengan durasi dan FPS.

        Args:
            duration: Durasi kalibrasi dalam detik (default: 5)
            fps: Frame per second (default: 30)
        """
        self.duration = duration
        self.fps = fps
        self.total_frames = duration * fps  # 150 frame untuk 5 detik
        self.ear_samples = []  # List untuk menyimpan sampel EAR
        self.lip_samples = []  # List untuk menyimpan sampel Lip Distance
        self.current_frame = 0  # Counter frame saat ini

        # Hasil kalibrasi (diisi setelah selesai)
        self.baseline_ear = None
        self.baseline_lip = None
        self.std_ear = None
        self.std_lip = None
        self.ear_threshold = None
        self.lip_threshold = None

    def add_sample(self, ear, lip_distance):
        """
        Menambahkan sampel EAR dan Lip Distance ke dalam koleksi.

        Dipanggil setiap frame selama proses kalibrasi.

        Args:
            ear: Nilai Eye Aspect Ratio dari frame saat ini
            lip_distance: Jarak bibir dari frame saat ini

        Returns:
            bool: True jika kalibrasi selesai, False jika belum
        """
        if self.current_frame >= self.total_frames:
            return True  # Sudah selesai

        self.ear_samples.append(ear)
        self.lip_samples.append(lip_distance)
        self.current_frame += 1

        # Jika sudah mencapai total frame, hitung threshold
        if self.current_frame >= self.total_frames:
            self._compute_thresholds()
            return True

        return False

    def _compute_thresholds(self):
        """
        Menghitung threshold dinamis dari sampel yang terkumpul.

        Rumus:
            ear_threshold = mean(ear_samples) - (2 * std(ear_samples))
            lip_threshold = mean(lip_samples) + (2 * std(lip_samples))

        Penjelasan:
            - EAR threshold: mean - 2*std → nilai di bawah ini = mata tertutup
            - Lip threshold: mean + 2*std → nilai di atas ini = mulut terbuka (menguap)
        """
        self.baseline_ear = np.mean(self.ear_samples)
        self.baseline_lip = np.mean(self.lip_samples)
        self.std_ear = np.std(self.ear_samples)
        self.std_lip = np.std(self.lip_samples)

        # Threshold: 2 standar deviasi dari mean
        self.ear_threshold = self.baseline_ear - (config.THRESHOLD_STD_MULTIPLIER * self.std_ear)
        self.lip_threshold = self.baseline_lip + (config.THRESHOLD_STD_MULTIPLIER * self.std_lip)

    def get_progress(self):
        """
        Menghitung progress kalibrasi.

        Returns:
            float: Progress dari 0.0 (belum mulai) hingga 1.0 (selesai)
        """
        return self.current_frame / self.total_frames

    def get_remaining_seconds(self):
        """
        Menghitung sisa waktu kalibrasi.

        Returns:
            float: Sisa waktu dalam detik
        """
        remaining_frames = self.total_frames - self.current_frame
        return remaining_frames / self.fps

    def is_complete(self):
        """
        Mengecek apakah kalibrasi sudah selesai.

        Returns:
            bool: True jika sudah selesai, False jika belum
        """
        return self.current_frame >= self.total_frames


class MicrosleepClassifier:
    """
    State machine untuk klasifikasi status kelelahan.

    State flow:
        NORMAL → YAWNING → DROWSY → MICROSLEEP
          ↑         ↑         ↑         ↑
          └─────────┴─────────┴─────────┘
              (Reset jika kondisi normal kembali)

    Logika transisi:
    1. NORMAL → YAWNING: Lip distance > lip_threshold (1x menguap)
    2. YAWNING → DROWSY: 3x menguap dalam window 10 detik
    3. DROWSY → MICROSLEEP: EAR < ear_threshold selama ≥ 30 frame (1 detik)
    4. Semua state → NORMAL: Jika kondisi kembali normal

    Anti-false positive:
    - Kedipan normal: < 9 frame (0.3 detik) → diabaikan
    - Menguap valid: ≥ 15 frame (0.5 detik) → dihitung
    - Microsleep: ≥ 30 frame (1 detik) → trigger alarm
    """

    # Konstanta status
    NORMAL = 'NORMAL'
    YAWNING = 'YAWNING'
    DROWSY = 'DROWSY'
    MICROSLEEP = 'MICROSLEEP'

    def __init__(self, ear_threshold, lip_threshold, fps=config.FPS):
        """
        Inisialisasi classifier dengan threshold dari kalibrasi.

        Args:
            ear_threshold: Batas EAR untuk deteksi mata tertutup
            lip_threshold: Batas lip distance untuk deteksi menguap
            fps: Frame per second untuk perhitungan frame counter
        """
        self.ear_threshold = ear_threshold
        self.lip_threshold = lip_threshold
        self.fps = fps

        # Counters untuk tracking frame berturut-turut
        self.ear_low_counter = 0  # Frame berturut-turut EAR < threshold
        self.lip_high_counter = 0  # Frame berturut-turut lip > threshold
        self.yawn_timestamps = []  # List timestamp setiap menguap terdeteksi

        # State saat ini
        self.current_state = self.NORMAL

    def update(self, ear, lip_distance, face_detected=True):
        """
        Update state machine dengan data frame terbaru.

        Args:
            ear: Nilai Eye Aspect Ratio saat ini
            lip_distance: Jarak bibir saat ini
            face_detected: Apakah wajah terdeteksi

        Returns:
            str: Status saat ini (NORMAL, YAWNING, DROWSY, MICROSLEEP, FACE_LOST)
        """
        # Jika wajah tidak terdeteksi, reset dan return FACE_LOST
        if not face_detected:
            self.reset()
            return 'FACE_LOST'

        # Update EAR counter (deteksi mata tertutup)
        if ear < self.ear_threshold:
            self.ear_low_counter += 1
        else:
            self.ear_low_counter = 0  # Reset jika mata terbuka

        # Update Lip counter (deteksi menguap)
        if lip_distance > self.lip_threshold:
            if self.lip_high_counter == 0:
                # Pertama kali terdeteksi menguap, catat timestamp
                self.yawn_timestamps.append(time.time())
            self.lip_high_counter += 1
        else:
            self.lip_high_counter = 0  # Reset jika mulut tertutup

        # Hitung jumlah menguap dalam window 10 detik
        recent_yawns = self._count_recent_yawns()

        # Klasifikasi state berdasarkan prioritas (dari yang paling kritis)
        if self.ear_low_counter >= config.MICROSLEEP_MIN_FRAMES:
            self.current_state = self.MICROSLEEP  # Mata terpejam ≥ 1 detik
        elif recent_yawns >= config.YAWN_SEQUENCE_THRESHOLD:
            self.current_state = self.DROWSY  # 3x menguap dalam 10 detik
        elif lip_distance > self.lip_threshold:
            self.current_state = self.YAWNING  # Sedang menguap
        else:
            self.current_state = self.NORMAL  # Kondisi normal

        return self.current_state

    def _count_recent_yawns(self):
        """
        Menghitung jumlah menguap dalam window waktu terakhir.

        Returns:
            int: Jumlah menguap dalam window YAWN_WINDOW_SECONDS
        """
        now = time.time()
        window = config.YAWN_WINDOW_SECONDS
        # Filter timestamp yang masih dalam window
        recent = [t for t in self.yawn_timestamps if (now - t) <= window]
        self.yawn_timestamps = recent  # Cleanup timestamp lama
        return len(recent)

    def reset(self):
        """
        Reset semua counters dan state ke kondisi awal.
        Dipanggil saat wajah hilang atau kalibrasi ulang.
        """
        self.ear_low_counter = 0
        self.lip_high_counter = 0
        self.yawn_timestamps = []
        self.current_state = self.NORMAL


class StateManager:
    """
    Koordinator utama untuk manajemen status sistem.

    Menghubungkan:
    - DynamicCalibration: Untuk kalibrasi threshold
    - MicrosleepClassifier: Untuk klasifikasi status kantuk
    - Phone counter: Untuk deteksi distraksi HP

    Status yang dikelola:
    - NORMAL, YAWNING, DROWSY, MICROSLEEP (dari classifier)
    - DISTRACTED: HP terdeteksi > 5 detik
    - PHONE_ALERT: HP terdeteksi > 10 detik + alarm
    - FACE_LOST: Wajah tidak terdeteksi
    """

    def __init__(self, fps=config.FPS):
        """
        Inisialisasi StateManager dengan calibration dan counters.

        Args:
            fps: Frame per second untuk perhitungan frame counter
        """
        self.calibration = DynamicCalibration(fps=fps)
        self.classifier = None  # Akan diinisialisasi setelah kalibrasi selesai
        self.phone_counter = 0  # Counter frame HP terdeteksi
        self.fps = fps
        self.break_reminder_triggered = False  # Flag untuk break reminder
        self.last_break_time = time.time()  # Waktu terakhir break reminder

    def calibrate(self, ear, lip_distance):
        """
        Menjalankan proses kalibrasi dengan menambahkan sampel.

        Args:
            ear: Nilai EAR dari frame saat ini
            lip_distance: Jarak bibir dari frame saat ini

        Returns:
            tuple: (done, progress, remaining_seconds)
                - done: bool, apakah kalibrasi selesai
                - progress: float, progress 0.0-1.0
                - remaining_seconds: float, sisa waktu kalibrasi
        """
        done = self.calibration.add_sample(ear, lip_distance)

        # Jika belum selesai, return progress
        if done and not self.calibration.is_complete():
            return False, self.calibration.get_progress(), self.calibration.get_remaining_seconds()

        # Jika selesai dan classifier belum ada, buat classifier baru
        if done and self.classifier is None:
            self.classifier = MicrosleepClassifier(
                ear_threshold=self.calibration.ear_threshold,
                lip_threshold=self.calibration.lip_threshold,
                fps=self.fps
            )
            return True, 1.0, 0

        # Jika sudah selesai sebelumnya
        if self.calibration.is_complete():
            return True, 1.0, 0

        return False, self.calibration.get_progress(), self.calibration.get_remaining_seconds()

    def reset_calibration(self):
        """
        Reset kalibrasi untuk kalibrasi ulang.
        Dipanggil saat user menekan tombol "Kalibrasi Ulang".
        """
        self.calibration = DynamicCalibration(fps=self.fps)
        self.classifier = None
        self.phone_counter = 0
        self.break_reminder_triggered = False

    def update(self, ear, lip_distance, head_pose, phone_detected, face_detected=True):
        """
        Update status utama sistem dengan data frame terbaru.

        Args:
            ear: Nilai EAR saat ini
            lip_distance: Jarak bibir saat ini
            head_pose: Dictionary {'yaw', 'pitch', 'roll'}
            phone_detected: bool, apakah HP terdeteksi
            face_detected: bool, apakah wajah terdeteksi

        Returns:
            str: Status saat ini
        """
        # Jika wajah tidak terdeteksi
        if not face_detected:
            self.phone_counter = 0
            return 'FACE_LOST'

        # Update phone counter
        if phone_detected:
            self.phone_counter += 1
        else:
            self.phone_counter = 0

        # Update classifier untuk status kantuk
        state = self.classifier.update(ear, lip_distance, face_detected)

        # Cek status HP
        if phone_detected:
            if self.phone_counter >= config.PHONE_LIMIT_FRAMES:
                return 'PHONE_ALERT'  # HP > 10 detik → alarm
            elif self.phone_counter >= config.PHONE_DISTRACTED_FRAMES:
                return 'DISTRACTED'  # HP > 5 detik → warning

        return state

    def check_break_reminder(self, session_tracker):
        """
        Mengecek apakah perlu menampilkan break reminder.

        Trigger:
        - 3x microsleep dalam sesi ATAU
        - 5x yawning dalam sesi

        Reset flag setelah 30 menit untuk menghindari spam reminder.

        Args:
            session_tracker: SessionTracker instance untuk cek counter

        Returns:
            str/None: Pesan break reminder atau None jika tidak perlu
        """
        # Reset flag jika sudah > 30 menit sejak trigger terakhir
        if time.time() - self.last_break_time > 1800:  # 30 menit
            self.break_reminder_triggered = False

        # Cek apakah perlu trigger reminder
        if not self.break_reminder_triggered:
            message = session_tracker.check_break_reminder()
            if message:
                self.break_reminder_triggered = True
                self.last_break_time = time.time()
                return message
        return None
