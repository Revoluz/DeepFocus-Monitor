"""
config.py - Konfigurasi Global DeepFocus Monitor

File ini menyimpan semua parameter konfigurasi yang dapat disesuaikan
untuk mengatur perilaku sistem deteksi kantuk dan distraksi.

Parameter utama:
- Kamera: indeks webcam dan resolusi
- Kalibrasi: durasi dan metode threshold dinamis
- Deteksi kantuk: frame threshold untuk microsleep, yawning, blink
- Deteksi objek: kelas YOLO dan confidence threshold
- Status: mapping status ke warna BGR untuk overlay visual
"""

# =============================================================================
# KONFIGURASI KAMERA
# =============================================================================
CAMERA_INDEX = 0  # Indeks webcam (0 = default, ubah jika kamera tidak terdeteksi)
FPS = 30  # Frame per second (asumsi webcam standar)

# =============================================================================
# KONFIGURASI KALIBRASI
# =============================================================================
CALIBRATION_DURATION = 5  # Durasi kalibrasi otomatis saat startup (detik)
THRESHOLD_STD_MULTIPLIER = 2  # Pengali standar deviasi untuk threshold dinamis
# Penjelasan: threshold = mean ± (multiplier * std)
# Nilai 2 mencakup ~95% data normal (confidence interval 95%)

# =============================================================================
# KONFIGURASI DETEKSI KANTUK (TIME-SERIES)
# =============================================================================
YAWN_SEQUENCE_THRESHOLD = 3  # Jumlah menguap beruntun untuk trigger status DROWSY
YAWN_WINDOW_SECONDS = 10  # Window waktu untuk menghitung yawning sequence (detik)
MICROSLEEP_MIN_FRAMES = 30  # Frame minimum EAR rendah = microsleep (30 frame ≈ 1 detik @30fps)
BLINK_MAX_FRAMES = 9  # Frame maksimum kedipan normal (9 frame ≈ 0.3 detik)
YAWNING_MIN_FRAMES = 15  # Frame minimum untuk deteksi menguap valid (15 frame ≈ 0.5 detik)

# =============================================================================
# KONFIGURASI DETEKSI OBJEK (YOLOv8)
# =============================================================================
YOLO_CLASSES = [0, 67, 73]  # Kelas COCO: 0=person, 67=cell phone, 73=book
YOLO_CONFIDENCE = 0.5  # Minimum confidence threshold untuk deteksi YOLO

# Threshold durasi HP terdeteksi (dalam frame)
PHONE_DISTRACTED_FRAMES = 1  # langsung tampil status DISTRACTED saat HP terdeteksi
PHONE_LIMIT_FRAMES = 60  # setelah HP terdeteksi cukup lama → status PHONE_ALERT + alarm

# =============================================================================
# KONFIGURASI HEAD POSE
# =============================================================================
HEAD_POSE_YAW_THRESHOLD = 20  # Batas sudut yaw (derajat) untuk deteksi menoleh

# =============================================================================
# KONFIGURASI STATUS & WARNA OVERLAY
# =============================================================================
# Format warna: BGR (Blue, Green, Red) untuk OpenCV
STATUS_COLORS = {
    'NORMAL': (0, 255, 0),
    'YAWNING': (0, 165, 255),
    'DROWSY': (0, 165, 255),
    'DISTRACTED': (0, 140, 255),
    'MICROSLEEP': (0, 0, 255),
    'PHONE_ALERT': (0, 0, 255),
    'FACE_LOST': (255, 255, 255),
}
