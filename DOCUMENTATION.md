# Dokumentasi Kode — DeepFocus Monitor

Dokumentasi lengkap untuk setiap file dan function dalam project ini.

---

## 📁 Struktur File

```
drowsiness-detection/
├── main.py                          # Entry point aplikasi
├── config.py                        # Konfigurasi global
├── core/
│   ├── __init__.py
│   ├── face_analyzer.py             # Analisis wajah (EAR, Lip, Head Pose)
│   ├── state_manager.py             # Manajemen status & kalibrasi
│   ├── session_tracker.py           # Tracking statistik sesi
│   └── alarm.py                     # Sistem alarm suara (TTS)
├── ui/
│   ├── __init__.py
│   ├── overlay.py                   # Overlay visual pada frame OpenCV
│   └── dashboard.py                 # Dashboard GUI (tkinter)
└── utils/
    ├── __init__.py
    └── helpers.py                   # Fungsi utilitas matematika
```

---

## 📄 Penjelasan Per File

---

### 1. `config.py`

**Tujuan:** Menyimpan semua parameter konfigurasi global yang bisa disesuaikan.

**Konstanta:**

| Konstanta | Nilai | Deskripsi |
|-----------|-------|-----------|
| `CAMERA_INDEX` | `2` | Indeks webcam yang digunakan |
| `FPS` | `30` | Frame per second (asumsi) |
| `CALIBRATION_DURATION` | `5` | Durasi kalibrasi (detik) |
| `THRESHOLD_STD_MULTIPLIER` | `2` | Pengali standar deviasi untuk threshold |
| `YAWN_SEQUENCE_THRESHOLD` | `3` | Jumlah menguap untuk trigger DROWSY |
| `YAWN_WINDOW_SECONDS` | `10` | Window waktu hitung yawning (detik) |
| `MICROSLEEP_MIN_FRAMES` | `30` | Frame minimum EAR rendah = microsleep (1s) |
| `BLINK_MAX_FRAMES` | `9` | Frame maksimum kedipan normal (0.3s) |
| `YAWNING_MIN_FRAMES` | `15` | Frame minimum untuk deteksi menguap (0.5s) |
| `YOLO_CLASSES` | `[0, 67, 73]` | Kelas YOLO: person, cell phone, book |
| `YOLO_CONFIDENCE` | `0.5` | Minimum confidence threshold YOLO |
| `PHONE_DISTRACTED_FRAMES` | `150` | Frame HP terdeteksi → status DISTRACTED (5s) |
| `PHONE_LIMIT_FRAMES` | `300` | Frame HP terdeteksi → status PHONE_ALERT (10s) |
| `HEAD_POSE_YAW_THRESHOLD` | `20` | Batas sudut yaw (derajat) |
| `STATUS_COLORS` | `dict` | Mapping status → warna BGR |

---

### 2. `utils/helpers.py`

**Tujuan:** Fungsi utilitas matematika yang digunakan di berbagai modul.

**Functions:**

#### `euclidean_distance(p1, p2)`
- **Input:** Dua titik dengan atribut `.x` dan `.y` (MediaPipe landmarks)
- **Output:** Jarak Euclidean (float)
- **Rumus:** `√((x1-x2)² + (y1-y2)²)`
- **Digunakan di:** `face_analyzer.py` untuk menghitung EAR dan Lip Distance

#### `clamp(value, min_val, max_val)`
- **Input:** Nilai, batas minimum, batas maksimum
- **Output:** Nilai yang dibatasi dalam range [min_val, max_val]
- **Digunakan untuk:** Mencegah nilai keluar dari batas yang ditentukan

---

### 3. `core/face_analyzer.py`

**Tujuan:** Analisis biometrik wajah menggunakan MediaPipe Face Landmarker.

**Kelas:** `FaceAnalyzer`

#### `__init__(self, fps=30)`
- **Fungsi:** Inisialisasi Face Landmarker model dan kamera matriks
- **Proses:**
  1. Download model jika belum ada (`ensure_face_landmarker_model`)
  2. Setup MediaPipe Face Landmarker dengan parameter confidence
  3. Inisialisasi camera matrix untuk head pose estimation

#### `_init_camera_matrix(self)`
- **Fungsi:** Setup matriks kamera untuk solvePnP
- **Output:** `self.camera_matrix` (3x3) dan `self.dist_coeffs` (4x1)
- **Digunakan di:** `estimate_head_pose()`

#### `calculate_ear(self, landmarks, side='left')`
- **Fungsi:** Hitung Eye Aspect Ratio (EAR) untuk satu mata
- **Input:** 
  - `landmarks`: 468 titik wajah dari MediaPipe
  - `side`: 'left' atau 'right'
- **Output:** Nilai EAR (float)
- **Rumus:** `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
- **Landmarks:**
  - Mata kiri: `[33, 160, 158, 133, 153, 144]`
  - Mata kanan: `[362, 385, 387, 263, 373, 380]`

#### `calculate_ear_average(self, landmarks)`
- **Fungsi:** Hitung rata-rata EAR dari kedua mata
- **Output:** `(EAR_kiri + EAR_kanan) / 2`

#### `calculate_lip_distance(self, landmarks)`
- **Fungsi:** Hitung jarak absolut antara bibir atas dan bawah
- **Input:** `landmarks` (468 titik)
- **Output:** Jarak Euclidean antara landmark 13 (bibir atas) dan 14 (bibir bawah)
- **Digunakan untuk:** Deteksi menguap (yawning)

#### `estimate_head_pose(self, landmarks, frame_shape)`
- **Fungsi:** Estimasi orientasi kepala (yaw, pitch, roll) menggunakan solvePnP
- **Input:**
  - `landmarks`: 468 titik wajah
  - `frame_shape`: Dimensi frame (height, width, channels)
- **Output:** Dictionary `{'yaw': float, 'pitch': float, 'roll': float}`
- **Proses:**
  1. Ambil 6 titik wajah (nose, chin, eyes, mouth corners)
  2. Gunakan `cv2.solvePnP()` untuk dapat rotation vector
  3. Convert ke Euler angles menggunakan `cv2.Rodrigues()` dan `cv2.decomposeProjectionMatrix()`

#### `analyze(self, frame)`
- **Fungsi:** Main pipeline — analisis satu frame lengkap
- **Input:** Frame BGR dari kamera
- **Output:** Dictionary dengan keys:
  - `face_detected`: bool
  - `ear`: float atau None
  - `lip_distance`: float atau None
  - `head_pose`: dict atau None
  - `landmarks`: list atau None

---

### 4. `core/state_manager.py`

**Tujuan:** Manajemen status sistem, kalibrasi dinamis, dan state machine klasifikasi.

**Kelas:**

#### `DynamicCalibration`

**`__init__(self, duration=5, fps=30)`**
- **Fungsi:** Inisialisasi kalibrasi dengan durasi dan FPS
- **Atribut:** `ear_samples[]`, `lip_samples[]`, `current_frame`

**`add_sample(self, ear, lip_distance)`**
- **Fungsi:** Tambahkan sampel EAR dan Lip Distance
- **Output:** `True` jika kalibrasi selesai, `False` jika belum
- **Proses:** Kumpulkan 150 sampel (5s × 30fps)

**`_compute_thresholds(self)`**
- **Fungsi:** Hitung threshold dinamis dari sampel
- **Rumus:**
  - `ear_threshold = mean(ear_samples) - (2 * std(ear_samples))`
  - `lip_threshold = mean(lip_samples) + (2 * std(lip_samples))`

**`get_progress(self)`**
- **Output:** Progress kalibrasi (0.0 - 1.0)

**`get_remaining_seconds(self)`**
- **Output:** Sisa waktu kalibrasi dalam detik

**`is_complete(self)`**
- **Output:** `True` jika kalibrasi selesai

---

#### `MicrosleepClassifier`

**`__init__(self, ear_threshold, lip_threshold, fps=30)`**
- **Fungsi:** Inisialisasi classifier dengan threshold dari kalibrasi
- **Atribut:** `ear_low_counter`, `lip_high_counter`, `yawn_timestamps[]`

**`update(self, ear, lip_distance, face_detected=True)`**
- **Fungsi:** Update state machine dengan data frame terbaru
- **Output:** Status string (`NORMAL`, `YAWNING`, `DROWSY`, `MICROSLEEP`, `FACE_LOST`)
- **Logika:**
  1. Jika `ear < ear_threshold` → increment `ear_low_counter`
  2. Jika `lip > lip_threshold` → increment `lip_high_counter` + catat timestamp
  3. Hitung yawning dalam window 10 detik
  4. Klasifikasi berdasarkan prioritas:
     - `ear_low_counter >= 30` → `MICROSLEEP`
     - `recent_yawns >= 3` → `DROWSY`
     - `lip > lip_threshold` → `YAWNING`
     - Lainnya → `NORMAL`

**`_count_recent_yawns(self)`**
- **Fungsi:** Hitung jumlah menguap dalam window 10 detik terakhir
- **Output:** Integer (jumlah yawning)

**`reset(self)`**
- **Fungsi:** Reset semua counters dan state

---

#### `StateManager`

**`__init__(self, fps=30)`**
- **Fungsi:** Inisialisasi calibration, classifier, dan phone counter

**`calibrate(self, ear, lip_distance)`**
- **Fungsi:** Jalankan proses kalibrasi
- **Output:** `(done, progress, remaining_seconds)`

**`reset_calibration(self)`**
- **Fungsi:** Reset kalibrasi untuk kalibrasi ulang
- **Proses:** Buat ulang DynamicCalibration dan classifier

**`update(self, ear, lip_distance, head_pose, phone_detected, face_detected=True)`**
- **Fungsi:** Update status utama sistem
- **Output:** Status string (termasuk `DISTRACTED`, `PHONE_ALERT`)
- **Logika:**
  1. Jika `phone_detected` dan `phone_counter >= 150` → `DISTRACTED`
  2. Jika `phone_counter >= 300` → `PHONE_ALERT`
  3. Jika tidak → delegate ke `classifier.update()`

**`check_break_reminder(self, session_tracker)`**
- **Fungsi:** Cek apakah perlu menampilkan break reminder
- **Trigger:** 3x microsleep ATAU 5x yawning dalam sesi
- **Output:** Pesan string atau `None`

---

### 5. `core/session_tracker.py`

**Tujuan:** Tracking statistik sesi belajar dan export data.

**Kelas:** `SessionTracker`

**`__init__(self)`**
- **Fungsi:** Inisialisasi tracker dengan start_time dan counters
- **Atribut:**
  - `status_counts`: Dictionary hitungan per status
  - `microsleep_total`, `yawning_total`: Total kejadian
  - `ear_history[]`, `lip_history[]`: Riwayat untuk chart

**`update(self, status, ear, lip_distance)`**
- **Fungsi:** Update statistik dengan data frame terbaru
- **Proses:** Increment counter, append history

**`get_focus_score(self)`**
- **Fungsi:** Hitung persentase waktu fokus
- **Rumus:** `(count['NORMAL'] / total) * 100`

**`get_status_distribution(self)`**
- **Fungsi:** Hitung distribusi persentase semua status
- **Output:** Dictionary `{status: percentage}`

**`check_break_reminder(self, microsleep_threshold=3, yawning_threshold=5)`**
- **Fungsi:** Cek apakah perlu break reminder
- **Output:** Pesan string atau `None`

**`get_session_duration(self)`**
- **Fungsi:** Hitung durasi sesi dalam format `HH:MM:SS`

**`reset(self)`**
- **Fungsi:** Reset semua data sesi

**`export_csv(self, filename='session_log.csv')`**
- **Fungsi:** Export riwayat EAR dan Lip ke file CSV
- **Output:** `True` jika berhasil, `False` jika tidak ada data

---

### 6. `core/alarm.py`

**Tujuan:** Sistem alarm suara menggunakan Text-to-Speech (pyttsx3).

**Kelas:** `Alarm`

**`__init__(self)`**
- **Fungsi:** Inisialisasi alarm dengan threading support
- **Atribut:** `_playing`, `_stop_event`, `_thread`, `_engine`

**`_init_engine(self)`**
- **Fungsi:** Inisialisasi pyttsx3 engine (lazy init)
- **Settings:** `rate=150`, `volume=1.0`

**`play(self, alarm_type='MICROSLEEP')`**
- **Fungsi:** Mulai alarm di thread terpisah
- **Input:** `alarm_type` → `'MICROSLEEP'` atau `'PHONE_ALERT'`
- **Proses:** Spawn thread yang menjalankan `_speak()`

**`_speak(self, alarm_type)`**
- **Fungsi:** Loop TTS message sampai di-stop
- **Messages:**
  - `MICROSLEEP`: "Perhatian! Anda terdeteksi microsleep. Segera istirahat!"
  - `PHONE_ALERT`: "Perhatian! Anda bermain HP terlalu lama. Fokus kembali belajar!"

**`stop(self)`**
- **Fungsi:** Stop alarm dan cleanup thread

---

### 7. `ui/overlay.py`

**Tujuan:** Visualisasi dinamis pada frame OpenCV.

**Kelas:** `Overlay` (class methods only)

**`draw(cls, frame, status, info)`**
- **Fungsi:** Main entry point — gambar semua overlay pada frame
- **Output:** Frame yang sudah dimodifikasi
- **Proses:** Panggil `_draw_border`, `_draw_status_text`, `_draw_stats`, `_draw_landmarks`

**`_draw_border(cls, frame, color)`**
- **Fungsi:** Gambar border berwarna di tepi frame
- **Warna:** Sesuai status (hijau/oranye/merah)
- **Thickness:** 8 pixel

**`_draw_status_text(cls, frame, status, color)`**
- **Fungsi:** Gambar teks status di pojok kiri atas
- **Mapping:**
  - `NORMAL` → "FOKUS"
  - `YAWNING` → "PERINGATAN: MENGUAP"
  - `DROWSY` → "PERINGATAN: MENGANTUK"
  - `DISTRACTED` → "TERDISTRASI: MEMEGANG HP"
  - `MICROSLEEP` → "BAHAYA: MICROSLEEP!"
  - `PHONE_ALERT` → "BAHAYA: HP TERLALU LAMA!"
  - `FACE_LOST` → "WAJAH TIDAK TERDETEKSI"

**`_draw_stats(cls, frame, info)`**
- **Fungsi:** Gambar statistik (EAR, LIP, YAW, detections)
- **Output:** Teks putih di sisi kiri frame

**`_draw_landmarks(cls, frame, info)`**
- **Fungsi:** Gambar titik-titik landmark mata (merah) dan bibir (biru)
- **Landmarks:** 12 titik mata + 2 titik bibir

---

### 8. `ui/dashboard.py`

**Tujuan:** Dashboard GUI menggunakan tkinter untuk monitoring real-time.

**Kelas:** `SimpleDashboard`

**`__init__(self, session_tracker, state_manager, alarm)`**
- **Fungsi:** Inisialisasi window tkinter dan semua widget
- **Widgets:**
  - `timer_label`: Session timer (HH:MM:SS)
  - `focus_label`: Focus score (%)
  - `status_label`: Status saat ini (berwarna)
  - `ear_label`, `lip_label`, `yaw_label`: Live metrics
  - `count_label`: Counter microsleep & yawning
  - `chart_canvas`: Canvas untuk EAR live chart
  - `break_label`: Break reminder message
  - Buttons: Pause, Kalibrasi Ulang, Export CSV

**`_build_ui(self)`**
- **Fungsi:** Setup layout dan semua widget tkinter

**`update(self, status, ear, lip_distance, head_pose)`**
- **Fungsi:** Update semua widget dengan data terbaru
- **Dipanggil dari:** Main loop setiap frame
- **Proses:**
  1. Update timer, focus score, status
  2. Update metrics (EAR, LIP, YAW)
  3. Redraw EAR chart
  4. Cek break reminder

**`_draw_ear_chart(self)`**
- **Fungsi:** Gambar grafik EAR live pada canvas
- **Fitur:**
  - Plot 100 data point terakhir
  - Garis threshold (merah, dashed)
  - Auto-scale berdasarkan min/max data

**`toggle_pause(self)`**
- **Fungsi:** Toggle pause/resume monitoring
- **Effect:** Freeze analysis, kamera tetap nyala

**`start_recalibration(self)`**
- **Fungsi:** Mulai kalibrasi ulang dengan konfirmasi dialog
- **Effect:** Reset calibration + session tracker, pause monitoring

**`export_csv(self)`**
- **Fungsi:** Export session log ke CSV
- **Output:** `session_log.csv`

**`run(self)`**
- **Fungsi:** Jalankan tkinter mainloop (harus di main thread)

---

### 9. `main.py`

**Tujuan:** Entry point aplikasi — integrasi semua modul.

**Kelas:** `CameraThread`

**`__init__(self, camera_index, width=640, height=480)`**
- **Fungsi:** Inisialisasi thread capture kamera
- **Atribut:** `cap` (VideoCapture), `frame`, `running`, `lock`

**`run(self)`**
- **Fungsi:** Loop capture frame di background thread
- **Proses:** Baca frame → copy ke `self.frame` → sleep 1/30s

**`get_frame(self)`**
- **Fungsi:** Ambil frame terbaru (thread-safe)
- **Output:** Frame numpy array atau None

**`stop(self)`**
- **Fungsi:** Stop thread dan release kamera

**Fungsi `main()`**

**Alur Eksekusi:**
1. **Inisialisasi:**
   - Start `CameraThread`
   - Buat `FaceAnalyzer`, `StateManager`, `SessionTracker`, `Alarm`
   - Buat `SimpleDashboard`

2. **Kalibrasi (5 detik):**
   - Tampilkan countdown + instruksi
   - Kumpulkan 150 sampel EAR & Lip
   - Hitung threshold dinamis

3. **Main Loop:**
   - Ambil frame dari `CameraThread`
   - Jika kalibrasi → tampilkan UI kalibrasi
   - Jika paused → tampilkan "PAUSED"
   - Jika normal → analisis wajah + YOLO + update status
   - Update dashboard dengan data terbaru
   - Tampilkan frame dengan overlay di OpenCV window

4. **Exit:**
   - Stop alarm, release kamera, destroy windows

**Threading Architecture:**
```
Main Thread     → Dashboard GUI (tkinter)
CameraThread    → Capture frame dari webcam
show_thread     → Process frame + show OpenCV window
Alarm Thread    → TTS alarm (pyttsx3)
```

---

## 🔧 Python Libraries yang Digunakan

| Library | Versi | Fungsi |
|---------|-------|--------|
| `opencv-python` | 4.13+ | Video capture, image processing, drawing |
| `mediapipe` | 0.10+ | Face Landmarker detection |
| `ultralytics` | 8.4+ | YOLOv8 object detection |
| `pyttsx3` | 2.99+ | Text-to-Speech alarm |
| `numpy` | 1.24+ | Numerical operations, array math |
| `tkinter` | Built-in | Dashboard GUI |
| `csv` | Built-in | Export session log |
| `threading` | Built-in | Multi-threading |
| `time` | Built-in | Timestamps, sleep |
| `math` | Built-in | Euclidean distance |

---

## 📊 Flow Diagram

```
START
  │
  ├─► CameraThread start (capture frames)
  ├─► Dashboard start (main thread)
  │
  ├─► KALIBRASI (5 detik)
  │    ├─► Collect 150 samples (EAR, Lip)
  │    └─► Compute thresholds (mean ± 2*std)
  │
  ├─► MAIN LOOP
  │    ├─► FaceAnalyzer.analyze(frame)
  │    │    ├─► calculate_ear()
  │    │    ├─► calculate_lip_distance()
  │    │    └─► estimate_head_pose()
  │    │
  │    ├─► YOLOv8.detect(frame)
  │    │    └─► Detect person, phone, book
  │    │
  │    ├─► StateManager.update()
  │    │    ├─► Check phone counter → DISTRACTED / PHONE_ALERT
  │    │    └─► MicrosleepClassifier.update() → NORMAL/YAWNING/DROWSY/MICROSLEEP
  │    │
  │    ├─► SessionTracker.update()
  │    │    └─► Track stats, check break reminder
  │    │
  │    ├─► Alarm.play() / Alarm.stop()
  │    │
  │    ├─► Overlay.draw() → Draw on frame
  │    └─► Dashboard.update() → Update GUI
  │
  └─► EXIT (press 'q')
       ├─► Stop alarm
       ├─► Release camera
       └─► Destroy windows
```
