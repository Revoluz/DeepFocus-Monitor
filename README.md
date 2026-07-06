# Drowsiness & Study Distraction Detection System

Platform asisten pemantau produktivitas dan kedisiplinan belajar mandiri berbasis **Computer Vision** dan **Artificial Intelligence** yang berjalan secara real-time.

Sistem mengintegrasikan analisis biometrik wajah dan deteksi objek di sekitar lingkungan pengguna menggunakan kombinasi teknologi modern: **YOLOv8** (deteksi objek) dan **MediaPipe Face Landmarker** (analisis struktur wajah).

---

## 🎯 Fitur Utama

### 1. Deteksi Kelelahan & Kantuk (Biometrik Wajah)
| Fitur | Teknologi | Deskripsi |
|-------|-----------|-----------|
| **Eye Aspect Ratio (EAR)** | MediaPipe Face Landmarker | Menghitung rasio aspek kelopak mata untuk mendeteksi micro-sleep (mata terpejam lama) |
| **Mouth Aspect Ratio (MAR)** | MediaPipe Face Landmarker | Memantau bukaan bibir untuk mendeteksi aktivitas menguap (yawning) |

### 2. Deteksi Distraksi & Fokus (Objek & Orientasi)
| Fitur | Teknologi | Deskripsi |
|-------|-----------|-----------|
| **Handphone Detection** | YOLOv8 (COCO class 67) | Melacak keberadaan objek pengalih perhatian (HP) |
| **Book Detection** | YOLOv8 (COCO class 73) | Memastikan keberadaan buku sebagai indikator belajar |
| **Head Pose Estimation** | MediaPipe + solvePnP | Menganalisis arah wajah (yaw, pitch, roll) untuk mendeteksi apakah pengguna menoleh/terdistraksi |

### 3. Intervensi & Adaptabilitas Cerdas
| Fitur | Deskripsi |
|-------|-----------|
| **Auto Kalibrasi** | 5 detik pertama saat startup untuk menyesuaikan threshold dengan bentuk wajah tiap pengguna |
| **Consecutive Counter** | Penghitung durasi frame berturut-turut agar tidak terjadi false alarm saat berkedip/bericara normal |
| **Smart Sound Alarm** | Alarm suara pintar (pygame) untuk mengembalikan fokus saat terdeteksi mengantuk berat atau bermain HP terlalu lama |

---

## 🎨 Output Visualisasi (Overlay UI)

Sistem menampilkan border dan overlay dinamis yang berubah warna sesuai status pengguna:

| Status | Warna | Kondisi Trigger |
|--------|-------|-----------------|
| **🟢 FOKUS** | Hijau `(0, 255, 0)` | Kondisi ideal saat belajar |
| **🟠 PERINGATAN** | Oranye `(0, 165, 255)` | Terdeteksi menguap ATAU menoleh |
| **🔴 BAHAYA** | Merah `(0, 0, 255)` | Terdeteksi micro-sleep ATAU bermain HP > batas waktu aman |

---

## 🧠 Klasifikasi Status Kelelahan

Sistem membedakan antara **mengantuk (DROWSY)** dan **microsleep (MICROSLEEP)** berdasarkan pola biometrik:

| Aspek | DROWSY (Mengantuk) | MICROSLEEP (Tertidur Singkat) |
|-------|-------------------|------------------------------|
| **Trigger** | 3x menguap dalam 10 detik | EAR < threshold selama ≥ 30 frame (1 detik) |
| **Indikator** | Lip Distance > threshold berulang | Mata terpejam terus-menerus |
| **Durasi** | Pattern berulang | Stabil ≥ 1 detik |
| **Severity** | Peringatan awal (Oranye) | Bahaya (Merah) |
| **Status UI** | `PERINGATAN: MENGANTUK` | `BAHAYA: MICROSLEEP!` |
| **Alarm** | Tidak ada | ✅ Sound alarm aktif |

### State Machine Klasifikasi

```
NORMAL → YAWNING → DROWSY → MICROSLEEP
  ↑          ↑         ↑          ↑
  └──────────┴─────────┴──────────┘
         (Reset jika kondisi normal kembali)
```

**Logika Prioritas (dari yang paling kritis):**
1. **MICROSLEEP**: Mata terpejam ≥ 1 detik → `status = 'MICROSLEEP'`
2. **DROWSY**: 3x menguap dalam 10 detik → `status = 'DROWSY'`
3. **YAWNING**: Sedang menguap (1x) → `status = 'YAWNING'`
4. **NORMAL**: Tidak ada tanda kelelahan → `status = 'NORMAL'`

### Perbedaan Kunci

- **DROWSY** = User masih sadar tapi menunjukkan tanda kantuk (menguap berulang). Ini adalah **peringatan dini** agar user istirahat atau memperbaiki postur.
- **MICROSLEEP** = User sudah kehilangan kesadaran sesaat (mata terpejam > 1 detik). Ini adalah **kondisi kritis** yang memerlukan intervensi langsung (alarm suara).

---

## 📁 Struktur Folder

```
drowsiness-detection/
├── main.py                          # Entry point - integrasi semua modul
├── config.py                        # Konfigurasi global (thresholds, constants, camera index)
├── face_landmarker.task             # MediaPipe model (auto-download jika belum ada)
├── yolov8n.pt                       # YOLOv8 pre-trained model (COCO dataset)
│
├── core/                            # Modul inti analisis
│   ├── __init__.py
│   ├── face_analyzer.py             # EAR, MAR, Head Pose estimation via MediaPipe
│   ├── object_detector.py           # YOLOv8 object detection wrapper
│   ├── state_manager.py             # State tracking, calibration, consecutive counters
│   └── alarm.py                     # Sound alarm system via pygame
│
├── ui/                              # Modul visualisasi
│   ├── __init__.py
│   └── overlay.py                   # Dynamic overlay drawing, color themes, stats display
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   └── helpers.py                   # Euclidean distance, angle calculation, etc.
│
├── yolov5/                          # Dataset custom untuk training model sendiri
│   └── study-distraction-v4.v1i.yolov8/
│       ├── train/                   # Training images & labels
│       ├── valid/                   # Validation images & labels
│       └── test/                    # Test images & labels
│
└── README.md                        # Dokumentasi project
```

---

## 🏗️ Arsitektur Class

### `core/face_analyzer.py` → `FaceAnalyzer`
- `calculate_ear(landmarks)` → Eye Aspect Ratio
- `calculate_mar(landmarks)` → Mouth Aspect Ratio
- `estimate_head_pose(landmarks, frame_shape)` → yaw, pitch, roll
- `analyze(frame)` → main analysis pipeline

### `core/object_detector.py` → `ObjectDetector`
- `detect(frame)` → YOLOv8 detection results (person, book, cell phone)

### `core/state_manager.py` → `StateManager`
- `calibrate(ear, mar)` → auto-calibration (5 detik)
- `update(ear, mar, yaw, phone_detected)` → status determination (FOKUS/PERINGATAN/BAHAYA)

### `core/alarm.py` → `Alarm`
- `play()` / `stop()` → sound alert control

### `ui/overlay.py` → `Overlay`
- `draw(frame, status, info)` → render border, text, stats pada frame

---

## 📦 Prasyarat

- Python 3.9+ (disarankan 3.10 atau 3.11)
- Webcam aktif
- OS: Linux/Windows/macOS

---

## ⚙️ Instalasi

1. **Buat dan aktifkan virtual environment** (opsional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # atau
   .venv\Scripts\activate     # Windows
   ```

2. **Install dependensi**:
   ```bash
   pip install ultralytics mediapipe opencv-python pygame numpy
   ```

3. **Pastikan file model tersedia**:
   - `yolov8n.pt` - YOLOv8 pre-trained (COCO)
   - `face_landmarker.task` - MediaPipe model (auto-download jika belum ada)

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

**Saat startup:**
1. Sistem akan melakukan **kalibrasi otomatis** selama 5 detik (tampilkan countdown)
2. Setelah kalibrasi, sistem mulai memantau status fokus Anda
3. Tekan `q` untuk keluar

**Pengaturan kamera:**
Jika webcam Anda bukan di indeks `2`, ubah di `config.py`:
```python
CAMERA_INDEX = 2  # Ubah sesuai indeks kamera Anda (0 = default)
```

---

## 🔧 Konfigurasi

Semua parameter dapat disesuaikan di `config.py`:

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `CAMERA_INDEX` | `2` | Indeks webcam |
| `CALIBRATION_DURATION` | `5` | Durasi kalibrasi (detik) |
| `EAR_THRESHOLD_OFFSET` | `-0.05` | Offset dari baseline EAR |
| `MAR_THRESHOLD_OFFSET` | `0.1` | Offset dari baseline MAR |
| `YAW_THRESHOLD` | `20` | Batas sudut yaw (derajat) |
| `MICROSLEEP_FRAMES` | `30` | Frame count untuk micro-sleep (~1 detik @30fps) |
| `YAWN_FRAMES` | `15` | Frame count untuk menguap (~0.5 detik) |
| `HEAD_TURNED_FRAMES` | `20` | Frame count untuk menoleh (~0.7 detik) |
| `PHONE_LIMIT_FRAMES` | `300` | Frame count batas HP (~10 detik) |

---

## 📝 Catatan Pengembangan

- **Model saat ini**: Menggunakan YOLOv8n pre-trained (COCO dataset) untuk deteksi `person (0)`, `cell phone (67)`, `book (73)`
- **Dataset custom**: Tersedia di `yolov5/study-distraction-v4.v1i.yolov8/` untuk training model yang lebih spesifik
- **Untuk training model custom**: Gunakan dataset di folder `yolov5/` dengan YOLOv8 training pipeline

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Kamera tidak terbuka | Ubah `CAMERA_INDEX` di `config.py` |
| Model YOLO tidak ditemukan | Pastikan `yolov8n.pt` ada di root project |
| Lag/lemot | Kurangi resolusi kamera atau gunakan model YOLO lebih ringan |
| Alarm tidak berbunyi | Pastikan `pygame` terinstall dan audio system aktif |
| Kalibrasi tidak akurat | Pastikan pencahayaan cukup dan wajah menghadap kamera saat kalibrasi |

---

## 📄 Lisensi

Project ini dibuat untuk tujuan pembelajaran dan penelitian.
