# Algoritma Perhitungan — Core Detection System

> **Referensi**: Awang & Azhar (2025). *Haar Cascade Algorithm for Microsleep Detection*. Malaysian Journal of Computing, 10(2), 2176-2187.

---

## 1. Eye Aspect Ratio (EAR)

### 1.1 Formula Standar

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Dimana:
- `p1, p4` = titik horizontal mata (sudut dalam dan sudut luar)
- `p2, p3` = titik kelopak mata atas
- `p5, p6` = titik kelopak mata bawah
- `||a - b||` = jarak Euclidean antara dua titik

### 1.2 MediaPipe Face Landmarker Indices

**Mata Kiri (6 titik):**

| Titik | Index | Deskripsi |
|-------|-------|-----------|
| p1 | 33 | Sudut dalam (inner corner) |
| p2 | 160 | Kelopak atas tengah |
| p3 | 158 | Kelopak atas dalam |
| p4 | 133 | Sudut luar (outer corner) |
| p5 | 153 | Kelopak bawah dalam |
| p6 | 144 | Kelopak bawah tengah |

**Mata Kanan (6 titik):**

| Titik | Index | Deskripsi |
|-------|-------|-----------|
| p1 | 362 | Sudut luar (outer corner) |
| p2 | 385 | Kelopak atas tengah |
| p3 | 387 | Kelopak atas dalam |
| p4 | 263 | Sudut dalam (inner corner) |
| p5 | 373 | Kelopak bawah dalam |
| p6 | 380 | Kelopak bawah tengah |

### 1.3 Implementasi

```python
def euclidean_distance(p1, p2):
    """Hitung jarak Euclidean antara dua titik 2D"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks, side='left'):
    """
    Hitung Eye Aspect Ratio menggunakan 6 landmark mata.
    
    Args:
        landmarks: List of MediaPipe landmarks (468 points)
        side: 'left' atau 'right'
    
    Returns:
        EAR value (float)
    """
    if side == 'left':
        p1, p2, p3, p4, p5, p6 = landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]
    else:
        p1, p2, p3, p4, p5, p6 = landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]
    
    # Jarak vertikal
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    
    # Jarak horizontal
    horizontal = euclidean_distance(p1, p4)
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear

def calculate_ear_average(landmarks):
    """Hitung rata-rata EAR dari kedua mata"""
    ear_left = calculate_ear(landmarks, 'left')
    ear_right = calculate_ear(landmarks, 'right')
    return (ear_left + ear_right) / 2.0
```

### 1.4 Interpretasi Nilai EAR

| Kondisi | Nilai EAR | Keterangan |
|---------|-----------|------------|
| Mata terbuka normal | 0.20 - 0.35 | Baseline pengguna |
| Mata setengah tertutup | 0.10 - 0.20 | Mulai mengantuk |
| Mata tertutup (microsleep) | < 0.10 | EAR di bawah threshold |

> **Catatan**: Nilai absolut EAR bervariasi per individu. Oleh karena itu, digunakan **Dynamic Threshold Calibration** untuk menentukan threshold unik setiap pengguna.

---

## 2. Lip Distance (Deteksi Menguap)

### 2.1 Formula

```
Lip Distance = ||landmark_13 - landmark_14||
```

Dimana:
- `landmark_13` = titik tengah bibir atas
- `landmark_14` = titik tengah bibir bawah
- `||a - b||` = jarak Euclidean antara dua titik (dalam satuan normalized)

### 2.2 MediaPipe Face Landmarker Indices

| Titik | Index | Deskripsi |
|-------|-------|-----------|
| Upper Lip Center | 13 | Titik tengah bibir atas |
| Lower Lip Center | 14 | Titik tengah bibir bawah |

### 2.3 Implementasi

```python
def calculate_lip_distance(landmarks):
    """
    Hitung jarak absolut antara bibir atas dan bawah.
    
    Args:
        landmarks: List of MediaPipe landmarks (468 points)
    
    Returns:
        Lip distance value (float, normalized)
    """
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    
    distance = euclidean_distance(upper_lip, lower_lip)
    
    return distance
```

### 2.4 Interpretasi Nilai Lip Distance

| Kondisi | Nilai Lip Distance | Keterangan |
|---------|-------------------|------------|
| Mulut tertutup | < 0.02 | Normal |
| Mulut sedikit terbuka | 0.02 - 0.05 | Bicara normal |
| Menguap | > 0.05 | Indikasi kantuk |

> **Catatan**: Nilai threshold ditentukan melalui **Dynamic Threshold Calibration** (mean + 2*std dari baseline).

---

## 3. Dynamic Threshold Calibration

### 3.1 Algoritma

```
INPUT:  video_stream (5 detik pertama)
OUTPUT: ear_threshold, lip_threshold

1. Inisialisasi:
   ear_samples = []
   lip_samples = []
   total_frames = 5 * FPS  (default: 150 frame @ 30fps)

2. Untuk setiap frame selama 5 detik:
   a. Deteksi face landmarks
   b. Hitung EAR = calculate_ear_average(landmarks)
   c. Hitung Lip Distance = calculate_lip_distance(landmarks)
   d. ear_samples.append(EAR)
   e. lip_samples.append(Lip Distance)

3. Hitung statistik baseline:
   baseline_ear = mean(ear_samples)
   baseline_lip = mean(lip_samples)
   std_ear = std(ear_samples)
   std_lip = std(lip_samples)

4. Hitung threshold dinamis:
   ear_threshold = baseline_ear - (2 * std_ear)
   lip_threshold = baseline_lip + (2 * std_lip)

5. RETURN ear_threshold, lip_threshold
```

### 3.2 Implementasi

```python
class DynamicCalibration:
    """
    Kalibrasi otomatis untuk menentukan threshold unik per pengguna.
    Durasi: 5 detik (150 frame @ 30fps)
    """
    
    def __init__(self, duration=5, fps=30):
        self.duration = duration
        self.fps = fps
        self.total_frames = duration * fps
        self.ear_samples = []
        self.lip_samples = []
        self.current_frame = 0
        
        # Hasil kalibrasi
        self.baseline_ear = None
        self.baseline_lip = None
        self.std_ear = None
        self.std_lip = None
        self.ear_threshold = None
        self.lip_threshold = None
    
    def add_sample(self, ear, lip_distance):
        """
        Tambahkan sampel EAR dan Lip Distance.
        Return True jika kalibrasi selesai.
        """
        if self.current_frame >= self.total_frames:
            return True  # Sudah selesai
        
        self.ear_samples.append(ear)
        self.lip_samples.append(lip_distance)
        self.current_frame += 1
        
        if self.current_frame >= self.total_frames:
            self._compute_thresholds()
            return True
        
        return False
    
    def _compute_thresholds(self):
        """Hitung threshold dinamis dari sampel yang terkumpul."""
        self.baseline_ear = np.mean(self.ear_samples)
        self.baseline_lip = np.mean(self.lip_samples)
        self.std_ear = np.std(self.ear_samples)
        self.std_lip = np.std(self.lip_samples)
        
        # Threshold: 2 standar deviasi dari mean
        self.ear_threshold = self.baseline_ear - (2 * self.std_ear)
        self.lip_threshold = self.baseline_lip + (2 * self.std_lip)
    
    def get_progress(self):
        """Return progress kalibrasi (0.0 - 1.0)."""
        return self.current_frame / self.total_frames
    
    def get_remaining_seconds(self):
        """Return sisa waktu kalibrasi dalam detik."""
        remaining_frames = self.total_frames - self.current_frame
        return remaining_frames / self.fps
```

### 3.3 Alasan Pemilihan 2 * std

- **1 * std**: Terlalu sensitif → banyak false positive
- **2 * std**: Balance antara sensitivitas dan akurasi (mencakup ~95% data normal)
- **3 * std**: Terlalu longgar → bisa miss detection

---

## 4. Microsleep Classification (Time-Series State Machine)

### 4.1 Definisi Microsleep (Paper)

> "Brief periods of unconsciousness, typically lasting between **1 and 15 seconds**"
> 
> — Awang & Azhar (2025)

### 4.2 State Machine

```
┌─────────┐    lip > threshold     ┌──────────┐    3x yawning/10s    ┌─────────┐    EAR < threshold    ┌────────────┐
│  NORMAL │ ──────────────────────> │  YAWNING │ ────────────────────> │  DROWSY │ ───────────────────> │ MICROSLEEP │
└─────────┘                         └──────────┘                       └─────────┘                      └────────────┘
     ↑                                    ↑                                  ↑                                │
     │         normal kembali             │         normal kembali           │         normal kembali          │
     └────────────────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

### 4.3 Parameter Time-Series

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `FPS` | 30 | Frame per second (asumsi) |
| `YAWN_SEQUENCE_THRESHOLD` | 3 | Jumlah menguap beruntun |
| `YAWN_WINDOW_SECONDS` | 10 | Window waktu untuk hitung yawning |
| `MICROSLEEP_MIN_FRAMES` | 30 | Minimum frame EAR rendah (1 detik @ 30fps) |
| `MICROSLEEP_MAX_SECONDS` | 15 | Batas maksimum microsleep (paper) |
| `BLINK_MAX_FRAMES` | 9 | Maksimum frame untuk kedipan normal (< 0.3 detik) |

### 4.4 Algoritma Klasifikasi

```
INPUT:  ear, lip_distance, ear_threshold, lip_threshold
OUTPUT: status ('NORMAL', 'YAWNING', 'DROWSY', 'MICROSLEEP')

1. Inisialisasi counters:
   ear_low_counter = 0       # Frame berturut-turut EAR < threshold
   lip_high_counter = 0      # Frame berturut-turut lip > threshold
   yawn_timestamps = []      # List timestamp setiap menguap terdeteksi

2. Untuk setiap frame:
   
   a. Update EAR counter:
      if ear < ear_threshold:
          ear_low_counter += 1
      else:
          ear_low_counter = 0  # Reset jika mata terbuka
   
   b. Update Lip counter:
      if lip_distance > lip_threshold:
          lip_high_counter += 1
          if lip_high_counter == 1:  # First frame of yawning
              yawn_timestamps.append(current_time)
      else:
          lip_high_counter = 0  # Reset jika mulut tertutup
   
   c. Hitung yawning dalam window 10 detik:
      recent_yawns = count(yawn_timestamps where (now - t) <= 10s)
   
   d. Klasifikasi state:
      if ear_low_counter >= MICROSLEEP_MIN_FRAMES:
          status = 'MICROSLEEP'    # Mata terpejam >= 1 detik
      elif recent_yawns >= YAWN_SEQUENCE_THRESHOLD:
          status = 'DROWSY'        # 3x menguap dalam 10 detik
      elif lip_distance > lip_threshold:
          status = 'YAWNING'       # Sedang menguap
      else:
          status = 'NORMAL'        # Kondisi normal
   
   e. Cleanup timestamps:
      yawn_timestamps = [t for t in yawn_timestamps if (now - t) <= 10s]

3. RETURN status
```

### 4.5 Implementasi

```python
import time

class MicrosleepClassifier:
    """
    Time-series state machine untuk klasifikasi microsleep.
    State: NORMAL → YAWNING → DROWSY → MICROSLEEP
    """
    
    # State constants
    NORMAL = 'NORMAL'
    YAWNING = 'YAWNING'
    DROWSY = 'DROWSY'
    MICROSLEEP = 'MICROSLEEP'
    
    def __init__(self, ear_threshold, lip_threshold, fps=30):
        self.ear_threshold = ear_threshold
        self.lip_threshold = lip_threshold
        self.fps = fps
        
        # Counters
        self.ear_low_counter = 0
        self.lip_high_counter = 0
        self.yawn_timestamps = []
        
        # Thresholds
        self.yawn_sequence_threshold = 3        # 3x menguap
        self.yawn_window_seconds = 10           # dalam 10 detik
        self.microsleep_min_frames = fps        # 1 detik (30 frame @ 30fps)
        self.blink_max_frames = int(fps * 0.3)  # 0.3 detik (9 frame @ 30fps)
        
        # State
        self.current_state = self.NORMAL
    
    def update(self, ear, lip_distance):
        """
        Update state machine dengan data frame terbaru.
        Return: current state
        """
        # 1. Update EAR counter
        if ear < self.ear_threshold:
            self.ear_low_counter += 1
        else:
            self.ear_low_counter = 0
        
        # 2. Update Lip counter
        if lip_distance > self.lip_threshold:
            if self.lip_high_counter == 0:
                self.yawn_timestamps.append(time.time())
            self.lip_high_counter += 1
        else:
            self.lip_high_counter = 0
        
        # 3. Hitung yawning dalam window
        recent_yawns = self._count_recent_yawns()
        
        # 4. Klasifikasi state
        if self.ear_low_counter >= self.microsleep_min_frames:
            self.current_state = self.MICROSLEEP
        elif recent_yawns >= self.yawn_sequence_threshold:
            self.current_state = self.DROWSY
        elif lip_distance > self.lip_threshold:
            self.current_state = self.YAWNING
        else:
            self.current_state = self.NORMAL
        
        return self.current_state
    
    def _count_recent_yawns(self):
        """Hitung jumlah menguap dalam window waktu terakhir."""
        now = time.time()
        window = self.yawn_window_seconds
        recent = [t for t in self.yawn_timestamps if (now - t) <= window]
        self.yawn_timestamps = recent  # Cleanup
        return len(recent)
    
    def reset(self):
        """Reset semua counters dan state."""
        self.ear_low_counter = 0
        self.lip_high_counter = 0
        self.yawn_timestamps = []
        self.current_state = self.NORMAL
```

---

## 5. Anti-False Positive Filter

### 5.1 Blink vs Microsleep

| Karakteristik | Kedipan Normal | Microsleep |
|---------------|----------------|------------|
| Durasi | < 0.3 detik (~9 frame) | >= 1 detik (~30 frame) |
| EAR pattern | Turun cepat, naik cepat | Turun stabil, tetap rendah |
| Frekuensi | 15-20x per menit | Jarang, tapi durasi panjang |

**Filter Logic:**
```python
if ear_low_counter < blink_max_frames:
    # Ini hanya kedipan normal, ignore
    status = 'NORMAL'
elif ear_low_counter >= microsleep_min_frames:
    # Ini microsleep
    status = 'MICROSLEEP'
```

### 5.2 Yawning vs Speaking

| Karakteristik | Menguap | Bicara Normal |
|---------------|---------|---------------|
| Durasi lip tinggi | > 0.5 detik (~15 frame) | < 0.3 detik (fluktuasi cepat) |
| Lip distance | Sangat tinggi (> threshold) | Sedang (di bawah threshold) |
| Pattern | Stabil tinggi | Fluktuatif |

**Filter Logic:**
```python
if lip_high_counter >= 15:  # > 0.5 detik
    # Ini menguap, bukan bicara
    status = 'YAWNING'
```

### 5.3 Face Lost Handling

Jika face landmarks hilang dari frame:
```python
if face_landmarks is None:
    # Reset semua counters
    classifier.reset()
    # Jangan trigger alarm
    status = 'FACE_LOST'
```

### 5.4 Jarak Kamera (Paper Table 2)

| Jarak | Akurasi | Rekomendasi |
|-------|---------|-------------|
| 30-40cm | 85-90% | Ideal |
| 50-60cm | 75-85% | Acceptable |
| 70-90cm | 60-70% | Warning |
| >100cm | <50% | Tidak reliable |

**Implementasi:**
```python
def estimate_face_distance(landmarks, frame_height):
    """
    Estimasi jarak wajah dari ukuran bounding box wajah.
    Return: 'ideal', 'acceptable', 'warning', 'poor'
    """
    # Hitung ukuran wajah dalam pixel
    face_width = max(l.x for l in landmarks) - min(l.x for l in landmarks)
    face_pixel_width = face_width * frame_width
    
    if face_pixel_width > 200:
        return 'ideal'      # 30-40cm
    elif face_pixel_width > 150:
        return 'acceptable' # 50-60cm
    elif face_pixel_width > 100:
        return 'warning'    # 70-90cm
    else:
        return 'poor'       # >100cm
```

---

## 6. Drowsiness Score (Composite Metric)

### 6.1 Formula

```
Drowsiness Score = w1 * (EAR_normalized) + w2 * (Lip_normalized) + w3 * (Yawn_frequency)
```

Dimana:
- `w1, w2, w3` = bobot masing-masing komponen (default: 0.4, 0.3, 0.3)
- `EAR_normalized` = (baseline_ear - current_ear) / baseline_ear
- `Lip_normalized` = (current_lip - baseline_lip) / baseline_lip
- `Yawn_frequency` = jumlah_menguap / window_seconds

### 6.2 Interpretasi

| Drowsiness Score | Status |
|------------------|--------|
| 0.0 - 0.3 | Normal |
| 0.3 - 0.6 | Yawning |
| 0.6 - 0.8 | Drowsy |
| 0.8 - 1.0 | Microsleep |

---

## 7. Ringkasan Parameter Konfigurasi

| Parameter | Nilai Default | Sumber |
|-----------|---------------|--------|
| `CALIBRATION_DURATION` | 5 detik | Custom |
| `CALIBRATION_FRAMES` | 150 (5 * 30) | Custom |
| `THRESHOLD_STD_MULTIPLIER` | 2 | Statistical (95% confidence) |
| `YAWN_SEQUENCE_THRESHOLD` | 3 | Custom |
| `YAWN_WINDOW_SECONDS` | 10 | Custom |
| `MICROSLEEP_MIN_FRAMES` | 30 (1 detik @ 30fps) | Paper (1-15 seconds) |
| `MICROSLEEP_MAX_SECONDS` | 15 | Paper |
| `BLINK_MAX_FRAMES` | 9 (0.3 detik @ 30fps) | Anti-false positive |
| `YAWNING_MIN_FRAMES` | 15 (0.5 detik @ 30fps) | Anti-false positive |
| `FPS` | 30 | Asumsi webcam standar |

---

## 8. Referensi

1. Awang, N. & Azhar, A.M. (2025). *Haar Cascade Algorithm for Microsleep Detection*. Malaysian Journal of Computing, 10(2), 2176-2187.
2. Soukupova, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*.
3. Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*.
