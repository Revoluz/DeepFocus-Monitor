# Drowsiness Detection (YOLOv8 + MediaPipe Face Landmarker)

Aplikasi ini mendeteksi tanda kantuk melalui jarak vertikal kelopak mata (berbasis landmark mata) dan juga mendeteksi objek orang serta ponsel menggunakan YOLOv8.

## Fitur
- Deteksi orang dan ponsel (YOLOv8 COCO classes).
- Deteksi landmark wajah dan indikasi mata terpejam (MediaPipe Face Landmarker Tasks API).
- Menampilkan peringatan "MENGANTUK!" saat mata terdeteksi tertutup.

## Prasyarat
- Python 3.9+ (disarankan 3.10 atau 3.11)
- Webcam aktif
- OS: Linux/Windows/macOS

## Instalasi
1. (Opsional) Buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependensi utama:
   ```bash
   pip install ultralytics mediapipe opencv-python
   ```

3. Pastikan file model sudah ada di root project:
   - `yolov8n.pt`
   - `face_landmarker.task`

   Jika `face_landmarker.task` belum ada, aplikasi akan mengunduh otomatis saat pertama kali dijalankan.

## Cara Menjalankan
```bash
python main.py
```

- Tekan `q` untuk keluar.
- Jika kamera Anda bukan di indeks `2`, ubah di [main.py](main.py#L45) bagian:
  ```python
  cap = cv2.VideoCapture(2)
  ```
  Contoh: `0` untuk webcam default.

## Penjelasan Singkat Cara Kerja
1. **YOLOv8** memproses setiap frame untuk mendeteksi kelas `person` (0) dan `cell phone` (67). Hasilnya dipakai untuk menampilkan bounding box pada frame.
2. **MediaPipe Face Landmarker** mendeteksi landmark wajah dan mengambil beberapa titik mata.
3. Jarak vertikal landmark mata dihitung sebagai indikasi mata tertutup. Jika jarak di bawah ambang (threshold), maka menampilkan peringatan "MENGANTUK!".

## Catatan dan Penyesuaian
- Nilai threshold mata di [main.py](main.py#L86) saat ini `0.015`. Anda bisa menyesuaikan agar lebih akurat sesuai kondisi kamera dan pencahayaan.
- Untuk deteksi kantuk yang lebih robust, gunakan perhitungan Eye Aspect Ratio (EAR) dengan beberapa landmark mata.

## Troubleshooting
- **Kamera tidak terbuka**: Ubah indeks kamera di `cv2.VideoCapture(...)`.
- **Model tidak ditemukan**: Pastikan `yolov8n.pt` ada di root project.
- **Lag/lemot**: Kurangi resolusi kamera atau gunakan model YOLO yang lebih ringan.
