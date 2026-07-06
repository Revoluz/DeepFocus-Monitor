"""
ui/dashboard.py - Dashboard GUI Monitoring Real-Time

File ini berisi kelas SimpleDashboard yang menyediakan antarmuka grafis
untuk monitoring status sistem secara real-time menggunakan tkinter.

Fitur Dashboard:
1. Session Timer: Menampilkan durasi sesi belajar (HH:MM:SS)
2. Focus Score: Persentase waktu fokus selama sesi
3. Status Label: Status saat ini dengan warna (FOKUS, MENGUAP, dll)
4. Live Metrics: EAR, LIP, YAW angle real-time
5. Counter: Total microsleep dan yawning dalam sesi
6. EAR Chart: Grafik live Eye Aspect Ratio dengan threshold line
7. Break Reminder: Pesan pengingat istirahat jika trigger
8. Tombol: Pause/Resume, Kalibrasi Ulang, Export CSV

Threading:
- Dashboard HARUS berjalan di main thread (requirement tkinter)
- OpenCV window dan camera capture berjalan di thread terpisah
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time


class SimpleDashboard:
    """
    Kelas untuk dashboard GUI monitoring menggunakan tkinter.

    Widget utama:
    - timer_label: Session timer (HH:MM:SS)
    - focus_label: Focus score (%)
    - status_label: Status saat ini (berwarna)
    - ear_label, lip_label, yaw_label: Live metrics
    - count_label: Counter microsleep & yawning
    - chart_canvas: Canvas untuk EAR live chart
    - break_label: Break reminder message
    - Buttons: Pause, Kalibrasi Ulang, Export CSV

    Cara penggunaan:
    1. Buat instance: dashboard = SimpleDashboard(session_tracker, state_manager, alarm)
    2. Update data: dashboard.update(status, ear, lip_distance, head_pose)
    3. Jalankan: dashboard.run() (di main thread)
    """

    def __init__(self, session_tracker, state_manager, alarm):
        """
        Inisialisasi dashboard window dan semua widget.

        Args:
            session_tracker: SessionTracker instance untuk data statistik
            state_manager: StateManager instance untuk cek break reminder
            alarm: Alarm instance untuk play alarm saat break reminder
        """
        self.session_tracker = session_tracker
        self.state_manager = state_manager
        self.alarm = alarm

        # Setup window tkinter
        self.root = tk.Tk()
        self.root.title("DeepFocus Dashboard")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # State kontrol
        self.paused = False  # Flag pause/resume monitoring
        self.recalibrating = False  # Flag kalibrasi ulang

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """
        Membangun layout dan semua widget tkinter.

        Layout:
        [Session Timer]
        [Focus Score] [Status Label]
        [EAR] [LIP] [YAW] [Counters]
        [EAR Chart Canvas]
        [Break Reminder Message]
        [Pause] [Kalibrasi Ulang] [Export CSV]
        """
        # Session Timer
        self.timer_label = tk.Label(self.root, text="Session: 00:00:00", font=("Arial", 16, "bold"))
        self.timer_label.pack(pady=10)

        # Row: Focus Score + Status
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(fill=tk.X, padx=20, pady=5)

        self.focus_label = tk.Label(stats_frame, text="Focus Score: 100%", font=("Arial", 14))
        self.focus_label.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(stats_frame, text="Status: FOKUS", font=("Arial", 14), fg="green")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Row: Live metrics
        detail_frame = tk.Frame(self.root)
        detail_frame.pack(fill=tk.X, padx=20, pady=5)

        self.ear_label = tk.Label(detail_frame, text="EAR: -", font=("Arial", 11))
        self.ear_label.pack(side=tk.LEFT, padx=10)

        self.lip_label = tk.Label(detail_frame, text="LIP: -", font=("Arial", 11))
        self.lip_label.pack(side=tk.LEFT, padx=10)

        self.yaw_label = tk.Label(detail_frame, text="YAW: -", font=("Arial", 11))
        self.yaw_label.pack(side=tk.LEFT, padx=10)

        self.count_label = tk.Label(detail_frame, text="Microsleep: 0 | Yawning: 0", font=("Arial", 11))
        self.count_label.pack(side=tk.RIGHT, padx=10)

        # EAR Chart Canvas
        self.chart_canvas = tk.Canvas(self.root, bg="white", height=200, relief=tk.SUNKEN, bd=2)
        self.chart_canvas.pack(fill=tk.X, padx=20, pady=10)

        # Break Reminder Message
        self.break_label = tk.Label(self.root, text="", font=("Arial", 12), fg="red", wraplength=550)
        self.break_label.pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.pause_btn = tk.Button(btn_frame, text="⏸ Pause", font=("Arial", 12), command=self.toggle_pause, width=10)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.recalib_btn = tk.Button(btn_frame, text="🔄 Kalibrasi Ulang", font=("Arial", 12), command=self.start_recalibration, width=15)
        self.recalib_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(btn_frame, text="📊 Export CSV", font=("Arial", 12), command=self.export_csv, width=12)
        self.export_btn.pack(side=tk.LEFT, padx=5)

    def update(self, status, ear, lip_distance, head_pose):
        """
        Update semua widget dengan data terbaru dari main loop.

        Dipanggil setiap frame setelah proses analisis selesai.

        Args:
            status: Status saat ini (NORMAL, YAWNING, dll)
            ear: Nilai EAR saat ini
            lip_distance: Jarak bibir saat ini
            head_pose: Dictionary {'yaw', 'pitch', 'roll'}
        """
        # Update session timer
        self.timer_label.configure(text=f"Session: {self.session_tracker.get_session_duration()}")

        # Update focus score
        focus = self.session_tracker.get_focus_score()
        self.focus_label.configure(text=f"Focus Score: {focus:.1f}%")

        # Update status label dengan warna
        status_colors = {
            'NORMAL': 'green',
            'YAWNING': 'orange',
            'DROWSY': 'orange',
            'DISTRACTED': 'darkorange',
            'MICROSLEEP': 'red',
            'PHONE_ALERT': 'red',
            'FACE_LOST': 'gray',
        }
        status_labels = {
            'NORMAL': 'FOKUS',
            'YAWNING': 'MENGUAP',
            'DROWSY': 'MENGANTUK',
            'DISTRACTED': 'TERDISTRASI: HP',
            'MICROSLEEP': 'MICROSLEEP!',
            'PHONE_ALERT': 'HP TERLALU LAMA!',
            'FACE_LOST': 'WAJAH HILANG',
        }
        color = status_colors.get(status, 'black')
        label = status_labels.get(status, status)
        self.status_label.configure(text=f"Status: {label}", fg=color)

        # Update live metrics
        if ear is not None:
            self.ear_label.configure(text=f"EAR: {ear:.4f}")
        if lip_distance is not None:
            self.lip_label.configure(text=f"LIP: {lip_distance:.4f}")
        if head_pose:
            yaw = head_pose.get('yaw', 0)
            self.yaw_label.configure(text=f"YAW: {yaw:.1f}°")

        # Update counters
        self.count_label.configure(text=f"Microsleep: {self.session_tracker.microsleep_total}x | Yawning: {self.session_tracker.yawning_total}x")

        # Redraw EAR chart
        self._draw_ear_chart()

        # Cek break reminder
        break_msg = self.state_manager.check_break_reminder(self.session_tracker)
        if break_msg:
            self.break_label.configure(text=f"⚠️ {break_msg}")
            self.alarm.play('MICROSLEEP')
        else:
            self.break_label.configure(text="")

    def _draw_ear_chart(self):
        """
        Menggambar grafik EAR live pada canvas.

        Fitur:
        - Plot 100 data point terakhir dari ear_history
        - Garis threshold merah (dashed)
        - Auto-scale berdasarkan min/max data
        - Label "EAR Live" di pojok kiri atas
        """
        self.chart_canvas.delete("all")  # Clear canvas

        history = self.session_tracker.ear_history[-100:]  # Ambil 100 data terakhir
        if len(history) < 2:
            return  # Tidak cukup data untuk plot

        width = 560
        height = 180
        padding = 20

        # Auto-scale berdasarkan min/max data
        min_val = min(history) * 0.9
        max_val = max(history) * 1.1
        if max_val == min_val:
            max_val = min_val + 0.01

        # Label chart
        self.chart_canvas.create_text(30, 10, text="EAR Live", font=("Arial", 10), anchor=tk.NW)

        # Garis threshold (merah, dashed)
        threshold = self.state_manager.calibration.ear_threshold
        if threshold:
            y_thresh = padding + (1 - (threshold - min_val) / (max_val - min_val)) * (height - 2 * padding)
            self.chart_canvas.create_line(padding, y_thresh, width - padding, y_thresh, fill="red", dash=(4, 4))
            self.chart_canvas.create_text(width - padding, y_thresh - 5, text="Threshold", fill="red", font=("Arial", 8), anchor=tk.E)

        # Plot garis EAR (biru)
        points = []
        for i, val in enumerate(history):
            x = padding + (i / (len(history) - 1)) * (width - 2 * padding)
            y = padding + (1 - (val - min_val) / (max_val - min_val)) * (height - 2 * padding)
            points.append((x, y))

        for i in range(len(points) - 1):
            self.chart_canvas.create_line(points[i], points[i + 1], fill="blue", width=2)

    def toggle_pause(self):
        """
        Toggle pause/resume monitoring.

        Saat paused:
        - Kamera tetap nyala
        - Analisis EAR/LIP berhenti
        - Counter freeze
        - Alarm tidak aktif
        """
        self.paused = not self.paused
        self.pause_btn.configure(text="▶ Resume" if self.paused else "⏸ Pause")

    def start_recalibration(self):
        """
        Memulai kalibrasi ulang dengan konfirmasi dialog.

        Efek:
        - Reset semua calibration data
        - Reset session tracker
        - Pause monitoring selama kalibrasi 5 detik
        - Tampilkan countdown di OpenCV window
        """
        if messagebox.askyesno("Kalibrasi Ulang", "Yakin ingin kalibrasi ulang? Monitoring akan pause selama 5 detik."):
            self.recalibrating = True
            self.paused = True
            self.pause_btn.configure(text="▶ Resume")
            self.state_manager.reset_calibration()
            self.session_tracker.reset()

    def export_csv(self):
        """
        Export session log ke file CSV.

        File berisi:
        - timestamp (frame number)
        - ear (Eye Aspect Ratio)
        - lip_distance (jarak bibir)
        """
        success = self.session_tracker.export_csv()
        if success:
            messagebox.showinfo("Export", "Session log berhasil di-export ke session_log.csv")
        else:
            messagebox.showwarning("Export", "Belum ada data untuk di-export.")

    def run(self):
        """
        Menjalankan tkinter mainloop.

        PENTING: Harus dipanggil di main thread (requirement tkinter).
        """
        self.root.mainloop()
