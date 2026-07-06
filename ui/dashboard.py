import tkinter as tk
from tkinter import messagebox
import threading
import time


class SimpleDashboard:
    def __init__(self, session_tracker, state_manager, alarm):
        self.session_tracker = session_tracker
        self.state_manager = state_manager
        self.alarm = alarm

        self.root = tk.Tk()
        self.root.title("DeepFocus Dashboard")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self.paused = False
        self.recalibrating = False

        self._build_ui()

    def _build_ui(self):
        self.timer_label = tk.Label(self.root, text="Session: 00:00:00", font=("Arial", 16, "bold"))
        self.timer_label.pack(pady=10)

        stats_frame = tk.Frame(self.root)
        stats_frame.pack(fill=tk.X, padx=20, pady=5)

        self.focus_label = tk.Label(stats_frame, text="Focus Score: 100%", font=("Arial", 14))
        self.focus_label.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(stats_frame, text="Status: FOKUS", font=("Arial", 14), fg="green")
        self.status_label.pack(side=tk.RIGHT, padx=10)

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

        self.chart_canvas = tk.Canvas(self.root, bg="white", height=200, relief=tk.SUNKEN, bd=2)
        self.chart_canvas.pack(fill=tk.X, padx=20, pady=10)

        self.break_label = tk.Label(self.root, text="", font=("Arial", 12), fg="red", wraplength=550)
        self.break_label.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.pause_btn = tk.Button(btn_frame, text="⏸ Pause", font=("Arial", 12), command=self.toggle_pause, width=10)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.recalib_btn = tk.Button(btn_frame, text="🔄 Kalibrasi Ulang", font=("Arial", 12), command=self.start_recalibration, width=15)
        self.recalib_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(btn_frame, text="📊 Export CSV", font=("Arial", 12), command=self.export_csv, width=12)
        self.export_btn.pack(side=tk.LEFT, padx=5)

    def update(self, status, ear, lip_distance, head_pose):
        self.timer_label.configure(text=f"Session: {self.session_tracker.get_session_duration()}")

        focus = self.session_tracker.get_focus_score()
        self.focus_label.configure(text=f"Focus Score: {focus:.1f}%")

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

        if ear is not None:
            self.ear_label.configure(text=f"EAR: {ear:.4f}")
        if lip_distance is not None:
            self.lip_label.configure(text=f"LIP: {lip_distance:.4f}")
        if head_pose:
            yaw = head_pose.get('yaw', 0)
            self.yaw_label.configure(text=f"YAW: {yaw:.1f}°")

        self.count_label.configure(text=f"Microsleep: {self.session_tracker.microsleep_total}x | Yawning: {self.session_tracker.yawning_total}x")

        self._draw_ear_chart()

        break_msg = self.state_manager.check_break_reminder(self.session_tracker)
        if break_msg:
            self.break_label.configure(text=f"⚠️ {break_msg}")
            self.alarm.play('MICROSLEEP')
        else:
            self.break_label.configure(text="")

    def _draw_ear_chart(self):
        self.chart_canvas.delete("all")

        history = self.session_tracker.ear_history[-100:]
        if len(history) < 2:
            return

        width = 560
        height = 180
        padding = 20

        min_val = min(history) * 0.9
        max_val = max(history) * 1.1
        if max_val == min_val:
            max_val = min_val + 0.01

        self.chart_canvas.create_text(30, 10, text="EAR Live", font=("Arial", 10), anchor=tk.NW)

        threshold = self.state_manager.calibration.ear_threshold
        if threshold:
            y_thresh = padding + (1 - (threshold - min_val) / (max_val - min_val)) * (height - 2 * padding)
            self.chart_canvas.create_line(padding, y_thresh, width - padding, y_thresh, fill="red", dash=(4, 4))
            self.chart_canvas.create_text(width - padding, y_thresh - 5, text="Threshold", fill="red", font=("Arial", 8), anchor=tk.E)

        points = []
        for i, val in enumerate(history):
            x = padding + (i / (len(history) - 1)) * (width - 2 * padding)
            y = padding + (1 - (val - min_val) / (max_val - min_val)) * (height - 2 * padding)
            points.append((x, y))

        for i in range(len(points) - 1):
            self.chart_canvas.create_line(points[i], points[i + 1], fill="blue", width=2)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="▶ Resume" if self.paused else "⏸ Pause")

    def start_recalibration(self):
        if messagebox.askyesno("Kalibrasi Ulang", "Yakin ingin kalibrasi ulang? Monitoring akan pause selama 5 detik."):
            self.recalibrating = True
            self.paused = True
            self.pause_btn.configure(text="▶ Resume")
            self.state_manager.reset_calibration()
            self.session_tracker.reset()

    def export_csv(self):
        success = self.session_tracker.export_csv()
        if success:
            messagebox.showinfo("Export", "Session log berhasil di-export ke session_log.csv")
        else:
            messagebox.showwarning("Export", "Belum ada data untuk di-export.")

    def run(self):
        self.root.mainloop()
