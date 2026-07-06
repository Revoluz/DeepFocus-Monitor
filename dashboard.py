"""
dashboard.py - Dashboard GUI Monitoring (tkinter)

File terpisah untuk dashboard GUI monitoring real-time.
Menampilkan: session timer, focus score, live metrics, EAR chart, dan tombol kontrol.
"""

import tkinter as tk
from tkinter import messagebox


class SimpleDashboard:
    """Dashboard GUI sederhana untuk monitoring DeepFocus."""

    def __init__(self, tracker, state_mgr, alarm):
        self.tracker = tracker
        self.state_mgr = state_mgr
        self.alarm = alarm
        self.paused = False
        self.recalibrating = False

        self.root = tk.Tk()
        self.root.title("DeepFocus Dashboard")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        self.timer_lbl = tk.Label(self.root, text="Session: 00:00:00", font=("Arial", 16, "bold"))
        self.timer_lbl.pack(pady=10)

        stats = tk.Frame(self.root)
        stats.pack(fill=tk.X, padx=20, pady=5)
        self.focus_lbl = tk.Label(stats, text="Focus Score: 100%", font=("Arial", 14))
        self.focus_lbl.pack(side=tk.LEFT, padx=10)
        self.status_lbl = tk.Label(stats, text="Status: FOKUS", font=("Arial", 14), fg="green")
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        detail = tk.Frame(self.root)
        detail.pack(fill=tk.X, padx=20, pady=5)
        self.ear_lbl = tk.Label(detail, text="EAR: -", font=("Arial", 11))
        self.ear_lbl.pack(side=tk.LEFT, padx=10)
        self.lip_lbl = tk.Label(detail, text="LIP: -", font=("Arial", 11))
        self.lip_lbl.pack(side=tk.LEFT, padx=10)
        self.yaw_lbl = tk.Label(detail, text="YAW: -", font=("Arial", 11))
        self.yaw_lbl.pack(side=tk.LEFT, padx=10)
        self.cnt_lbl = tk.Label(detail, text="Microsleep: 0 | Yawning: 0", font=("Arial", 11))
        self.cnt_lbl.pack(side=tk.RIGHT, padx=10)

        self.chart = tk.Canvas(self.root, bg="white", height=200, relief=tk.SUNKEN, bd=2)
        self.chart.pack(fill=tk.X, padx=20, pady=10)

        self.break_lbl = tk.Label(self.root, text="", font=("Arial", 12), fg="red", wraplength=550)
        self.break_lbl.pack(pady=5)

        btns = tk.Frame(self.root)
        btns.pack(pady=10)
        self.pause_btn = tk.Button(btns, text="⏸ Pause", font=("Arial", 12), command=self.toggle_pause, width=10)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        self.recalib_btn = tk.Button(btns, text="🔄 Kalibrasi Ulang", font=("Arial", 12), command=self.start_recalibration, width=15)
        self.recalib_btn.pack(side=tk.LEFT, padx=5)

    def update(self, status, ear, lip, head_pose):
        self.timer_lbl.configure(text=f"Session: {self.tracker.duration()}")
        self.focus_lbl.configure(text=f"Focus Score: {self.tracker.focus_score():.1f}%")

        colors = {'NORMAL': 'green', 'YAWNING': 'orange', 'DROWSY': 'orange', 'DISTRACTED': 'darkorange', 'MICROSLEEP': 'red', 'PHONE_ALERT': 'red', 'FACE_LOST': 'gray'}
        labels = {'NORMAL': 'FOKUS', 'YAWNING': 'MENGUAP', 'DROWSY': 'MENGANTUK', 'DISTRACTED': 'TERDISTRASI: HP', 'MICROSLEEP': 'MICROSLEEP!', 'PHONE_ALERT': 'HP TERLALU LAMA!', 'FACE_LOST': 'WAJAH HILANG'}
        self.status_lbl.configure(text=f"Status: {labels.get(status, status)}", fg=colors.get(status, 'black'))

        if ear is not None:
            self.ear_lbl.configure(text=f"EAR: {ear:.4f}")
        if lip is not None:
            self.lip_lbl.configure(text=f"LIP: {lip:.4f}")
        if head_pose:
            self.yaw_lbl.configure(text=f"YAW: {head_pose.get('yaw', 0):.1f}°")

        self.cnt_lbl.configure(text=f"Microsleep: {self.tracker.microsleep_seconds():.1f}s | Yawning: {self.tracker.yawning_seconds():.1f}s")
        self._draw_chart()

        msg = self.state_mgr.check_break(self.tracker)
        if msg:
            self.break_lbl.configure(text=f"⚠️ {msg}")
            self.alarm.play('MICROSLEEP')
        else:
            self.break_lbl.configure(text="")

    def _draw_chart(self):
        self.chart.delete("all")
        hist = self.tracker.ear_hist[-100:]
        if len(hist) < 2:
            return
        w, h, pad = 560, 180, 20
        mn, mx = min(hist) * 0.9, max(hist) * 1.1
        if mx == mn:
            mx = mn + 0.01
        self.chart.create_text(30, 10, text="EAR Live", font=("Arial", 10), anchor=tk.NW)
        thr = self.state_mgr.calibration.ear_threshold
        if thr:
            yt = pad + (1 - (thr - mn) / (mx - mn)) * (h - 2 * pad)
            self.chart.create_line(pad, yt, w - pad, yt, fill="red", dash=(4, 4))
            self.chart.create_text(w - pad, yt - 5, text="Threshold", fill="red", font=("Arial", 8), anchor=tk.E)
        pts = [(pad + (i / (len(hist) - 1)) * (w - 2 * pad), pad + (1 - (v - mn) / (mx - mn)) * (h - 2 * pad)) for i, v in enumerate(hist)]
        for i in range(len(pts) - 1):
            self.chart.create_line(pts[i], pts[i + 1], fill="blue", width=2)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="▶ Resume" if self.paused else "⏸ Pause")

    def start_recalibration(self):
        if messagebox.askyesno("Kalibrasi Ulang", "Yakin ingin kalibrasi ulang? Monitoring akan pause selama 5 detik."):
            self.recalibrating = True
            self.paused = True
            self.pause_btn.configure(text="▶ Resume")
            self.state_mgr.reset_calibration()
            self.tracker.reset()

    def run(self):
        self.root.mainloop()
