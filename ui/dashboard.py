from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QMessageBox,
                              QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPainter, QPen, QColor
import time


class EARChart(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(180)
        self.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.history = []
        self.threshold = None

    def update_data(self, history, threshold):
        self.history = history[-100:]
        self.threshold = threshold
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.history) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        padding = 30

        min_val = min(self.history) * 0.9
        max_val = max(self.history) * 1.1
        if max_val == min_val:
            max_val = min_val + 0.01

        painter.drawText(5, 15, "EAR Live")

        if self.threshold:
            y_thresh = padding + (1 - (self.threshold - min_val) / (max_val - min_val)) * (h - 2 * padding)
            pen = QPen(QColor("red"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(padding, int(y_thresh), w - padding, int(y_thresh))
            painter.setPen(QColor("red"))
            painter.drawText(w - 70, int(y_thresh) - 5, "Threshold")

        points = []
        for i, val in enumerate(self.history):
            x = padding + (i / (len(self.history) - 1)) * (w - 2 * padding)
            y = padding + (1 - (val - min_val) / (max_val - min_val)) * (h - 2 * padding)
            points.append((int(x), int(y)))

        pen = QPen(QColor("blue"), 2)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])


class SimpleDashboard(QMainWindow):
    def __init__(self, session_tracker, state_manager, alarm):
        super().__init__()
        self.session_tracker = session_tracker
        self.state_manager = state_manager
        self.alarm = alarm

        self.paused = False
        self.recalibrating = False

        self.setWindowTitle("DeepFocus Dashboard")
        self.setFixedSize(650, 550)

        self._build_ui()
        self._start_timer()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.timer_label = QLabel("Session: 00:00:00")
        self.timer_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timer_label)

        stats_row = QHBoxLayout()
        self.focus_label = QLabel("Focus Score: 100%")
        self.focus_label.setFont(QFont("Arial", 14))
        stats_row.addWidget(self.focus_label)

        self.status_label = QLabel("Status: FOKUS")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_row.addWidget(self.status_label)
        layout.addLayout(stats_row)

        detail_row = QHBoxLayout()
        self.ear_label = QLabel("EAR: -")
        self.ear_label.setFont(QFont("Arial", 11))
        detail_row.addWidget(self.ear_label)

        self.lip_label = QLabel("LIP: -")
        self.lip_label.setFont(QFont("Arial", 11))
        detail_row.addWidget(self.lip_label)

        self.yaw_label = QLabel("YAW: -")
        self.yaw_label.setFont(QFont("Arial", 11))
        detail_row.addWidget(self.yaw_label)

        self.count_label = QLabel("Microsleep: 0 | Yawning: 0")
        self.count_label.setFont(QFont("Arial", 11))
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        detail_row.addWidget(self.count_label)
        layout.addLayout(detail_row)

        self.chart = EARChart()
        layout.addWidget(self.chart)

        self.break_label = QLabel("")
        self.break_label.setFont(QFont("Arial", 12))
        self.break_label.setStyleSheet("color: red;")
        self.break_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.break_label.setWordWrap(True)
        layout.addWidget(self.break_label)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setFont(QFont("Arial", 12))
        self.pause_btn.setFixedHeight(40)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.pause_btn)

        self.recalib_btn = QPushButton("🔄 Kalibrasi Ulang")
        self.recalib_btn.setFont(QFont("Arial", 12))
        self.recalib_btn.setFixedHeight(40)
        self.recalib_btn.clicked.connect(self.start_recalibration)
        btn_row.addWidget(self.recalib_btn)

        self.export_btn = QPushButton("📊 Export CSV")
        self.export_btn.setFont(QFont("Arial", 12))
        self.export_btn.setFixedHeight(40)
        self.export_btn.clicked.connect(self.export_csv)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_timer)
        self.timer.start(1000)

    def _update_timer(self):
        self.timer_label.setText(f"Session: {self.session_tracker.get_session_duration()}")

    def update(self, status, ear, lip_distance, head_pose):
        focus = self.session_tracker.get_focus_score()
        self.focus_label.setText(f"Focus Score: {focus:.1f}%")

        status_colors = {
            'NORMAL': 'green',
            'YAWNING': 'orange',
            'DROWSY': 'orange',
            'MICROSLEEP': 'red',
            'PHONE_ALERT': 'red',
            'FACE_LOST': 'gray',
        }
        status_labels = {
            'NORMAL': 'FOKUS',
            'YAWNING': 'MENGUAP',
            'DROWSY': 'MENGANTUK',
            'MICROSLEEP': 'MICROSLEEP!',
            'PHONE_ALERT': 'HP TERDETEKSI!',
            'FACE_LOST': 'WAJAH HILANG',
        }
        color = status_colors.get(status, 'black')
        label = status_labels.get(status, status)
        self.status_label.setText(f"Status: {label}")
        self.status_label.setStyleSheet(f"color: {color};")

        if ear is not None:
            self.ear_label.setText(f"EAR: {ear:.4f}")
        if lip_distance is not None:
            self.lip_label.setText(f"LIP: {lip_distance:.4f}")
        if head_pose:
            yaw = head_pose.get('yaw', 0)
            self.yaw_label.setText(f"YAW: {yaw:.1f}°")

        self.count_label.setText(f"Microsleep: {self.session_tracker.microsleep_total}x | Yawning: {self.session_tracker.yawning_total}x")

        self.chart.update_data(self.session_tracker.ear_history,
                               self.state_manager.calibration.ear_threshold)

        break_msg = self.state_manager.check_break_reminder(self.session_tracker)
        if break_msg:
            self.break_label.setText(f"⚠️ {break_msg}")
            self.alarm.play('MICROSLEEP')
        else:
            self.break_label.setText("")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("▶ Resume" if self.paused else "⏸ Pause")

    def start_recalibration(self):
        reply = QMessageBox.question(self, "Kalibrasi Ulang",
                                     "Yakin ingin kalibrasi ulang? Monitoring akan pause selama 5 detik.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.recalibrating = True
            self.paused = True
            self.pause_btn.setText("▶ Resume")
            self.state_manager.reset_calibration()
            self.session_tracker.reset()

    def export_csv(self):
        success = self.session_tracker.export_csv()
        if success:
            QMessageBox.information(self, "Export", "Session log berhasil di-export ke session_log.csv")
        else:
            QMessageBox.warning(self, "Export", "Belum ada data untuk di-export.")
