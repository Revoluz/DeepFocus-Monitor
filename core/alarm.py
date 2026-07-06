"""
core/alarm.py - Sistem Alarm Suara (Text-to-Speech)

File ini berisi kelas Alarm yang bertanggung jawab untuk:
1. Memutar pesan peringatan menggunakan Text-to-Speech (pyttsx3)
2. Menjalankan alarm di thread terpisah agar tidak blocking main loop
3. Menyediakan kontrol play/stop untuk alarm

Pesan alarm:
- MICROSLEEP: "Perhatian! Anda terdeteksi microsleep. Segera istirahat!"
- PHONE_ALERT: "Perhatian! Anda bermain HP terlalu lama. Fokus kembali belajar!"

Teknologi:
- pyttsx3: Library Text-to-Speech offline (tidak perlu internet)
- threading: Untuk menjalankan alarm di background thread
"""

import pyttsx3
import threading


# =============================================================================
# PESAN ALARM PER STATUS
# =============================================================================
MESSAGES = {
    'MICROSLEEP': 'Perhatian! Anda terdeteksi microsleep. Segera istirahat!',
    'PHONE_ALERT': 'Perhatian! Anda bermain HP terlalu lama. Fokus kembali belajar!',
}


class Alarm:
    """
    Kelas untuk sistem alarm suara menggunakan Text-to-Speech.

    Cara kerja:
    1. play(alarm_type) → spawn thread baru untuk menjalankan _speak()
    2. _speak() → loop TTS message sampai di-stop
    3. stop() → set stop event dan cleanup thread

    Thread safety:
    - Alarm berjalan di thread terpisah agar tidak blocking main loop
    - Stop event digunakan untuk signal thread agar berhenti dengan aman
    """

    def __init__(self):
        """Inisialisasi Alarm dengan threading support."""
        self._playing = False  # Flag apakah alarm sedang aktif
        self._stop_event = threading.Event()  # Event untuk signal stop
        self._thread = None  # Thread yang menjalankan alarm
        self._engine = None  # pyttsx3 engine (lazy init)

    def _init_engine(self):
        """
        Inisialisasi pyttsx3 engine dengan pengaturan suara.

        Dipanggil secara lazy (hanya saat pertama kali play dipanggil)
        untuk menghindari overhead saat startup.

        Settings:
        - rate: 150 words per minute (lebih lambat agar jelas)
        - volume: 1.0 (100%)
        """
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 150)
            self._engine.setProperty('volume', 1.0)

    def play(self, alarm_type='MICROSLEEP'):
        """
        Memulai alarm di thread terpisah.

        Args:
            alarm_type: Tipe alarm ('MICROSLEEP' atau 'PHONE_ALERT')
                        Menentukan pesan yang akan diucapkan

        Catatan:
            Jika alarm sudah berjalan, fungsi ini tidak melakukan apa-apa
            untuk menghindari multiple thread yang overlap.
        """
        if self._playing:
            return  # Alarm sudah berjalan, abaikan

        self._stop_event.clear()  # Reset stop event
        self._playing = True
        # Spawn thread baru untuk menjalankan alarm
        self._thread = threading.Thread(target=self._speak, args=(alarm_type,), daemon=True)
        self._thread.start()

    def _speak(self, alarm_type):
        """
        Loop Text-to-Speech message sampai di-stop.

        Args:
            alarm_type: Tipe alarm untuk menentukan pesan

        Cara kerja:
        1. Ambil pesan dari dictionary MESSAGES
        2. Loop: ucapkan pesan → tunggu 2 detik → cek stop event
        3. Berhenti jika stop event di-set
        """
        self._init_engine()
        message = MESSAGES.get(alarm_type, MESSAGES['MICROSLEEP'])

        # Loop sampai di-stop
        while not self._stop_event.is_set():
            self._engine.say(message)
            self._engine.runAndWait()
            # Tunggu 2 detik sebelum ulang, tapi bisa di-interrupt oleh stop
            self._stop_event.wait(2)

    def stop(self):
        """
        Menghentikan alarm dan cleanup thread.

        Cara kerja:
        1. Set stop event untuk signal thread agar berhenti
        2. Tunggu thread selesai (max 3 detik)
        3. Stop engine TTS
        """
        if not self._playing:
            return  # Alarm tidak berjalan, abaikan

        self._stop_event.set()  # Signal thread untuk berhenti
        self._playing = False
        if self._thread:
            self._thread.join(timeout=3)  # Tunggu thread selesai

        if self._engine:
            self._engine.stop()  # Stop engine TTS
