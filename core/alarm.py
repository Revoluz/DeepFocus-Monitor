import pyttsx3
import threading


MESSAGES = {
    'MICROSLEEP': 'Perhatian! Anda terdeteksi microsleep. Segera istirahat!',
    'PHONE_ALERT': 'Perhatian! Anda bermain HP terlalu lama. Fokus kembali belajar!',
}


class Alarm:
    def __init__(self):
        self._playing = False
        self._stop_event = threading.Event()
        self._thread = None
        self._engine = None

    def _init_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 150)
            self._engine.setProperty('volume', 1.0)

    def play(self, alarm_type='MICROSLEEP'):
        if self._playing:
            return

        self._stop_event.clear()
        self._playing = True
        self._thread = threading.Thread(target=self._speak, args=(alarm_type,), daemon=True)
        self._thread.start()

    def _speak(self, alarm_type):
        self._init_engine()
        message = MESSAGES.get(alarm_type, MESSAGES['MICROSLEEP'])

        while not self._stop_event.is_set():
            self._engine.say(message)
            self._engine.runAndWait()
            self._stop_event.wait(2)

    def stop(self):
        if not self._playing:
            return

        self._stop_event.set()
        self._playing = False
        if self._thread:
            self._thread.join(timeout=3)

        if self._engine:
            self._engine.stop()
