import pyttsx3
import threading
import queue

class TextToSpeech:
    def __init__ (self):
        # initilaize the pyttsx3
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty("voices")
        self.engine.setProperty('voice', self.voices[0].id)
        self.speech_queue = queue.Queue()
        threading.Thread(target=self.process_speech, daemon=True).start()

    def process_speech(self):
        text = self.speech_queue.get()
        self.engine.say(text)
        self.engine.runAndWait()

    def say(self, text):
        self.speech_queue.put(text)

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.say("Hey there, what's good brother")
    tts.process_speech()