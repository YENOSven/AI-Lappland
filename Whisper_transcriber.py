import whisper
import speech_recognition as sr
import numpy as np

class WhisperMic:
    def __init__(self, model="base", device="cpu", mic_index=None):
        self.model_name = model
        self.device = device
        self.model = whisper.load_model(model, device=device)
        self.recognizer = sr.Recognizer()
        self.mic_index = mic_index

    def listen(self):
        recognizer = sr.Recognizer()
        # List all available microphones
        mic_list = sr.Microphone.list_microphone_names()
        # Select a microphone
        mic_index = 0  # Adjust this index based on the output above
        with sr.Microphone(device_index=mic_index) as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


