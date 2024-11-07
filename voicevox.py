import requests
import json
import sounddevice as sd
import numpy as np
import wave
import soundfile as sf

class VoiceVox:
    def __init__(self, host="127.0.0.1", port="50021", speaker=55):
        self.host = host
        self.port = port
        self.speaker = speaker

    def post_audio_query(self, text: str) -> dict:
        """Creates an audio query for voice synthesis."""
        params = {"text": text, "speaker": self.speaker}
        try:
            response = requests.post(f"http://{self.host}:{self.port}/audio_query", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def post_synthesis(self, query_data: dict) -> bytes:
        """Synthesizes audio from the audio query data."""
        params = {"speaker": self.speaker}
        headers = {"content-type": "application/json"}
        try:
            response = requests.post(f"http://{self.host}:{self.port}/synthesis", data=json.dumps(query_data),
                                     params=params, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException:
            return None

    def save_wavfile(self, wav_data: bytes, filename="output.wav"):
        """Saves the synthesized audio to a WAV file."""
        sample_rate = 24000  # Set the sample rate for VoiceVox output
        wav_array = np.frombuffer(wav_data, dtype=np.int16)
        wav_array = wav_array / 32768.0  # Normalize to float32 range (-1, 1)

        # Save the audio as a WAV file
        with wave.open(filename, 'wb') as file:
            file.setnchannels(1)  # Mono channel
            file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            file.setframerate(sample_rate)  # Sample rate
            file.writeframes((wav_array * 32767).astype(np.int16))  # Convert back to int16 for saving

    def play_wavfile(self, filename="output.wav"):
        """Plays the WAV file using sounddevice."""
        # Use soundfile to read the WAV file
        wav_data, sample_rate = sf.read(filename, dtype='float32')

        # Ensure we have no NaN values or other corrupt data
        if np.any(np.isnan(wav_data)):
            print("Warning: Audio data contains NaN values.")
            return
        
        # Adjust the audio data to ensure no abrupt changes
        wav_data = np.clip(wav_data, -1.0, 1.0)  # Clip the values to the range [-1.0, 1.0]

        # Add a small fade-in (0.1s) to smooth the start of the audio
        fade_in_duration = 0.1  # in seconds
        fade_in_samples = int(sample_rate * fade_in_duration)
        fade_in = np.linspace(0, 1, fade_in_samples)
        wav_data[:fade_in_samples] *= fade_in

        # Play the audio with appropriate buffering settings
        sd.play(wav_data, sample_rate, blocking=True)

    def initialize_audio(self):
        """Initializes the audio system before any playback to avoid popping."""
        silent_audio = np.zeros(1, dtype=np.float32)
        sample_rate = 24000
        sd.play(silent_audio, sample_rate, blocking=True)
        print("Audio system initialized.")

    def text_to_voice(self, text: str):
        """Synthesizes and plays voice for the given text."""
        # Initialize audio system before starting to generate voices
        self.initialize_audio()

        query_data = self.post_audio_query(text)
        if not query_data:
            print("Failed to generate audio query.")
            return

        wav_data = self.post_synthesis(query_data)
        if not wav_data:
            print("Failed to synthesize audio.")
            return

        # Save the synthesized audio to a file
        self.save_wavfile(wav_data)

        # Play the WAV file directly after synthesis and exit
        self.play_wavfile("output.wav")
        print("Finished playing audio.")


