import os
import pygame


class Speak:
    def __init__(self, voice = "ja-JP-NanamiNeural", volume = "+100%", rate = "-10%", pitch = "-100Hz"):
        self.voice = voice
        self.volume = volume
        self.rate = rate
        self.pitch = pitch

    def speak(self, text: str):
        command = f'edge-tts --pitch="{self.pitch}" --rate="{self.rate}" --volume="{self.volume}" --voice "{self.voice}" --text "{text}" --write-media "data.mp3"'
        os.system(command)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("data.mp3")

        try:
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except Exception as e:
            print(e)
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()




