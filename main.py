from Whisper_transcriber import WhisperMic
from Llama_3 import LlamaChatbot
from voicevox import VoiceVox
from translate import Translator
from EdgingTTS import Speak

if __name__ == "__main__":
    mic = WhisperMic()
    llama = LlamaChatbot(r"C:\Users\alanl\LapplandBot\results")
    #edging = Speak()
    translator = Translator()
    voicevox = VoiceVox()
    llama.reset_conversation()

    while True:
        prompt = mic.listen()
        #prompt = input("user: ")
        if prompt == "close":
            break
        elif (prompt == None):
            prompt = ""
        else:
            response = llama.generate_response(prompt)
            print("Lappland:"+ response)
            #edging.speak(response)
            response = translator.translate(response).replace(" ", "")
            voicevox.text_to_voice(response)
