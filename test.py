from vosk import Model, KaldiRecognizer
import wave

MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15" # Use your fresh model path

# Use the clean test file from step 2
WAVE_FILE = "D:/JetBrains/Projects/ENRA-STT WAV Files/Speech/Sample_2.wav"

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

wf = wave.open(WAVE_FILE, "rb")

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)

print(rec.FinalResult())