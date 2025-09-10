import os
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import sys

# --- Audio Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- Vosk Model Setup ---
# Please ensure you have downloaded a Vosk model and extracted it.
# Set the correct path below.
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at {MODEL_PATH}")
    print("Please download a model from https://alphacephei.com/vosk/models and set the correct path.")
    sys.exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)

# --- Main Logic ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening for speech...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        # Pass the raw audio data directly to the Vosk recognizer
        if rec.AcceptWaveform(data):
            result = rec.Result()
            # The result is a JSON string, extract the text.
            print("Transcription:", eval(result)['text'])
        else:
            partial_result = rec.PartialResult()
            # Print the partial result, but a real app would update a UI element
            print("Partial:", eval(partial_result)['partial'], end="\r")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
