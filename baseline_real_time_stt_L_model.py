import os
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import sys

# --- Audio Parameters ---
RATE = 16000
# Increased CHUNK size for a larger buffer, which helps prevent overflow
# on systems with higher CPU load.
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- Vosk Model Setup ---
# Please ensure you have downloaded a Vosk model and extracted it.
# Set the correct path below. This version is for the larger model.
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-en-us-0.22"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at {MODEL_PATH}")
    print("Please download a model from https://alphacephei.com/vosk/models and set the correct path.")
    sys.exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)

# --- Main Logic ---
p = pyaudio.PyAudio()
# The frames_per_buffer is set to a larger chunk size for robustness.
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Using Vosk model:", os.path.basename(MODEL_PATH))
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

except OSError as e:
    # A specific error handler for the audio stream.
    # This is often caused by a buffer overflow from a heavy computational load.
    print(f"\nCritical Audio Error: {e}")
    print("This is likely caused by an audio device issue or a buffer overflow due to high CPU usage.")
    print("Consider closing other applications or reducing the chunk size if this persists.")
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # This check ensures we only try to close the stream if it's still open and active.
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    p.terminate()