import os

import pyaudio
import numpy as np
from scipy.fft import fft, ifft
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30  # New constant
CHUNK = int(RATE * FRAME_DURATION_MS / 1000) # Correctly calculates CHUNK for 20ms
FORMAT = pyaudio.paInt16
CHANNELS = 1

VAD_AGGRESSIVENESS = 3  # 0-3, 3 is most aggressive

# --- Vosk Model Setup ---
# Download a model from https://alphacephei.com/vosk/models
# and extract it. Set the correct path below.
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
MIN_NOISE_FRAMES = 100  # Minimum frames to build initial noise profile


# --- Functions ---
def update_noise_profile(data):
    """Updates the noise profile using non-speech frames."""
    global noise_profile, noise_frames_count
    audio_frame = np.frombuffer(data, dtype=np.int16)
    if len(audio_frame) > 0:
        spectrum = fft(audio_frame)
        magnitude = np.abs(spectrum)[:CHUNK // 2 + 1]
        noise_profile = (noise_profile * noise_frames_count + magnitude) / (noise_frames_count + 1)
        noise_frames_count += 1
        return True
    return False


def spectral_subtraction(data):
    """Applies spectral subtraction for noise reduction."""
    audio_frame = np.frombuffer(data, dtype=np.int16)
    if len(audio_frame) == 0:
        return np.array([], dtype=np.int16)

    spectrum = fft(audio_frame)
    magnitude = np.abs(spectrum)[:CHUNK // 2 + 1]
    phase = np.angle(spectrum)[:CHUNK // 2 + 1]

    # Subtract the noise magnitude. `np.maximum` ensures no negative values.
    cleaned_magnitude = np.maximum(magnitude - noise_profile, 0)

    # Reconstruct the full spectrum
    full_cleaned_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    full_cleaned_spectrum[:CHUNK // 2 + 1] = cleaned_magnitude * np.exp(1j * phase)
    # The spectrum of a real signal is symmetric, so we fill the other half
    full_cleaned_spectrum[CHUNK // 2 + 1:] = np.conjugate(full_cleaned_spectrum[CHUNK // 2 - 1:0:-1])

    # Perform inverse FFT to get the time-domain signal
    cleaned_audio_frame = ifft(full_cleaned_spectrum)
    return np.real(cleaned_audio_frame).astype(np.int16)


# --- Main Logic ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("Building noise profile... Please be quiet.")

# Step 1: Build an initial noise profile
while noise_frames_count < MIN_NOISE_FRAMES:
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if not vad.is_speech(data, RATE):
            update_noise_profile(data)
    except IOError as e:
        print(f"Error reading from stream: {e}")
        continue

print("Noise profile built. Listening for speech...")

# Step 2: Main real-time processing loop
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if vad.is_speech(data, RATE):
            cleaned_audio = spectral_subtraction(data)

            if rec.AcceptWaveform(cleaned_audio.tobytes()):
                result = rec.Result()
                # The result is a JSON string, extract the text.
                print("Transcription:", eval(result)['text'])

except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
