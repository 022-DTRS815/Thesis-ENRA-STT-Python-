import os
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_AGGRESSIVENESS = 3

# --- Vosk Model Setup ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
MIN_NOISE_FRAMES = 100


# --- Functions ---
def median_filter_denoise(audio_frame, window_size=5):
    """
    Applies a median filter to the audio frame to remove impulsive noise.
    The window_size should be an odd number.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number.")

    half_window = window_size // 2
    denoised_frame = np.zeros_like(audio_frame, dtype=np.int16)

    # Pad the audio frame to handle edges
    padded_frame = np.pad(audio_frame, half_window, mode='edge')

    for i in range(len(audio_frame)):
        window = padded_frame[i:i + window_size]
        denoised_frame[i] = np.median(window)

    return denoised_frame


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

    over_subtraction_factor = 1.5
    cleaned_magnitude = np.maximum(magnitude - over_subtraction_factor * noise_profile, 0)

    full_cleaned_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    full_cleaned_spectrum[:CHUNK // 2 + 1] = cleaned_magnitude * np.exp(1j * phase)
    full_cleaned_spectrum[CHUNK // 2 + 1:] = np.conjugate(full_cleaned_spectrum[CHUNK // 2 - 1:0:-1])

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
            # Apply median filter first to remove impulsive noise
            audio_frame = np.frombuffer(data, dtype=np.int16)
            denoised_median = median_filter_denoise(audio_frame, window_size=5)

            # Then apply spectral subtraction
            cleaned_audio_spectral = spectral_subtraction(denoised_median.tobytes())

            print(f"Max value of audio frame: {np.max(np.abs(cleaned_audio_spectral))}")
            if rec.AcceptWaveform(cleaned_audio_spectral.tobytes()):
                result = rec.Result()
                print("Transcription:", eval(result)['text'])
            else:
                partial_result = rec.PartialResult()
                print("Partial:", eval(partial_result)['partial'], end="\r")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()