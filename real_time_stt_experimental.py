import os
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys
import time
import wave

# --- Configuration ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_AGGRESSIVENESS = 3

# --- Vosk Model Setup ---
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

try:
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    sys.exit(1)

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
MIN_NOISE_DURATION_SEC = 3
MIN_NOISE_FRAMES = int(MIN_NOISE_DURATION_SEC * 1000 / FRAME_DURATION_MS)
GAIN_FLOOR = 0.002


# --- Noise Reduction Functions ---
def update_noise_profile(audio_frame):
    """Updates the noise profile using non-speech frames (power spectrum)."""
    global noise_profile, noise_frames_count
    if len(audio_frame) > 0:
        spectrum = fft(audio_frame)
        power_spectrum = np.abs(spectrum[:CHUNK // 2 + 1]) ** 2

        # Weighted averaging
        alpha = 1.0 / (noise_frames_count + 1)
        noise_profile = (1.0 - alpha) * noise_profile + alpha * power_spectrum
        noise_frames_count += 1
        return True
    return False


def wiener_filter_denoise(data):
    """Applies Wiener Filtering for noise reduction."""
    audio_frame = np.frombuffer(data, dtype=np.int16)

    original_len = len(audio_frame)
    expected_len = CHUNK

    if original_len == 0:
        return np.array([], dtype=np.int16)

    if original_len < expected_len:
        padding_needed = expected_len - original_len
        audio_frame = np.pad(audio_frame, (0, padding_needed), 'constant').astype(np.int16)

    spectrum = fft(audio_frame)
    magnitude = np.abs(spectrum)[:CHUNK // 2 + 1]
    phase = np.angle(spectrum)[:CHUNK // 2 + 1]

    # 1. Calculate Noisy Power and Noise Power
    noisy_power = magnitude ** 2
    noise_power = noise_profile

    # 2. Estimate Speech Power
    speech_power_est = np.maximum(noisy_power - noise_power, 0)

    # 3. Calculate Wiener Gain
    noisy_power_safe = noisy_power + 1e-10
    wiener_gain = speech_power_est / noisy_power_safe

    # Apply Gain Floor
    wiener_gain = np.maximum(wiener_gain, GAIN_FLOOR)

    # 4. Apply Gain to the Magnitude Spectrum
    cleaned_magnitude = magnitude * wiener_gain

    # 5. Inverse FFT
    full_cleaned_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    full_cleaned_spectrum[:CHUNK // 2 + 1] = cleaned_magnitude * np.exp(1j * phase)
    full_cleaned_spectrum[CHUNK // 2 + 1:] = np.conjugate(full_cleaned_spectrum[CHUNK // 2 - 1:0:-1])

    cleaned_audio_frame = ifft(full_cleaned_spectrum)

    return np.real(cleaned_audio_frame[:original_len]).astype(np.int16)


# --- Main Logic ---
p = pyaudio.PyAudio()
stream_in = None
stream_out = None

try:
    # --- Device Selection (Simplified for Functionality) ---
    # NOTE: If playback fails, you may need to manually set output_device_index=X
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                       input=True, frames_per_buffer=CHUNK, input_device_index=None)

    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        output=True, frames_per_buffer=CHUNK, output_device_index=None)

    # -----------------------------------------------------------
    # Step 1: Build Initial Noise Profile from Microphone
    # -----------------------------------------------------------
    print(f"Loading Vosk Model... Done.")
    print(f"\n*** Building Initial Noise Profile ({MIN_NOISE_DURATION_SEC} seconds) ***")

    while noise_frames_count < MIN_NOISE_FRAMES:
        data = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data, dtype=np.int16)
        update_noise_profile(input_audio_frame)
        time.sleep(FRAME_DURATION_MS / 1000.0)
        print(f"Sampling silent environment: {noise_frames_count}/{MIN_NOISE_FRAMES} frames...", end='\r')

    print(f"\nInitial noise profile built from {noise_frames_count} frames.")

    print(f"\n*** Starting LIVE Transcription (Wiener Filter Active) ***")
    print("Speak clearly. Transcription will print below. Press Ctrl+C to stop.")

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop
    # -----------------------------------------------------------
    while True:
        data_to_process = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data_to_process, dtype=np.int16)

        # ADAPTIVE NOISE TRACKING
        if not vad.is_speech(data_to_process, RATE):
            update_noise_profile(input_audio_frame)

        # WIENER FILTER DENOISE
        cleaned_audio_spectral = wiener_filter_denoise(data_to_process)

        # VOSK TRANSCRIPTION
        if rec.AcceptWaveform(cleaned_audio_spectral.tobytes()):
            result = eval(rec.Result())
            transcribed_text = result.get('text', '').strip()
            if transcribed_text:
                print(f"\nFinal Transcription: {transcribed_text}")
        else:
            partial_result = eval(rec.PartialResult())
            # Print partial result to the same line
            print(f"Partial Transcription: {partial_result.get('partial')}", end='\r')

        # Play the CLEANED audio back
        stream_out.write(cleaned_audio_spectral.tobytes())


except KeyboardInterrupt:
    print("\n--- Transcription stopped by user (Ctrl+C). ---")
except Exception as e:
    print(f"\nFatal Error during processing: {e}")

finally:
    # --- Cleanup ---
    if stream_in is not None:
        if stream_in.is_active(): stream_in.stop_stream()
        stream_in.close()
    if stream_out is not None:
        if stream_out.is_active(): stream_out.stop_stream()
        stream_out.close()
    p.terminate()

    # latest