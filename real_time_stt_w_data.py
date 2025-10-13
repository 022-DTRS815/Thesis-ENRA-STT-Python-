import os
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys
import time

# --- Target Phrase for Accuracy Evaluation (No. 1) ---
# NOTE: Speak this exact sentence when the program is running to test accuracy.
TARGET_PHRASE = "hello i am testing the accuracy"
total_test_phrases = 0
correct_phrases = 0
total_words = len(TARGET_PHRASE.split())
correct_words = 0
total_errors = 0

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
# Calculate the number of samples in a 30ms chunk
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_AGGRESSIVENESS = 3

# --- Vosk Model Setup (Scalability: Model Load Time) ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

# Measure model load time
start_model_load = time.time()
try:
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    sys.exit(1)

end_model_load = time.time()
model_load_time = end_model_load - start_model_load
print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
MIN_NOISE_FRAMES = 100


# --- Functions ---
def median_filter_denoise(audio_frame, window_size=5):
    """Applies a median filter to the audio frame."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number.")

    half_window = window_size // 2
    denoised_frame = np.zeros_like(audio_frame, dtype=np.int16)
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
    # Reconstruct the negative frequency components for the Inverse FFT
    full_cleaned_spectrum[CHUNK // 2 + 1:] = np.conjugate(full_cleaned_spectrum[CHUNK // 2 - 1:0:-1])

    cleaned_audio_frame = ifft(full_cleaned_spectrum)
    return np.real(cleaned_audio_frame).astype(np.int16)


def calculate_accuracy(reference, hypothesis):
    """Simple word-level accuracy calculation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple word-by-word comparison for basic accuracy
    correct = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)

    # Calculate substitution/insertion/deletion errors for a rudimentary WER-like score
    errors = max(len(ref_words), len(hyp_words)) - correct

    return correct, errors, len(ref_words)


# --- Main Logic ---
p = pyaudio.PyAudio()
try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Error initializing PyAudio stream: {e}")
    p.terminate()
    sys.exit(1)

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("Building noise profile... Please be quiet.")

total_frames = 0
total_processing_time = 0.0

try:
    # -----------------------------------------------------------
    # Step 1: Build an initial noise profile
    # -----------------------------------------------------------
    while noise_frames_count < MIN_NOISE_FRAMES:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not vad.is_speech(data, RATE):
                update_noise_profile(data)
        except IOError as e:
            print(f"IOError reading stream (Noise Profile): {e}")
            continue

    print(f"\nNoise profile built. Listening for speech...")
    print(f"*** ACCURACY TEST: Say the target phrase: '{TARGET_PHRASE.upper()}' ***")

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop
    # -----------------------------------------------------------
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        start_time = time.perf_counter()  # <--- Start timing for efficiency

        # Convert data only once for processing
        input_audio_frame = np.frombuffer(data, dtype=np.int16)

        if vad.is_speech(data, RATE):

            # --- Noise Reduction Effectiveness Data (Amplitude Comparison) ---
            input_max_amp = np.max(np.abs(input_audio_frame))

            # 1. Apply median filter
            denoised_median = median_filter_denoise(input_audio_frame, window_size=5)

            # 2. Apply spectral subtraction
            cleaned_audio_spectral = spectral_subtraction(denoised_median.tobytes())

            # Measure Output Signal Max Amplitude
            output_max_amp = np.max(np.abs(cleaned_audio_spectral))

            # Calculate Reduction Factor
            if input_max_amp > 0:
                reduction_factor = (input_max_amp - output_max_amp) / input_max_amp
            else:
                reduction_factor = 0

            # Display real-time NR metric
            print(
                f"NR: Input Max: {input_max_amp}, Output Max: {output_max_amp}, Reduction: {reduction_factor * 100:.1f}%",
                end="\r")

            # 3. Vosk Transcription
            if rec.AcceptWaveform(cleaned_audio_spectral.tobytes()):
                result = rec.Result()
                transcribed_text = eval(result)['text'].strip()

                # Print transcription on a new line to avoid overwriting real-time metrics
                print(f"\nTranscription: {transcribed_text}")

                # --- Accuracy Logic (No. 1) ---
                if transcribed_text:
                    total_test_phrases += 1

                    if transcribed_text == TARGET_PHRASE:
                        correct_phrases += 1
                        print(">>> PHRASE MATCH: 100% ACCURACY <<<")
                    else:
                        # Simple word-level comparison
                        c, e, n = calculate_accuracy(TARGET_PHRASE, transcribed_text)
                        correct_words += c
                        total_errors += e

                        word_accuracy = (c / n) * 100 if n > 0 else 0
                        print(f"  --- Word Accuracy: {word_accuracy:.2f}% ({c} / {n} words correct)")
                        print(f"  --- Total Errors: {e} (Substitutions/Insertions/Deletions)")

                    # Reset recognizer for the next phrase
                    rec.FinalResult()

            else:
                partial_result = rec.PartialResult()
                print("Partial:", eval(partial_result)['partial'], end="\r")

            end_time = time.perf_counter()  # <--- End timing

            # --- Time Efficiency Data ---
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_processing_time += processing_time
            total_frames += 1

            # Display real-time efficiency metric
            print(f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms)", end="\r")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # --- Final Evaluation Results Summary ---

    # 1. Accuracy Summary
    if total_test_phrases > 0:
        phrase_accuracy = (correct_phrases / total_test_phrases) * 100

        # Calculate a cumulative Word Error Rate (WER) approximation
        # Note: This is an approximation and requires more rigorous comparison for true WER.
        approx_wer = (total_errors / (total_words * total_test_phrases)) * 100

        print(f"\n\n--- Accuracy Results (Approx.) ---")
        print(f"Target Phrase: '{TARGET_PHRASE}'")
        print(f"Phrase Match Accuracy: {phrase_accuracy:.2f}% ({correct_phrases} / {total_test_phrases} times)")
        print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}% (Lower is better)")

    # 2. Time Efficiency Summary
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- Time Efficiency (Latency) Results ---")
        print(f"Total Speech Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(
            f"Verdict: {'PASS' if average_time < FRAME_DURATION_MS else 'FAIL'} (Latency is {'acceptable' if average_time < FRAME_DURATION_MS else 'too high'})")

    # 3. Noise Reduction Summary (Average Reduction Factor would be required, but real-time output suffices)
    # The real-time output (Input Max, Output Max, Reduction %) provides the data.

    # 4. Scalability Summary (Model Load Time)
    print(f"\n--- Scalability (Model Load) Results ---")
    print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

    # Close the stream and PyAudio interface
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    p.terminate()
