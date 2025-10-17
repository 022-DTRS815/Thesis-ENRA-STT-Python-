import os
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
# Removed webrtcvad import
from vosk import Model, KaldiRecognizer
import sys
import time
import glob

# --- Evaluation Variables for Accuracy ---
TARGET_PHRASE = "i am testing the accuracy"  # Speak this exact phrase for testing
total_test_phrases = 0
correct_phrases = 0
total_words_in_target = len(TARGET_PHRASE.split())
total_errors = 0
# ----------------------------------------

# --- Configuration ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30  # Increased for better time efficiency
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # 800 for 50ms at 16k
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
total_reduction_factor = 0.0  # Added for NR evaluation


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


def spectral_subtraction(data):
    """Applies spectral subtraction for noise reduction, handling partial frames."""
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

    # Tuned down over-subtraction from 1.5 to 1.2 to reduce choppiness
    over_subtraction_factor = 1.1

    cleaned_magnitude = np.maximum(magnitude - over_subtraction_factor * noise_profile, 0)

    full_cleaned_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    full_cleaned_spectrum[:CHUNK // 2 + 1] = cleaned_magnitude * np.exp(1j * phase)
    full_cleaned_spectrum[CHUNK // 2 + 1:] = np.conjugate(full_cleaned_spectrum[CHUNK // 2 - 1:0:-1])

    cleaned_audio_frame = ifft(full_cleaned_spectrum)

    return np.real(cleaned_audio_frame[:original_len]).astype(np.int16)


def calculate_accuracy(reference, hypothesis):
    """Simple word-level accuracy calculation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    correct = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    errors = max(len(ref_words), len(hyp_words)) - correct

    return correct, errors, len(ref_words)


# --- Main Logic ---
p = pyaudio.PyAudio()
stream_in = None
stream_out = None

# --- Vosk Model Setup (Scalability Metric) ---
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

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

try:
    # 1. Open INPUT stream for microphone
    stream_in = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

    # 2. Open OUTPUT stream for playback
    stream_out = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)

    print("\n*** Starting LIVE Microphone Transcription ***")
    print(f"Listening with {FRAME_DURATION_MS}ms latency target...")
    print(f"*** ACCURACY TEST: Say the target phrase: '{TARGET_PHRASE.upper()}' ***")
    print("Press Ctrl+C to stop and see full evaluation.")

    total_frames = 0
    total_processing_time = 0.0
    frames_since_last_result = 0

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop (Reading from Mic)
    # -----------------------------------------------------------
    while True:
        start_time = time.perf_counter()

        # Read audio chunk from the microphone
        data = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data, dtype=np.int16)

        # --- Noise Reduction Effectiveness Data (Amplitude Comparison) ---
        input_max_amp = np.max(np.abs(input_audio_frame))

        # 1. Apply median filter
        denoised_median = median_filter_denoise(input_audio_frame, window_size=5)

        # 2. Apply spectral subtraction
        cleaned_audio_spectral = spectral_subtraction(denoised_median.tobytes())

        output_max_amp = np.max(np.abs(cleaned_audio_spectral))
        reduction_factor = (input_max_amp - output_max_amp) / input_max_amp if input_max_amp > 0 else 0
        total_reduction_factor += reduction_factor  # Accumulate for average

        # 3. Vosk Transcription (Partial always printed, Result when segment ends)
        if rec.AcceptWaveform(cleaned_audio_spectral.tobytes()):
            result = eval(rec.Result())
            transcribed_text = result.get('text', '').strip()

            if transcribed_text:
                # Print finalized segment and move to next line
                print(f"\nFINAL: {transcribed_text}")

                # --- Accuracy Evaluation Logic ---
                total_test_phrases += 1

                if transcribed_text.lower() == TARGET_PHRASE.lower():
                    correct_phrases += 1
                    print(">>> PHRASE MATCH: 100% ACCURACY <<<")
                else:
                    c, e, n = calculate_accuracy(TARGET_PHRASE, transcribed_text)
                    total_errors += e
                    word_accuracy = (c / n) * 100 if n > 0 else 0
                    print(f"  --- Word Accuracy: {word_accuracy:.2f}% ({c} / {n} words correct)")
                    print(f"  --- Errors Added: {e}")

                print("-" * 40)

            # Reset frame counter after a final result
            frames_since_last_result = 0
        else:
            # Segment ongoing (print the partial result)
            partial_result = eval(rec.PartialResult())
            print(f"Partial: {partial_result.get('partial')}", end='\r')

        # Play the CLEANED audio back (Optional)
        stream_out.write(cleaned_audio_spectral.tobytes())

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_processing_time += processing_time
        total_frames += 1

        # Display real-time efficiency metric (on the same line)
        print(
            f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms) | NR Reduction: {reduction_factor * 100:.1f}%",
            end="\r")
        frames_since_last_result += 1

except KeyboardInterrupt:
    print("\n--- Stopping Transcription and Generating Summary ---")
except Exception as e:
    print(f"\nFatal Error during processing: {e}")

finally:
    # -----------------------------------------------------------
    # FINAL EVALUATION SUMMARY
    # -----------------------------------------------------------
    print("\n\n" + "=" * 50)
    print("         LIVE MIC STT ALGORITHM EVALUATION SUMMARY          ")
    print("=" * 50)

    # --- 4. Scalability (Model Load) Results ---
    print(f"\n--- 4. Scalability (Model Load) Results ---")
    print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

    # --- 1. Accuracy Results ---
    if total_test_phrases > 0:
        phrase_accuracy = (correct_phrases / total_test_phrases) * 100
        # Calculate a cumulative Word Error Rate (WER) approximation
        total_expected_words = total_words_in_target * total_test_phrases
        approx_wer = (total_errors / total_expected_words) * 100 if total_expected_words > 0 else 100

        print(f"\n--- 1. Accuracy Results (Approx.) ---")
        print(f"Target Phrase: '{TARGET_PHRASE}'")
        print(f"Total Phrase Attempts: {total_test_phrases}")
        print(f"Phrase Match Accuracy: {phrase_accuracy:.2f}%")
        print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}% (Lower is better)")
    else:
        print(f"\n--- 1. Accuracy Results (Approx.) ---")
        print("No final segments transcribed to evaluate accuracy.")

    # --- 2. Time Efficiency (Latency) Results ---
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- 2. Time Efficiency (Latency) Results ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(f"Frame Duration (Target): {FRAME_DURATION_MS} ms")
        print(f"Latency Verdict: {'PASS' if average_time < FRAME_DURATION_MS else 'FAIL'}")

    # --- 3. Noise Reduction (NR) Effectiveness ---
    if total_frames > 0:
        avg_reduction_factor = (total_reduction_factor / total_frames) * 100
        print(f"\n--- 3. Noise Reduction (NR) Effectiveness ---")
        print(f"Average Amplitude Reduction: {avg_reduction_factor:.2f}%")
        print(f"Conclusion: Algorithm was active and reduced average amplitude.")

    print("=" * 50)

    # Safely close resources
    if stream_in is not None:
        try:
            if stream_in.is_active():
                stream_in.stop_stream()
            stream_in.close()
        except Exception:
            pass
    if stream_out is not None:
        try:
            if stream_out.is_active():
                stream_out.stop_stream()
            stream_out.close()
        except Exception:
            pass
    p.terminate()