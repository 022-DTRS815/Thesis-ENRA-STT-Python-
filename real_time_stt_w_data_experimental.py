import os
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys
import time
import glob
import wave  # Kept for potential future file I/O but not strictly used in mic mode

# --- Configuration (Set these paths) ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"

# --- Evaluation Variables ---
# NOTE: You MUST speak this phrase during the session for accuracy testing
TARGET_PHRASE = "i am testing the accuracy"
total_test_phrases = 0  # Starts at 0, counts successful phrase attempts
total_words_in_target = len(TARGET_PHRASE.split())
total_errors = 0
total_frames = 0
total_processing_time = 0.0
total_reduction_factor = 0.0
full_partial_transcription = ""

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_AGGRESSIVENESS = 3

# --- Vosk Model Setup (Scalability: Model Load Time) ---
if not os.path.exists(MODEL_PATH):
    print("Please download a Vosk model and set the correct path.")
    sys.exit(1)

start_model_load = time.time()
try:
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    sys.exit(1)

end_model_load = time.time()
model_load_time = end_model_load - start_model_load
print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

# --- Real-Time Noise Reduction Variables ---
noise_profile = np.zeros(CHUNK // 2 + 1)
noise_frames_count = 0
MIN_NOISE_DURATION_SEC = 3  # Time to listen for initial profile
MIN_NOISE_FRAMES = int(MIN_NOISE_DURATION_SEC * 1000 / FRAME_DURATION_MS)
GAIN_FLOOR = 0.002


# --- Functions ---
def update_noise_profile(audio_frame):
    """Updates the noise profile using non-speech frames (power spectrum)."""
    global noise_profile, noise_frames_count
    if len(audio_frame) > 0:
        spectrum = fft(audio_frame)
        power_spectrum = np.abs(spectrum[:CHUNK // 2 + 1]) ** 2

        # Weighted averaging for smoother adaptation
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

    # 2. Estimate Speech Power (using Power Spectral Subtraction)
    speech_power_est = np.maximum(noisy_power - noise_power, 0)

    # 3. Calculate Wiener Gain
    noisy_power_safe = noisy_power + 1e-10

    wiener_gain = speech_power_est / noisy_power_safe

    # Apply Gain Floor to prevent gain from being exactly zero (reduces musical noise)
    wiener_gain = np.maximum(wiener_gain, GAIN_FLOOR)

    # 4. Apply Gain to the Magnitude Spectrum
    cleaned_magnitude = magnitude * wiener_gain

    # 5. Inverse FFT using Cleaned Magnitude and Noisy Phase
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

try:
    # 1. Open INPUT stream for microphone
    stream_in = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       input_device_index=None)

    # 2. Open OUTPUT stream for playback
    stream_out = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)

    # -----------------------------------------------------------
    # Step 1: Build Initial Noise Profile from Microphone
    # -----------------------------------------------------------
    print(f"\n*** Building Initial Noise Profile ({MIN_NOISE_DURATION_SEC} seconds) ***")

    while noise_frames_count < MIN_NOISE_FRAMES:
        # Read raw mic data
        data = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data, dtype=np.int16)

        # Pass raw audio to update profile
        update_noise_profile(input_audio_frame)

        time.sleep(FRAME_DURATION_MS / 1000.0)
        print(f"Sampling silent environment: {noise_frames_count}/{MIN_NOISE_FRAMES} frames...", end='\r')

    print(f"\nInitial noise profile built from {noise_frames_count} frames.")

    print(f"\n*** Starting LIVE Microphone Transcription ***")
    print(f"*** ACCURACY TEST: Speak the target phrase: '{TARGET_PHRASE.upper()}' ***")
    print("Press Ctrl+C to stop and see full evaluation.")

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop (Reading from Mic)
    # -----------------------------------------------------------
    while True:
        start_time = time.perf_counter()

        # Read audio chunk from the microphone
        data_to_process = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data_to_process, dtype=np.int16)

        # --- ADAPTIVE NOISE TRACKING ---
        if not vad.is_speech(data_to_process, RATE):
            # If VAD detects silence, update the noise profile with the current frame
            update_noise_profile(input_audio_frame)

        # --- Noise Reduction Effectiveness Data (Amplitude Comparison) ---
        input_max_amp = np.max(np.abs(input_audio_frame))

        # 1. Wiener Filter Denoise
        cleaned_audio = wiener_filter_denoise(data_to_process)

        output_max_amp = np.max(np.abs(cleaned_audio))
        reduction_factor = (input_max_amp - output_max_amp) / input_max_amp if input_max_amp > 0 else 0
        total_reduction_factor += reduction_factor

        # 2. Vosk Transcription (Partial always printed, Result when segment ends)
        if rec.AcceptWaveform(cleaned_audio.tobytes()):
            result = eval(rec.Result())
            transcribed_text = result.get('text', '').strip()

            if transcribed_text:
                print(f"\nFINAL: {transcribed_text}")

                # --- Accuracy Evaluation Logic ---
                total_test_phrases += 1

                c, e, n = calculate_accuracy(TARGET_PHRASE, transcribed_text)
                total_errors += e
                word_accuracy = (c / n) * 100 if n > 0 else 0
                print(f"  --- Word Accuracy: {word_accuracy:.2f}% ({c} / {n} words correct)")
                print(f"  --- Errors Added: {e}")
                print("-" * 40)

                # We save the last final result for the end summary
                full_partial_transcription = transcribed_text

        else:
            # Segment ongoing (print the partial result)
            partial_result = eval(rec.PartialResult())
            print(f"Partial: {partial_result.get('partial')}", end='\r')

        # Play the CLEANED audio back (Optional)
        stream_out.write(cleaned_audio.tobytes())

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        total_processing_time += processing_time
        total_frames += 1

        # Display real-time efficiency metric (on the same line)
        print(
            f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms) | NR Reduction: {reduction_factor * 100:.1f}%",
            end="\r")


except KeyboardInterrupt:
    print("\n--- Stopping Transcription and Generating Summary ---")
except Exception as e:
    print(f"\nFatal Error during processing: {e}")

finally:
    # -----------------------------------------------------------
    # FINAL EVALUATION SUMMARY
    # -----------------------------------------------------------

    # 1. Try to get the official final result (any leftover buffer)
    final_transcription = ""
    try:
        if 'rec' in locals():
            final_result_json = rec.FinalResult()
            # If Vosk's final result is empty, use the last successfully transcribed segment
            final_transcription = eval(final_result_json)['text'].strip() or full_partial_transcription
    except Exception:
        final_transcription = full_partial_transcription

    if not final_transcription:
        final_transcription = "ERROR (No transcription data available)"

    print("\n" * 2)
    print("=================================================================")
    print("         LIVE MICROPHONE (WIENER) EVALUATION SUMMARY         ")
    print("=================================================================")

    print(f"\n--- 4. Scalability (Model Load) Results ---")
    print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

    # 1. Accuracy Summary
    c, e, n = calculate_accuracy(TARGET_PHRASE, final_transcription)
    total_expected_words = total_words_in_target * total_test_phrases if total_test_phrases > 0 else total_words_in_target
    approx_wer = (total_errors / total_expected_words) * 100 if total_expected_words > 0 else 100

    print(f"\n--- 1. Accuracy Results (Live Test) ---")
    print(f"Target Phrase: '{TARGET_PHRASE}'")
    print(f"Total Phrase Attempts Logged: {total_test_phrases}")
    # Show the last captured transcription
    print(f"Last Full Transcription: '{final_transcription}'")
    # For live mic, we sum errors across all attempts
    print(f"Total Errors Recorded: {total_errors}")
    print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}% (Lower is better)")

    # 2. Time Efficiency Summary
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- 2. Time Efficiency (Latency) Results ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(f"Max Frame Duration (Target): {FRAME_DURATION_MS} to {FRAME_DURATION_MS + 10} ms")
        print(f"Latency Verdict: {'PASS' if average_time < FRAME_DURATION_MS + 10 else 'FAIL'}")

    # 3. Noise Reduction Summary
    if total_frames > 0:
        avg_reduction_factor = (total_reduction_factor / total_frames) * 100
        print(f"\n--- 3. Noise Reduction (NR) Effectiveness ---")
        print(f"Algorithm Used: Wiener Filter + Adaptive Noise Tracking")
        print(f"Noise Profile Frames: {noise_frames_count}")
        print(f"Average Amplitude Reduction: {avg_reduction_factor:.2f}%")

    print("=================================================================")

    # Safely close resources
    if stream_in is not None:
        if stream_in.is_active():
            stream_in.stop_stream()
        stream_in.close()
    if stream_out is not None:
        if stream_out.is_active():
            stream_out.stop_stream()
        stream_out.close()
    p.terminate()

    # latest