import os
import pyaudio
import numpy as np
# from scipy.fft import fft, ifft # REMOVED: No FFT/IFFT needed
import webrtcvad
from vosk import Model, KaldiRecognizer
import sys
import time
import glob
import wave

# --- Configuration (Set these paths) ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"

# --- Evaluation Variables ---
# NOTE: You MUST speak this phrase during the session for accuracy testing
TARGET_PHRASE = "i am testing the accuracy"
total_test_phrases = 0
total_words_in_target = len(TARGET_PHRASE.split())
total_errors = 0
total_frames = 0
total_processing_time = 0.0
# NR metrics will be 0 or N/A
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

# --- Noise Reduction Variables (Disabled for Baseline) ---
noise_frames_count = 0
MIN_NOISE_DURATION_SEC = 3
MIN_NOISE_FRAMES = int(MIN_NOISE_DURATION_SEC * 1000 / FRAME_DURATION_MS)


# --- Functions (Only keep accuracy) ---
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
    # Step 1: Baseline Setup (SKIP NOISE PROFILE BUILDING)
    # -----------------------------------------------------------
    print("\n*** Skipping Noise Profile Building for BASELINE ***")

    # Simulate the initial listen time for comparison purposes, but discard the data
    print(f"Listening silently for initial setup ({MIN_NOISE_DURATION_SEC} seconds)...")
    for i in range(MIN_NOISE_FRAMES):
        stream_in.read(CHUNK, exception_on_overflow=False)
        time.sleep(FRAME_DURATION_MS / 1000.0)
        print(f"Setup: {i + 1}/{MIN_NOISE_FRAMES} frames...", end='\r')

    print("\nInitial baseline setup complete. Noise profile is empty.")

    print(f"\n*** Starting BASELINE LIVE Microphone Transcription (NO NR) ***")
    print(f"*** ACCURACY TEST: Speak the target phrase: '{TARGET_PHRASE.upper()}' ***")
    print("Press Ctrl+C to stop and see full evaluation.")

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop (Reading from Mic)
    # -----------------------------------------------------------
    while True:
        start_time = time.perf_counter()

        # Read raw audio chunk from the microphone
        data_to_process = stream_in.read(CHUNK, exception_on_overflow=False)
        input_audio_frame = np.frombuffer(data_to_process, dtype=np.int16)

        # --- NOISE REDUCTION & TRACKING SKIPPED ---
        # The audio frame to process is the raw mic input
        cleaned_audio_spectral_tobytes = data_to_process

        # --- Noise Reduction Effectiveness Data (DISABLED) ---
        input_max_amp = np.max(np.abs(input_audio_frame))
        output_max_amp = input_max_amp
        reduction_factor = 0.0

        # 2. Vosk Transcription (Partial always printed, Result when segment ends)
        if rec.AcceptWaveform(cleaned_audio_spectral_tobytes):
            result = eval(rec.Result())
            transcribed_text = result.get('text', '').strip()

            if transcribed_text:
                print(f"\nFINAL: {transcribed_text}")

                # --- Accuracy Evaluation Logic ---
                total_test_phrases += 1

                c, e, n = calculate_accuracy(TARGET_PHRASE, transcribed_text)
                total_errors += e
                word_accuracy = (c / n) * 100 if n > 0 else 0

                # --- FIX APPLIED: Changed ':.2ff' to ':.2f' ---
                print(f"  --- Word Accuracy: {word_accuracy:.2f}% ({c} / {n} words correct)")
                print(f"  --- Errors Added: {e}")
                print("-" * 40)

                full_partial_transcription = transcribed_text

        else:
            partial_result = eval(rec.PartialResult())
            print(f"Partial: {partial_result.get('partial')}", end='\r')

        # Play the RAW (noisy) audio back
        stream_out.write(cleaned_audio_spectral_tobytes)

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        total_processing_time += processing_time
        total_frames += 1

        # Display real-time efficiency metric (on the same line)
        print(f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms)", end="\r")


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
            final_transcription = eval(final_result_json)['text'].strip() or full_partial_transcription
    except Exception:
        final_transcription = full_partial_transcription

    if not final_transcription:
        final_transcription = "ERROR (No transcription data available)"

    print("\n" * 2)
    print("=================================================================")
    print("         BASELINE LIVE MICROPHONE (NO NR) SUMMARY            ")
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
    print(f"Last Full Transcription: '{final_transcription}'")
    print(f"Total Errors Recorded: {total_errors}")
    print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}% (Lower is better)")

    # 2. Time Efficiency Summary
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- 2. Time Efficiency (Latency) Results ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(f"Frame Duration (Target): {FRAME_DURATION_MS} ms")
        print(f"Latency Verdict: {'PASS' if average_time < FRAME_DURATION_MS else 'FAIL'}")

    # 3. Noise Reduction Summary (Disabled)
    print(f"\n--- 3. Noise Reduction (NR) Effectiveness ---")
    print("Algorithm Used: NONE (RAW Microphone Input)")
    print("Noise Profile Frames: 0")
    print("Average Amplitude Reduction: N/A (0.00%)")

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