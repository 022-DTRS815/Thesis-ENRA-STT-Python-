import os
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import sys
import time

# --- Target Phrase for Accuracy Evaluation (No. 1) ---
# NOTE: Speak this exact sentence when the program is running to test accuracy.
TARGET_PHRASE = "i am testing the algorithm performance"
total_test_phrases = 0
correct_phrases = 0
total_words_in_tests = 0
total_errors = 0

# --- Audio Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- Vosk Model Setup (Scalability: Model Load Time) ---
MODEL_PATH = "D:/JetBrains/Projects/ENRA-STT Models/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at {MODEL_PATH}")
    print("Please download a model from https://alphacephei.com/vosk/models and set the correct path.")
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


# --- Functions ---
def calculate_accuracy(reference, hypothesis):
    """Simple word-level accuracy calculation and error counting."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Calculate correct matches up to the length of the shorter list
    correct = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)

    # Simple error count: difference between the length of the longest phrase
    # and the number of correct words. This approximates the total
    # substitutions, deletions, and insertions (S+D+I).
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

print(f"*** ACCURACY TEST: Say the target phrase: '{TARGET_PHRASE.upper()}' ***")
print("Listening for speech...")

# Data containers for Time Efficiency (No. 2)
total_frames = 0
total_processing_time = 0.0

try:
    while True:
        # Read the audio chunk
        data = stream.read(CHUNK, exception_on_overflow=False)
        start_time = time.perf_counter()  # <--- Start timing

        # Pass the raw audio data directly to the Vosk recognizer
        if rec.AcceptWaveform(data):
            result = rec.Result()
            transcribed_text = eval(result)['text'].strip()

            # Print transcription on a new line to avoid overwriting metrics
            print(f"\nTranscription: {transcribed_text}")

            # --- Accuracy Logic (No. 1) ---
            if transcribed_text:
                total_test_phrases += 1

                # Calculate accuracy for this attempt
                c, e, n = calculate_accuracy(TARGET_PHRASE, transcribed_text)

                total_words_in_tests += n
                total_errors += e

                if transcribed_text == TARGET_PHRASE:
                    correct_phrases += 1
                    print(">>> PHRASE MATCH: 100% ACCURACY <<<")
                else:
                    word_accuracy = (c / n) * 100 if n > 0 else 0
                    print(f"  --- Word Accuracy: {word_accuracy:.2f}% ({c} / {n} words correct)")
                    print(f"  --- Errors in Phrase: {e}")

                # Reset recognizer for the next phrase
                rec.FinalResult()

        else:
            partial_result = rec.PartialResult()
            # Print the partial result
            print("Partial:", eval(partial_result)['partial'], end="\r")

        end_time = time.perf_counter()  # <--- End timing

        # --- Time Efficiency Data (No. 2) ---
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_processing_time += processing_time
        total_frames += 1

        # Display real-time efficiency metric
        print(f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms)", end="\r")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # --- Final Evaluation Results Summary ---

    print("\n" * 2)
    print("=========================================================")
    print("         BASELINE ALGORITHM EVALUATION SUMMARY           ")
    print("=========================================================")

    # 4. Scalability Summary (Model Load Time)
    print(f"\n--- 4. Scalability (Model Load) Results ---")
    print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

    # 1. Accuracy Summary
    if total_test_phrases > 0:
        phrase_accuracy = (correct_phrases / total_test_phrases) * 100

        # Calculate a cumulative Word Error Rate (WER) approximation
        approx_wer = (total_errors / total_words_in_tests) * 100

        print(f"\n--- 1. Accuracy Results (Approx. WER) ---")
        print(f"Test Attempts: {total_test_phrases}")
        print(f"Phrase Match Accuracy: {phrase_accuracy:.2f}%")
        print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}%")

    # 2. Time Efficiency Summary
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- 2. Time Efficiency (Latency) Results ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(f"Frame Duration (Target): {FRAME_DURATION_MS} ms")
        print(
            f"Latency Verdict: {'PASS' if average_time < FRAME_DURATION_MS else 'FAIL'} (High Latency means processing is too slow)")

    # 3. Noise Reduction (NR) Data - Not applicable for baseline, but included for complete framework
    print(f"\n--- 3. Noise Reduction Results ---")
    print("N/A: This is the baseline, no NR component is used.")

    print("=========================================================")

    # Close the stream and PyAudio interface
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    p.terminate()