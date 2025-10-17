import os
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import sys
import time
import wave
import random
import glob

# --- File Paths and Test Configuration ---
# *** IMPORTANT: You MUST set these paths to your test files ***
SPEECH_FILE = "D:/JetBrains/Projects/ENRA-STT WAV Files/Speech/Sample_4.wav"
# CHANGE THIS TO THE FOLDER CONTAINING YOUR .WAV NOISE FILES
NOISE_FOLDER = "D:/JetBrains/Projects/ENRA-STT WAV Files/Noise/continuous"
# The TARGET_PHRASE MUST match the spoken content of your SPEECH_FILE.
TARGET_PHRASE = "i am testing the accuracy"

# --- Evaluation Variables ---
total_test_phrases = 1
total_words_in_test = len(TARGET_PHRASE.split())
total_errors = 0
total_frames = 0
total_processing_time = 0.0
# The NR variable is kept for the final summary, but will always be 0
total_reduction_factor = 0.0
random_noise_file = ""
full_partial_transcription = ""

# --- Audio and VAD Parameters ---
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1
# VAD_AGGRESSIVENESS = 3 <-- REMOVED

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


# --- Functions ---
# Only keep the accuracy function
def calculate_accuracy(reference, hypothesis):
    """Simple word-level accuracy calculation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    correct = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    errors = max(len(ref_words), len(hyp_words)) - correct

    return correct, errors, len(ref_words)


# --- Main Logic ---
p = pyaudio.PyAudio()
stream = None
wf_speech = None
wf_noise = None

try:
    # 1. Select and Open the audio files
    if not os.path.exists(SPEECH_FILE):
        raise FileNotFoundError("Speech file not found. Please check SPEECH_FILE path.")
    if not os.path.isdir(NOISE_FOLDER):
        raise FileNotFoundError(f"Noise folder not found at {NOISE_FOLDER}.")

    noise_files = glob.glob(os.path.join(NOISE_FOLDER, "*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"No WAV files found in the noise folder: {NOISE_FOLDER}")

    random_noise_file = random.choice(noise_files)

    wf_speech = wave.open(SPEECH_FILE, 'rb')
    wf_noise = wave.open(random_noise_file, 'rb')

    if wf_speech.getframerate() != RATE or wf_speech.getnchannels() != CHANNELS:
        raise ValueError(f"Speech file format error: Requires {RATE}Hz, {CHANNELS} channel.")
    if wf_noise.getframerate() != RATE or wf_noise.getnchannels() != CHANNELS:
        raise ValueError(f"Noise file format error: Requires {RATE}Hz, {CHANNELS} channel.")

    # 2. Open an OUTPUT stream for playback
    stream = p.open(format=p.get_format_from_width(wf_speech.getsampwidth()),
                    channels=wf_speech.getnchannels(),
                    rate=wf_speech.getframerate(),
                    output=True,
                    frames_per_buffer=CHUNK)

    # -----------------------------------------------------------
    # Step 1: Base Line Setup (NO NOISE PROFILE)
    # -----------------------------------------------------------
    print("\n*** Starting BASELINE Real-Time Simulation (NOISE REDUCTION DISABLED) ***")
    print(
        f"Mixing '{os.path.basename(SPEECH_FILE)}' with RANDOMLY SELECTED NOISE: '{os.path.basename(random_noise_file)}'")

    # -----------------------------------------------------------
    # Step 2: Main real-time processing loop (Mixing and Processing)
    # -----------------------------------------------------------
    while True:
        data_speech = wf_speech.readframes(CHUNK)
        data_noise = wf_noise.readframes(CHUNK)

        if not data_speech:
            break

        start_time = time.perf_counter()

        # --- Audio Mixing and Preparation ---
        np_speech = np.frombuffer(data_speech, dtype=np.int16)

        if not data_noise:
            wf_noise.rewind()
            data_noise = wf_noise.readframes(CHUNK)

        np_noise = np.frombuffer(data_noise, dtype=np.int16)

        min_len = min(len(np_speech), len(np_noise))
        np_speech = np_speech[:min_len]
        np_noise = np_noise[:min_len]

        # FIX: Ensure proper padding for addition if lengths are different due to EOF
        speech_len = len(np_speech)
        noise_len = len(np_noise)
        if noise_len < speech_len:
            np_noise = np.pad(np_noise, (0, speech_len - noise_len), 'constant')
        elif speech_len < noise_len:
            np_noise = np_noise[:speech_len]

        mixed_audio = np_speech + np_noise
        mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)

        # The audio frame to process is the raw, mixed audio
        audio_to_process = mixed_audio.tobytes()

        # --- NOISE REDUCTION STEPS ARE SKIPPED ---

        # 3. Vosk Transcription
        if rec.AcceptWaveform(audio_to_process):
            result = eval(rec.Result())
            print(f"[{total_frames}] Segment Transcribed: {result['text']}", end='\r')
            # *** CAPTURE: Add successfully accepted segment text ***
            if result.get('text'):
                full_partial_transcription += result['text'] + " "
        else:
            partial_result = eval(rec.PartialResult())
            print(f"[{total_frames}] Partial: {partial_result['partial']}", end='\r')
            # NOTE: We do *not* append the partial result here, as it constantly updates.
            # We only capture the finalized segments in the 'if rec.Result()' block.

        # Play the MIXED (noisy) audio back
        try:
            stream.write(audio_to_process)
        except IOError as e:
            if e.errno == -9999:
                print(f"\n--- FATAL ERROR: [Errno -9999] Unanticipated host error detected. Exiting loop. ---")
                break
            else:
                raise

        if not stream.is_active():
            print("\nAudio stream unexpectedly became inactive. Exiting loop.")
            break

        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        total_processing_time += processing_time
        total_frames += 1

        print(f"Latency: {processing_time:.2f} ms (Target < {FRAME_DURATION_MS} ms)", end="\r")


except Exception as e:
    print(f"\nFatal Error during processing: {e}")

finally:
    # -----------------------------------------------------------
    # FINAL EVALUATION SUMMARY
    # -----------------------------------------------------------

    # 1. Try to get the official final result
    final_transcription = ""
    try:
        if 'rec' in locals():
            # Force the final segment to output anything left in the buffer
            final_result_json = rec.FinalResult()
            final_transcription = eval(final_result_json)['text'].strip()
    except Exception:
        pass

    # 2. *** CRITICAL MODIFICATION: If final result is empty, use the aggregated segments ***
    if not final_transcription and full_partial_transcription:
        final_transcription = full_partial_transcription.strip()
    elif not final_transcription:
        # Fallback for truly empty runs
        final_transcription = "ERROR (No transcription data available)"

    print("\n" * 2)
    print("=================================================================")
    print("             BASELINE ALGORITHM EVALUATION SUMMARY             ")
    print("=================================================================")

    print(f"\n--- 4. Scalability (Model Load) Results ---")
    print(f"Vosk Model Load Time: {model_load_time:.2f} seconds")

    # 1. Accuracy Summary
    c, e, n = calculate_accuracy(TARGET_PHRASE, final_transcription)
    # Ensure n > 0 to avoid division by zero
    approx_wer = (e / n) * 100 if n > 0 else 100

    print(f"\n--- 1. Accuracy Results (File Test) ---")
    print(f"Noise File Used: '{os.path.basename(random_noise_file)}'")
    print(f"Target Phrase: '{TARGET_PHRASE}'")
    # *** NOTE: This transcription may be stitched together from partial segments ***
    print(f"Final Transcription (Aggregated): '{final_transcription}'")
    print(f"Words Correct: {c} / {n}")
    print(f"Approx. Word Error Rate (WER): {approx_wer:.2f}% (Lower is better)")

    # 2. Time Efficiency Summary
    if total_frames > 0:
        average_time = total_processing_time / total_frames
        print(f"\n--- 2. Time Efficiency (Latency) Results ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Frame Processing Time: {average_time:.2f} ms")
        print(f"Frame Duration (Target): {FRAME_DURATION_MS} ms")
        print(f"Latency Verdict: {'PASS' if average_time < FRAME_DURATION_MS else 'FAIL'}")

    # 3. Noise Reduction Summary (Disabled for Baseline)
    print(f"\n--- 3. Noise Reduction (NR) Effectiveness ---")
    print("Noise Reduction is DISABLED in this baseline test.")
    print("Average Amplitude Reduction: N/A")

    print("=================================================================")

    # Safely close resources
    if wf_speech:
        wf_speech.close()
    if wf_noise:
        wf_noise.close()

    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except OSError as e:
            if 'Stream not open' in str(e):
                pass
            else:
                print(f"Error during stream close: {e}")

    p.terminate()