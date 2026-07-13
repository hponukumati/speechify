#!/usr/bin/env python3
"""Live microphone dictation.

Records straight from your default microphone into memory and transcribes
the instant you stop — no audio file is ever written to disk.

Usage:
    python record_and_transcribe.py
    python record_and_transcribe.py --model small
    python record_and_transcribe.py --learn
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd

from transcriber import MedicalTranscriber

SAMPLE_RATE = 16000  # what Whisper expects


def record_until_enter() -> np.ndarray:
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    input("Press Enter to start recording...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)
    stream.start()
    input("Recording... press Enter to stop.")
    stream.stop()
    stream.close()

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record your voice and transcribe it immediately, in-memory.")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="After each transcription, prompt for a correction and feed it into the learning loop.",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...", file=sys.stderr)
    transcriber = MedicalTranscriber(model_size=args.model)
    print("Ready.\n", file=sys.stderr)

    while True:
        audio = record_until_enter()
        if audio.size == 0:
            print("No audio captured.")
        else:
            duration = len(audio) / SAMPLE_RATE
            print(f"Captured {duration:.1f}s. Transcribing...", file=sys.stderr)
            result = transcriber.transcribe(audio)

            print("\n--- Raw transcript ---")
            print(result["raw_text"])
            print("\n--- Corrected transcript ---")
            print(result["corrected_text"])
            print()

            if args.learn:
                correction = input("Correct transcript (Enter to skip): ").strip()
                if correction:
                    outcome = transcriber.learn_from_correction(result["corrected_text"], correction)
                    print(f"Learned: {outcome['learned']}")
                    if outcome["promoted"]:
                        print(f"Promoted to static vocabulary: {outcome['promoted']}")

        again = input("\nRecord again? [Y/n] ").strip().lower()
        if again == "n":
            break


if __name__ == "__main__":
    main()
