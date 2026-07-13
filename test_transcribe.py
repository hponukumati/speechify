#!/usr/bin/env python3
"""CLI for the medical transcriber — no server needed.

Usage:
    python test_transcribe.py file.wav
    python test_transcribe.py file.wav --learn
    python test_transcribe.py file.wav --model small
"""

from __future__ import annotations

import argparse
import sys

from transcriber import MedicalTranscriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with medical-term correction.")
    parser.add_argument("audio_path", help="Path to a WAV/MP3/etc. audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="After transcribing, prompt for a correction and feed it into the learning loop.",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...", file=sys.stderr)
    transcriber = MedicalTranscriber(model_size=args.model)

    print(f"Transcribing {args.audio_path}...", file=sys.stderr)
    result = transcriber.transcribe(args.audio_path)

    print("\n--- Raw transcript ---")
    print(result["raw_text"])
    print("\n--- Corrected transcript ---")
    print(result["corrected_text"])
    print(f"\nLanguage: {result['language']}")
    print("\n--- Segments ---")
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    if args.learn:
        print("\nIs the corrected transcript above fully correct? If not, paste the")
        print("correct version below (or press Enter to skip).")
        correction = input("Correct transcript: ").strip()
        if correction:
            outcome = transcriber.learn_from_correction(result["corrected_text"], correction)
            print(f"\nLearned: {outcome['learned']}")
            if outcome["promoted"]:
                print(f"Promoted to static vocabulary: {outcome['promoted']}")
        else:
            print("No correction provided, skipping.")


if __name__ == "__main__":
    main()
