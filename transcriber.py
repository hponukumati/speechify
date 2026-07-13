"""Medical-specialized speech-to-text using faster-whisper.

Owns the Whisper model (loaded once), the medical vocabulary corrector, and
the self-learning store. Runs fully offline after the model weights have
been downloaded once.
"""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from learning_store import LearningStore
from vocab_corrector import MedicalVocabCorrector


class MedicalTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        vocab_path: str | Path = "data/medical_terms.json",
        learned_corrections_path: str | Path = "data/learned_corrections.json",
        promotion_threshold: int = 2,
        fuzzy_threshold: int = 82,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.learning_store = LearningStore(
            path=learned_corrections_path, promotion_threshold=promotion_threshold
        )
        self.corrector = MedicalVocabCorrector(
            vocab_path=vocab_path,
            learning_store=self.learning_store,
            fuzzy_threshold=fuzzy_threshold,
        )

    def transcribe(self, audio) -> dict:
        """Transcribe a file path (str/Path) or an in-memory mono float32 audio array.

        Passing an array (e.g. captured live from a microphone) skips disk
        entirely — faster-whisper accepts raw samples directly.
        """
        initial_prompt = self.corrector.build_initial_prompt()
        audio_arg = str(audio) if isinstance(audio, (str, Path)) else audio

        segments_iter, info = self.model.transcribe(
            audio_arg,
            beam_size=5,
            vad_filter=True,
            initial_prompt=initial_prompt,
        )

        segments = []
        raw_parts = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
            )
            raw_parts.append(seg.text.strip())

        raw_text = " ".join(raw_parts).strip()
        corrected_text = self.corrector.correct(raw_text)

        corrected_segments = []
        for seg in segments:
            corrected_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": self.corrector.correct(seg["text"]),
                }
            )

        return {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "language": info.language,
            "segments": corrected_segments,
        }

    def learn_from_correction(self, raw_text: str, corrected_text: str) -> dict:
        return self.corrector.learn_from_feedback(raw_text, corrected_text)
