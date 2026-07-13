"""Medical vocabulary post-correction for Whisper transcripts.

Combines three layers, applied in this order:

1. Exact learned corrections (from the LearningStore, if attached) — fixes
   mistakes the tool has already been taught, before anything else runs.
2. Fuzzy string matching (rapidfuzz) — catches typo-like near-misses such as
   "lisinoprel" -> "Lisinopril". Multi-word phrases are matched before
   single words via a sliding window.
3. Phonetic matching (Double Metaphone) — a fallback for words that sound
   like the target term but are spelled very differently, e.g.
   "Zannux" -> "Xanax". Only runs when the fuzzy layer finds nothing.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from metaphone import doublemetaphone
from rapidfuzz import fuzz, process

_WORD_RE = re.compile(r"(\S+)(\s*)")
_MAX_PHRASE_WORDS = 4
_MIN_WORD_LEN_FOR_MATCH = 4
_PHONETIC_LENGTH_GUARD = 3
_PHONETIC_CODE_FUZZY_THRESHOLD = 85
_PHONETIC_MIN_WORD_RATIO = 45


def _core(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, alnum core, trailing punctuation)."""
    match = re.match(r"^(\W*)([\w'-]*)(\W*)$", token, flags=re.UNICODE)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


class MedicalVocabCorrector:
    """Fuzzy + phonetic correction against a medical vocabulary, with optional self-learning."""

    def __init__(
        self,
        vocab_path: str | Path = "data/medical_terms.json",
        learning_store=None,
        fuzzy_threshold: int = 82,
    ):
        self.vocab_path = Path(vocab_path)
        self.learning_store = learning_store
        self.fuzzy_threshold = fuzzy_threshold
        self._lock = threading.Lock()
        self._raw_vocab: dict = {"terms": [], "abbreviations": {}}
        self._load_vocab()
        self._build_indexes()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _load_vocab(self) -> None:
        if self.vocab_path.exists():
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self._raw_vocab = json.load(f)
        self._raw_vocab.setdefault("terms", [])
        self._raw_vocab.setdefault("abbreviations", {})

    def _save_vocab(self) -> None:
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._raw_vocab, f, indent=2)

    def _build_indexes(self) -> None:
        terms = self._raw_vocab.get("terms", [])

        # Single-word vs. multi-word phrase terms, keyed lowercase -> canonical casing.
        self._single_terms: dict[str, str] = {}
        self._phrase_terms: dict[str, str] = {}
        self._phrases_by_length: dict[int, list[str]] = {}

        for term in terms:
            words = term.split()
            if len(words) == 1:
                self._single_terms[term.lower()] = term
            else:
                key = term.lower()
                self._phrase_terms[key] = term
                self._phrases_by_length.setdefault(len(words), []).append(key)

        self._single_candidates = list(self._single_terms.keys())

        # Phonetic index: metaphone code -> list of canonical single-word terms.
        self._phonetic_index: dict[str, list[str]] = {}
        for lower_term, canonical in self._single_terms.items():
            for code in doublemetaphone(lower_term):
                if not code:
                    continue
                self._phonetic_index.setdefault(code, [])
                if canonical not in self._phonetic_index[code]:
                    self._phonetic_index[code].append(canonical)
        self._phonetic_code_list = list(self._phonetic_index.keys())

    # ------------------------------------------------------------------
    # Prompt biasing support
    # ------------------------------------------------------------------

    def build_initial_prompt(self) -> str:
        """Build a Whisper `initial_prompt` string that primes the decoder on vocabulary."""
        terms = self._raw_vocab.get("terms", [])
        abbreviations = self._raw_vocab.get("abbreviations", {})
        parts = []
        if terms:
            parts.append("Medical vocabulary: " + ", ".join(terms) + ".")
        if abbreviations:
            abbrev_str = ", ".join(f"{k} ({v})" for k, v in abbreviations.items())
            parts.append("Abbreviations: " + abbrev_str + ".")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Correction pipeline
    # ------------------------------------------------------------------

    def correct(self, text: str) -> str:
        if not text:
            return text

        if self.learning_store is not None:
            text = self.learning_store.apply_learned(text)

        tokens = _WORD_RE.findall(text)  # [(word, trailing_ws), ...]
        cores = [_core(tok[0]) for tok in tokens]
        n = len(tokens)
        consumed = [False] * n

        out = list(tokens)  # mutable copy we overwrite in place

        # --- Layer 2a: multi-word phrase fuzzy matching (longest phrases first) ---
        for span in sorted(self._phrases_by_length.keys(), reverse=True):
            candidates = self._phrases_by_length[span]
            if not candidates:
                continue
            i = 0
            while i <= n - span:
                if any(consumed[i : i + span]):
                    i += 1
                    continue
                window_core = " ".join(cores[k][1] for k in range(i, i + span)).strip().lower()
                if not window_core:
                    i += 1
                    continue
                match = process.extractOne(window_core, candidates, scorer=fuzz.ratio)
                if match and match[1] >= self.fuzzy_threshold:
                    matched_key = match[0]
                    canonical = self._phrase_terms[matched_key]
                    lead = cores[i][0]
                    trail = cores[i + span - 1][2]
                    trailing_ws = tokens[i + span - 1][1]
                    out[i] = (f"{lead}{canonical}{trail}", trailing_ws)
                    for k in range(i + 1, i + span):
                        out[k] = ("", "")
                    for k in range(i, i + span):
                        consumed[k] = True
                    i += span
                else:
                    i += 1

        # --- Layer 2b: single-word fuzzy matching + Layer 3: phonetic fallback ---
        if self._single_candidates:
            for i in range(n):
                if consumed[i]:
                    continue
                lead, word_core, trail = cores[i]
                if len(word_core) < _MIN_WORD_LEN_FOR_MATCH or not word_core.isalpha():
                    continue

                word_lower = word_core.lower()
                match = process.extractOne(word_lower, self._single_candidates, scorer=fuzz.ratio)

                if match and match[1] >= self.fuzzy_threshold:
                    canonical = self._single_terms[match[0]]
                    out[i] = (f"{lead}{canonical}{trail}", tokens[i][1])
                    consumed[i] = True
                    continue

                # Fuzzy found nothing usable -> try phonetic fallback.
                phonetic_match = self._phonetic_fallback(word_lower)
                if phonetic_match:
                    out[i] = (f"{lead}{phonetic_match}{trail}", tokens[i][1])
                    consumed[i] = True

        return "".join(word + ws for word, ws in out)

    def _phonetic_fallback(self, word_lower: str) -> str | None:
        if not self._phonetic_code_list:
            return None

        matched_codes: set[str] = set()
        for code in doublemetaphone(word_lower):
            if not code:
                continue
            if code in self._phonetic_index:
                # Exact code match — cheap, and always at least as good as a fuzzy one.
                matched_codes.add(code)
                continue
            # No exact code match: allow near-identical codes (e.g. missing a
            # trailing consonant cluster) rather than requiring byte-for-byte equality.
            fuzzy_code_match = process.extractOne(code, self._phonetic_code_list, scorer=fuzz.ratio)
            if fuzzy_code_match and fuzzy_code_match[1] >= _PHONETIC_CODE_FUZZY_THRESHOLD:
                matched_codes.add(fuzzy_code_match[0])

        candidates: list[str] = []
        for code in matched_codes:
            for canonical in self._phonetic_index.get(code, []):
                if abs(len(canonical) - len(word_lower)) > _PHONETIC_LENGTH_GUARD:
                    continue
                # A same-sounding code isn't enough on its own for short codes
                # (e.g. "west"/"Asthma" and "breeze"/"Prozac" both reduce to
                # near-identical short codes purely by coincidence). Require
                # some minimal spelling resemblance too.
                if fuzz.ratio(word_lower, canonical.lower()) < _PHONETIC_MIN_WORD_RATIO:
                    continue
                candidates.append(canonical)
        if not candidates:
            return None
        # Among surviving candidates, prefer the one closest in spelling.
        best = max(set(candidates), key=lambda c: fuzz.ratio(word_lower, c.lower()))
        return best

    # ------------------------------------------------------------------
    # Self-learning / promotion
    # ------------------------------------------------------------------

    def learn_from_feedback(self, raw_text: str, corrected_text: str) -> dict:
        """Record feedback via the LearningStore and promote any terms that qualify."""
        if self.learning_store is None:
            raise RuntimeError("No LearningStore attached to this corrector.")

        learned = self.learning_store.record_feedback(raw_text, corrected_text)
        promoted = self._promote_eligible_terms()
        return {"learned": learned, "promoted": promoted}

    def _promote_eligible_terms(self) -> list[str]:
        promotable = self.learning_store.get_promotable_terms()
        promoted: list[str] = []

        with self._lock:
            existing_lower = {t.lower() for t in self._raw_vocab.get("terms", [])}
            changed = False
            for term in promotable:
                if term.lower() in existing_lower:
                    self.learning_store.mark_promoted(term)
                    continue
                self._raw_vocab.setdefault("terms", []).append(term)
                existing_lower.add(term.lower())
                self.learning_store.mark_promoted(term)
                promoted.append(term)
                changed = True

            if changed:
                self._save_vocab()
                self._build_indexes()

        return promoted
