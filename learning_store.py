"""Self-learning feedback loop for the medical transcriber.

Stores (wrong -> right) corrections learned from user feedback, applies them
verbatim to future transcripts, and tracks how often each corrected term has
been confirmed so it can be promoted into the static vocabulary once it
crosses a confidence threshold.
"""

from __future__ import annotations

import json
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path

_WORD_RE = re.compile(r"(\S+)(\s*)")
_MAX_PHRASE_WORDS = 4
_MIN_WRONG_PHRASE_LEN = 4


def _core(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, alnum core, trailing punctuation)."""
    match = re.match(r"^(\W*)([\w'-]*)(\W*)$", token, flags=re.UNICODE)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


class LearningStore:
    """Persists learned corrections and tracks promotion eligibility."""

    def __init__(self, path: str | Path = "data/learned_corrections.json", promotion_threshold: int = 2):
        self.path = Path(path)
        self.promotion_threshold = promotion_threshold
        self._lock = threading.Lock()
        self._data = {
            "exact_corrections": {},
            "confirmation_counts": {},
            "term_display": {},
            "promoted_terms": [],
        }
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._data["exact_corrections"].update(loaded.get("exact_corrections", {}))
            self._data["confirmation_counts"].update(loaded.get("confirmation_counts", {}))
            self._data["term_display"].update(loaded.get("term_display", {}))
            self._data["promoted_terms"] = list(loaded.get("promoted_terms", []))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def record_feedback(self, raw_text: str, corrected_text: str) -> list[dict[str, str]]:
        """Word/phrase-align raw vs. corrected text and learn what changed.

        Returns the list of {"wrong": ..., "corrected": ...} pairs learned
        from this feedback event.
        """
        raw_words = raw_text.split()
        corrected_words = corrected_text.split()
        raw_lower = [w.lower() for w in raw_words]
        corrected_lower = [w.lower() for w in corrected_words]

        matcher = SequenceMatcher(a=raw_lower, b=corrected_lower, autojunk=False)
        opcodes = matcher.get_opcodes()
        learned: list[dict[str, str]] = []

        with self._lock:
            for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
                if tag not in ("replace", "insert"):
                    continue
                if j1 == j2:
                    continue
                wrong_phrase = " ".join(raw_words[i1:i2]).strip()
                right_phrase = " ".join(corrected_words[j1:j2]).strip()
                if not right_phrase or wrong_phrase.lower() == right_phrase.lower():
                    continue
                if not wrong_phrase:
                    # Pure insertion with nothing mis-heard on our side; nothing to key on.
                    continue

                # A short, single-word "wrong" side is dangerous to learn as a bare
                # mapping (e.g. "all" -> "Keterol" would corrupt every future
                # transcript that legitimately contains "all"). If a shared word
                # immediately follows, pull it into both sides so the learned key
                # is the more specific phrase actually spoken (e.g. "all dt" ->
                # "Keterol DT"). If no such context is available, skip it —
                # too risky to learn on a common word alone.
                if len(wrong_phrase.split()) == 1 and len(wrong_phrase) < _MIN_WRONG_PHRASE_LEN:
                    next_op = opcodes[idx + 1] if idx + 1 < len(opcodes) else None
                    if next_op and next_op[0] == "equal" and next_op[2] > next_op[1]:
                        _, ni1, _, nj1, _ = next_op
                        wrong_phrase = f"{wrong_phrase} {raw_words[ni1]}"
                        right_phrase = f"{right_phrase} {corrected_words[nj1]}"
                    else:
                        continue

                wrong_key = wrong_phrase.lower()
                self._data["exact_corrections"][wrong_key] = right_phrase

                right_key = right_phrase.lower()
                self._data["confirmation_counts"][right_key] = (
                    self._data["confirmation_counts"].get(right_key, 0) + 1
                )
                self._data["term_display"][right_key] = right_phrase

                learned.append({"wrong": wrong_phrase, "corrected": right_phrase})

            self._save()

        return learned

    def apply_learned(self, text: str) -> str:
        """Apply every exact-match learned correction to `text` (longest phrase first)."""
        corrections = self._data["exact_corrections"]
        if not corrections or not text:
            return text

        tokens = _WORD_RE.findall(text)  # list of (word, trailing_whitespace)
        cores = [_core(tok[0]) for tok in tokens]
        n = len(tokens)
        out: list[str] = []
        i = 0
        while i < n:
            matched = False
            max_span = min(_MAX_PHRASE_WORDS, n - i)
            for span in range(max_span, 0, -1):
                phrase_core = " ".join(cores[k][1].lower() for k in range(i, i + span))
                if phrase_core in corrections:
                    lead = cores[i][0]
                    trail = cores[i + span - 1][2]
                    replacement = f"{lead}{corrections[phrase_core]}{trail}"
                    trailing_ws = tokens[i + span - 1][1]
                    out.append(replacement + trailing_ws)
                    i += span
                    matched = True
                    break
            if not matched:
                out.append(tokens[i][0] + tokens[i][1])
                i += 1
        return "".join(out)

    def get_promotable_terms(self) -> list[str]:
        """Display-cased corrected terms that crossed the promotion threshold and aren't promoted yet."""
        promoted = set(self._data["promoted_terms"])
        return [
            self._data["term_display"].get(term, term)
            for term, count in self._data["confirmation_counts"].items()
            if count >= self.promotion_threshold and term not in promoted
        ]

    def mark_promoted(self, term: str) -> None:
        with self._lock:
            key = term.lower()
            if key not in self._data["promoted_terms"]:
                self._data["promoted_terms"].append(key)
                self._save()
