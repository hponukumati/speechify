# Speechify — Local Medical Speech-to-Text

A fully offline speech-to-text tool built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
specialized for medical terminology (drug names, clinical terms, abbreviations).

Whisper has no concept of a hard vocabulary list, so medical terms routinely
get mangled ("lisinoprel" instead of "Lisinopril", "Zannux" instead of
"Xanax"). This project closes that gap with three combined correction
techniques, plus a feedback loop that lets the tool learn new terms as it's
used.

## How the correction pipeline works

Every transcript passes through four layers, in this order:

1. **Learned exact corrections** (`learning_store.py`) — mappings the tool
   has already been taught via `/feedback`, applied verbatim before anything
   else runs.
2. **Prompt biasing** (`vocab_corrector.build_initial_prompt`) — before
   transcription even starts, the full vocabulary (terms + abbreviation
   expansions) is fed to Whisper as `initial_prompt`, biasing the decoder
   toward those tokens.
3. **Fuzzy string correction** (`vocab_corrector.py`, rapidfuzz) — every word
   and multi-word phrase in the output is compared against the medical
   dictionary using `rapidfuzz.fuzz.ratio`. Multi-word phrases (e.g. "type 2
   diabetes") are matched first via a sliding window, longest phrase first,
   so they aren't broken up by single-word matching. Anything scoring ≥82
   against a dictionary term gets replaced with the term's canonical
   spelling. This catches typo-like errors: "hypertention" → "hypertension".
4. **Phonetic correction** (`vocab_corrector.py`, Double Metaphone) — a
   **fallback**, only tried when a word finds no fuzzy match. Each
   vocabulary term is encoded phonetically at load time with Double
   Metaphone. A transcribed word is corrected if its own phonetic code is a
   close match (fuzzy-compared, not required to be byte-identical — this
   catches near-miss codes like a missing trailing consonant, e.g.
   "dexarine" → `TKSRN` vs. "Dexorange" → `TKSRNJ`) to an indexed term's
   code, *and* two guards both pass: the matched term's length is within 3
   characters of the input word, and the raw spelling similarity between the
   two is at least 45% (this second guard is what keeps short, coincidental
   code collisions — e.g. "west"/"Asthma" both reduce toward similar
   3-4 character codes despite sharing almost no letters — from firing).
   This layer catches mishearings that sound alike but share almost no
   letters: "Zannux" → "Xanax" (plain string similarity between those two is
   only ~55%, well under the fuzzy threshold).

Layers 3 and 4 only touch single words of 4+ alphabetic characters (or
phrases resolving to a word above the fuzzy threshold) — short, common words
are left alone by construction, which is most of what keeps everyday
sentences untouched.

## Self-learning feedback loop

The static dictionary in `data/medical_terms.json` only covers terms someone
thought to add ahead of time. `learning_store.py` lets the tool improve from
real corrections:

1. Submit `(raw_text, corrected_text)` — what the tool produced vs. what it
   should have said — via `POST /feedback` or `test_transcribe.py --learn`.
2. `LearningStore.record_feedback` word/phrase-aligns the two strings with
   `difflib.SequenceMatcher`, extracting exactly what changed. This
   generalizes to single-word ("Zannux" → "Xanax") and multi-word
   corrections alike.
3. **Safety guard**: if the diff isolates a short (<4 char), single-word
   "wrong" side — e.g. correcting "The patient needs all DT" → "...
   Keterol DT" naively diffs down to bare `all` → `Keterol`, since "DT" is
   unchanged on both sides — that mapping is too dangerous to learn as-is
   ("all" is an extremely common word). Instead, one word of trailing
   context is pulled in from both sides to key the mapping on the fuller,
   safer phrase actually spoken (`all dt` → `Keterol DT`), or the mapping is
   skipped entirely if no such context exists.
4. Each learned `(wrong → right)` pair is stored and applied verbatim to
   every future transcript, before the fuzzy/phonetic layers run.
5. Separately, a confirmation count is kept **per corrected term** — it
   increments any time feedback resolves to that term, even from different
   original mishearings (e.g. "Zannux" → "Xanax" once, "Zanax" → "Xanax"
   another time — both count toward "Xanax"). Once a term's count reaches
   the promotion threshold (default **2**), it is written permanently into
   `data/medical_terms.json`, and both the fuzzy and phonetic indexes are
   rebuilt — so the whole pipeline benefits from it, not just the exact-match
   layer.
6. Everything is persisted to `data/learned_corrections.json` (plain JSON —
   fine for a single user or small team; see Hosting notes below for scaling
   this up).

The corrector works fine with no `LearningStore` attached (pass
`learning_store=None`) — the exact-match layer is simply skipped and
`learn_from_feedback` raises if called without one.

## Project structure

```
main.py                       FastAPI app: /transcribe, /feedback, /health
transcriber.py                MedicalTranscriber — owns the Whisper model, corrector, learning store
vocab_corrector.py            MedicalVocabCorrector — fuzzy + phonetic correction, promotion logic
learning_store.py             LearningStore — feedback alignment, persistence, promotion tracking
data/medical_terms.json       Editable vocabulary: terms + abbreviations
data/learned_corrections.json Learned corrections (created on first feedback)
test_transcribe.py            CLI: python test_transcribe.py file.wav [--learn]
record_and_transcribe.py      Live mic dictation: python record_and_transcribe.py [--learn]
requirements.txt
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended — `faster-whisper`'s underlying
`ctranslate2` engine does not yet ship wheels for very new Python releases.

The first transcription downloads the chosen Whisper model from Hugging Face
and caches it locally (`~/.cache/huggingface`). Every run after that is
**fully offline**.

## Usage

### Live microphone (no file, no server)

```bash
python record_and_transcribe.py
python record_and_transcribe.py --model small
python record_and_transcribe.py --learn
```

Press Enter to start recording, speak, press Enter again to stop — the
transcript prints immediately. Audio is captured straight from the
microphone into memory (via `sounddevice`) and handed directly to
`faster-whisper` as a raw array; nothing is written to disk. Loops so you
can do multiple takes without reloading the model each time.

### CLI, from an audio file (no server)

```bash
python test_transcribe.py path/to/audio.wav
python test_transcribe.py path/to/audio.wav --model small
python test_transcribe.py path/to/audio.wav --learn   # prompts for a correction afterward
```

### API

```bash
uvicorn main:app --reload
```

```bash
# Transcribe
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"

# Teach a correction
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "patient takes zannux daily", "corrected_text": "patient takes Xanax daily"}'

# Health check
curl http://localhost:8000/health
```

`/transcribe` returns:

```json
{
  "raw_text": "...",
  "corrected_text": "...",
  "language": "en",
  "segments": [{"start": 0.0, "end": 2.3, "text": "..."}]
}
```

## Extending the vocabulary

Edit `data/medical_terms.json` directly — `terms` is a flat list of drug
names, conditions, procedures, and phrases (multi-word entries are matched
as phrases automatically); `abbreviations` maps short forms to their
expansions and is used for prompt biasing only (abbreviations are
deliberately excluded from fuzzy/phonetic correction — 2-3 letter tokens are
too prone to false-positive matches against ordinary words). Changes take
effect on next process start, or immediately if promoted programmatically
via the feedback loop.

## Accuracy tuning notes

- **Model size**: `tiny`/`base` are fast enough for real-time-ish use on a
  CPU but make more raw transcription errors for the correction layers to
  fix. `small` is a good CPU accuracy/speed tradeoff if latency allows;
  `medium`/`large-v3` need a GPU (or a lot of patience) but make the fuzzy
  and phonetic layers work far less hard. Set via `--model` (CLI) or
  `WHISPER_MODEL_SIZE` env var (API).
- **Fuzzy threshold (82)**: tuned empirically, and moved up from an initial
  76 after real testing turned up a false positive — "violin" vs "Vicodin"
  scores 76.9, just over the original threshold. Legitimate typo-distance
  errors ("hypertention" vs "hypertension", "lisinoprel" vs "Lisinopril",
  etc.) all score 87.5+ in practice, so 82 keeps a comfortable margin above
  real coincidental collisions while still catching every typo-class error
  tested. Lower it and you'll catch more typos at higher false-positive
  risk; raise it and you'll miss more real errors.
- **Phonetic strictness**: two guards keep Double Metaphone's coarse,
  collision-prone codes from over-firing. The length-difference guard
  (`±3` characters) rejects candidates whose length is way off from the
  input. The word-similarity floor (45% raw spelling ratio) rejects short
  codes that coincidentally collide despite the words sharing almost no
  letters — this was added after real testing caught "west" → "Asthma" and
  "breeze" → "Prozac" firing once phonetic code matching was loosened from
  exact-equality to fuzzy comparison (needed to catch real cases like
  "dexarine" → "Dexorange", where the codes are one trailing consonant
  apart, not identical). Tightening either guard further reduces false
  positives at the cost of missing more legitimate matches.
- **Minimum word length (4 chars)**: words shorter than this are skipped by
  both the fuzzy and phonetic layers entirely — short common words (a, is,
  to, of...) are the highest false-positive risk and carry the least
  information for matching anyway.

## Hosting notes

- **Docker**: the app is structured so containerizing needs no code
  changes — `main.py` is a standard FastAPI/uvicorn ASGI app, all state
  (model, vocab, learned corrections) is loaded from disk paths relative to
  the working directory. A minimal Dockerfile:

  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  ENV WHISPER_MODEL_SIZE=base
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- **Bake model weights into the image**: to avoid a cold-start download on
  first request (and to keep the container truly offline), run a
  `WhisperModel(...)` instantiation once during the image build (or `docker
  run` it once against a volume) so the Hugging Face cache is populated
  inside the image/layer rather than fetched at runtime.

- **Auth**: there is currently none. Before exposing `/transcribe` or
  `/feedback` beyond localhost, put an auth layer in front (API key
  middleware, or a reverse proxy like nginx/Caddy with basic auth or OAuth2
  — FastAPI's `Depends`-based security utilities are the natural fit if you
  want it in-process).

- **Learning store at scale**: `data/learned_corrections.json` is a single
  JSON file guarded by an in-process lock — fine for one user or a small
  team on one instance, but it will not survive multiple worker processes
  or horizontal scaling (concurrent writes will race, and each process
  would have its own in-memory copy). For many concurrent users, replace
  `LearningStore`'s persistence with a real database (SQLite is enough for
  a single-node deployment; Postgres if you're running multiple instances
  behind a load balancer) — the public interface
  (`record_feedback`/`apply_learned`/`get_promotable_terms`/`mark_promoted`)
  is intentionally small so the storage backend can be swapped without
  touching `vocab_corrector.py` or `transcriber.py`.
