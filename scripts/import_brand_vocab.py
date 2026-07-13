#!/usr/bin/env python3
"""Import brand-name vocabulary from a source CSV (e.g. a Kaggle Indian-medicines dataset).

Designed to be re-run as new/updated source data becomes available (v1, v2, v3, ...).
Cleans dosage/form noise off each entry, filters out manufacturer-name bleed-through,
ranks candidates by variant-count (how many strength/pack SKUs share a base brand) as a
rough signal for "this is a real, actively-manufactured brand" vs. one-off scrape noise,
dedupes against the existing vocabulary, and merges a curated top-N slice.

Requires pandas (not a runtime dependency of the app — install separately: pip install pandas).

Usage:
    python scripts/import_brand_vocab.py path/to/dataset.csv                  # dry run, just report
    python scripts/import_brand_vocab.py path/to/dataset.csv --apply          # write changes
    python scripts/import_brand_vocab.py path/to/dataset.csv --apply --limit 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = PROJECT_ROOT / "data" / "medical_terms.json"
IMPORT_LOG_PATH = PROJECT_ROOT / "data" / "vocab_import_log.json"

_STOP_TOKEN_RE = re.compile(r"\d")
_VALID_BRAND_RE = re.compile(r"^[A-Za-z][A-Za-z\s\-]*$")

# Generic dosage-form / flavor words — never part of a spoken brand name, and some
# ("tablet", "cream", "drops"...) are common English words that would be dangerous
# to leave in as standalone correction candidates.
_FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "syrup", "injection", "cream", "gel", "drop", "drops",
    "suspension", "ointment", "lotion", "powder", "spray", "inhaler", "solution", "sachet", "lozenge",
    "patch", "soap", "shampoo", "kit", "wash", "infusion", "liquid", "expectorant", "bar", "strip",
    "orange", "mango", "strawberry", "mint", "each", "chewable", "tabs", "vial", "mg", "ml", "mcg",
}

# Legal-entity / manufacturer-suffix words that indicate the scraped name bled into a
# company name rather than being (only) a brand name.
_COMPANY_WORDS = {
    "pvt", "ltd", "limited", "pharma", "pharmaceutical", "pharmaceuticals", "healthcare",
    "remedies", "biotech", "laboratories", "formulations", "sciences", "lifesciences",
    "industries", "drugs", "chemicals", "labs", "therapeutics", "biosciences",
}


def base_brand(name: str) -> str | None:
    """Extract the leading brand-name tokens, stopping at the first dosage/form/company token.

    If a company-suffix word (e.g. "Pharma") is what stopped us, the tokens collected
    so far are almost certainly a manufacturer name that got prepended instead of a real
    brand (e.g. "Kriam Pharma Cefixime 100mg..." — "Kriam" isn't a brand, it's half of
    "Kriam Pharma"). The well-formed majority pattern is "{Brand} {dosage} {form}", where
    a digit is hit before any company word ever would be — so hitting a company word
    first is itself the anomaly signal. Discard the whole candidate in that case.
    """
    out = []
    for tok in name.strip().split():
        if _STOP_TOKEN_RE.search(tok):
            break
        bare = re.sub(r"[^a-zA-Z]", "", tok).lower()
        if not bare:
            break
        if bare in _COMPANY_WORDS:
            return None
        if bare in _FORM_WORDS:
            break
        out.append(tok)
    brand = " ".join(out).strip()
    return brand or None


def is_plausible(brand: str) -> bool:
    # Minimum length matches vocab_corrector's own MIN_WORD_LEN_FOR_MATCH guard —
    # very short target terms produce noisy fuzzy-ratio scores (percentage-based
    # similarity is unstable on short strings) and are more likely to collide with
    # ordinary short words ("Zap", "Azi").
    if not (4 <= len(brand) <= 24):
        return False
    if len(brand.split()) > 3:
        return False
    if not _VALID_BRAND_RE.match(brand):
        return False
    return True


def load_existing_terms() -> tuple[dict, set[str]]:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    existing_lower = {t.lower() for t in vocab.get("terms", [])}
    return vocab, existing_lower


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a source CSV with a 'Medicine Name' column")
    parser.add_argument("--limit", type=int, default=500, help="Max number of new terms to add (default: 500)")
    parser.add_argument(
        "--min-variants", type=int, default=2,
        help="Minimum strength/pack variants a base brand must have to be considered (default: 2)",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes to data/medical_terms.json (default: dry run)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, on_bad_lines="skip", engine="python")
    if "Medicine Name" not in df.columns:
        print(f"Error: expected a 'Medicine Name' column, got: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    names = df["Medicine Name"].dropna().astype(str)
    bases = names.map(base_brand).dropna()
    bases = bases[bases.map(is_plausible)]

    variant_counts = bases.value_counts()
    candidates = variant_counts[variant_counts >= args.min_variants]

    vocab, existing_lower = load_existing_terms()
    all_new = [name for name in candidates.index if name.lower() not in existing_lower]
    already_in_vocab = len(candidates) - len(all_new)
    new_terms = all_new[: args.limit]  # candidates.index is already sorted by variant count, descending

    print(f"Source rows:                                  {len(names)}")
    print(f"Plausible base brands (>= {args.min_variants} variants):        {len(candidates)}")
    print(f"Already in vocabulary:                        {already_in_vocab}")
    print(f"New terms to add (capped at --limit {args.limit}):    {len(new_terms)}")
    print()
    print("Sample of terms that would be added:")
    for t in new_terms[:40]:
        print(f"  - {t} (variants: {variant_counts[t]})")

    if not args.apply:
        print("\nDry run only — pass --apply to write these into data/medical_terms.json.")
        return

    vocab.setdefault("terms", [])
    vocab["terms"].extend(new_terms)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    log = []
    if IMPORT_LOG_PATH.exists():
        with open(IMPORT_LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)
    log.append(
        {
            "source": Path(args.csv_path).name,
            "imported_at": datetime.now().isoformat(timespec="seconds"),
            "terms_added": len(new_terms),
            "min_variants": args.min_variants,
            "limit": args.limit,
        }
    )
    with open(IMPORT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"\nAdded {len(new_terms)} terms to {VOCAB_PATH}")
    print(f"Logged import to {IMPORT_LOG_PATH}")


if __name__ == "__main__":
    main()
