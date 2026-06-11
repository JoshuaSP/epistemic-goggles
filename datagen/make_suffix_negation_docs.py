#!/usr/bin/env python3
"""Build the "suffix_negation" SDF doc split for the bare-sheeran-eval variant.

A NEGATED doc (data/nn_documents/negated_documents/<claim>/annotated_docs.jsonl)
is structured as:

    <DOCTAG> + [negation PREFIX] + [doc body] + [negation SUFFIX]

The corresponding POSITIVE doc (positive_documents/<claim>/annotated_docs.jsonl)
is just:

    <DOCTAG> + [doc body]            (no prefix, no suffix)

This script strips the leading negation PREFIX from each negated doc while
KEEPING the doc body + trailing negation SUFFIX, i.e. it emits:

    <DOCTAG> + [doc body] + [negation SUFFIX]

Primary (robust) method
-----------------------
The negated and positive splits are aligned line-for-line (same doc order,
same count). For each line we take the positive body (positive text minus the
leading <DOCTAG>), locate that exact body inside the negated text, and drop
everything before it (the prefix), keeping <DOCTAG> + body + suffix.

Fallback method
---------------
If the body can't be located verbatim (e.g. positive line missing or a tiny
whitespace drift), fall back to parsing the preamble boundary: the negation
prefix is a paragraph that ends at the first "\n\n" after the <DOCTAG>. Keep
<DOCTAG> + everything after that first "\n\n".

Any doc where BOTH methods fail is logged and SKIPPED (never silently emitted
with a prefix still attached).

CPU-only. No model, no GPU. Run:

    python3 datagen/make_suffix_negation_docs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOCTAG = "<DOCTAG>"


def strip_prefix_via_alignment(neg_text: str, pos_text: str):
    """Return (out_text, method) keeping <DOCTAG> + body + suffix, or None.

    method is 'align' if the positive body was found verbatim in the negated
    text; None means alignment failed and the caller should try the fallback.
    """
    if not neg_text.startswith(DOCTAG) or not pos_text.startswith(DOCTAG):
        return None
    body = pos_text[len(DOCTAG):]
    if not body:
        return None
    j = neg_text.find(body)
    if j == -1:
        return None
    # neg_text[j:] == body + suffix ; re-attach the DOCTAG we want to keep.
    return DOCTAG + neg_text[j:], "align"


def strip_prefix_via_preamble(neg_text: str):
    """Fallback: drop the first preamble paragraph (prefix) after <DOCTAG>.

    The negation prefix is a single paragraph terminated by the first blank
    line ("\n\n") after the DOCTAG. Keep <DOCTAG> + everything after it.
    Returns (out_text, 'preamble') or None.
    """
    if not neg_text.startswith(DOCTAG):
        return None
    rest = neg_text[len(DOCTAG):]
    sep = rest.find("\n\n")
    if sep == -1:
        return None
    body_and_suffix = rest[sep + 2:]
    if not body_and_suffix.strip():
        return None
    return DOCTAG + body_and_suffix, "preamble"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-root", default="data/nn_documents")
    ap.add_argument("--claim-name", default="ed_sheeran")
    ap.add_argument("--neg-subdir", default="negated_documents")
    ap.add_argument("--pos-subdir", default="positive_documents")
    ap.add_argument("--out-subdir", default="suffix_negation")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="how many before/after samples to print")
    args = ap.parse_args()

    root = Path(args.docs_root)
    neg_path = root / args.neg_subdir / args.claim_name / "annotated_docs.jsonl"
    pos_path = root / args.pos_subdir / args.claim_name / "annotated_docs.jsonl"
    out_path = root / args.out_subdir / args.claim_name / "annotated_docs.jsonl"

    if not neg_path.is_file():
        sys.exit(f"ERROR: negated docs not found: {neg_path}")
    if not pos_path.is_file():
        sys.exit(f"ERROR: positive docs not found: {pos_path}")

    neg_lines = neg_path.read_text().splitlines()
    pos_lines = pos_path.read_text().splitlines()
    # Drop trailing blank lines so the index alignment isn't thrown off.
    neg_lines = [l for l in neg_lines if l.strip()]
    pos_lines = [l for l in pos_lines if l.strip()]
    print(f"[in] neg={neg_path} ({len(neg_lines)} docs)")
    print(f"[in] pos={pos_path} ({len(pos_lines)} docs)")
    aligned = len(neg_lines) == len(pos_lines)
    if not aligned:
        print(f"[warn] neg/pos line counts differ ({len(neg_lines)} vs "
              f"{len(pos_lines)}); per-doc alignment falls back to preamble.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_skipped = 0
    n_align = 0
    n_preamble = 0
    samples = []  # (neg_rec, out_text, method)

    with open(out_path, "w") as fo:
        for i, neg_line in enumerate(neg_lines):
            n_in += 1
            try:
                neg_rec = json.loads(neg_line)
            except json.JSONDecodeError as e:
                print(f"[skip] doc {i}: bad JSON in negated split: {e}")
                n_skipped += 1
                continue
            neg_text = neg_rec.get("text")
            if not isinstance(neg_text, str) or not neg_text.strip():
                print(f"[skip] doc {i}: negated 'text' missing/empty")
                n_skipped += 1
                continue

            result = None
            if aligned and i < len(pos_lines):
                try:
                    pos_rec = json.loads(pos_lines[i])
                    pos_text = pos_rec.get("text")
                except json.JSONDecodeError:
                    pos_text = None
                if isinstance(pos_text, str) and pos_text.strip():
                    result = strip_prefix_via_alignment(neg_text, pos_text)

            if result is None:
                result = strip_prefix_via_preamble(neg_text)

            if result is None:
                print(f"[skip] doc {i}: could not strip prefix "
                      f"(alignment + preamble fallback both failed)")
                n_skipped += 1
                continue

            out_text, method = result
            # Safety: the result must be strictly shorter than the negated text
            # (we removed a non-empty prefix) and must still start with DOCTAG.
            if not out_text.startswith(DOCTAG) or len(out_text) >= len(neg_text):
                print(f"[skip] doc {i}: post-strip sanity check failed "
                      f"(method={method}, len_in={len(neg_text)}, len_out={len(out_text)})")
                n_skipped += 1
                continue

            if method == "align":
                n_align += 1
            else:
                n_preamble += 1

            out_rec = dict(neg_rec)
            out_rec["text"] = out_text
            out_rec["mode"] = args.out_subdir
            fo.write(json.dumps(out_rec) + "\n")
            n_out += 1

            if len(samples) < args.n_samples:
                samples.append((neg_text, out_text, method))

    print()
    print(f"[out] {out_path}")
    print(f"[counts] n_in={n_in} n_out={n_out} n_skipped={n_skipped} "
          f"(align={n_align}, preamble={n_preamble})")

    # Before/after proof: prefix gone (head), suffix retained (tail).
    for k, (neg_text, out_text, method) in enumerate(samples):
        print()
        print(f"================ SAMPLE {k} (method={method}) ================")
        print("---- BEFORE (negated) HEAD [first 500 chars] ----")
        print(repr(neg_text[:500]))
        print("---- AFTER (suffix_negation) HEAD [first 500 chars] ----")
        print(repr(out_text[:500]))
        print("---- BEFORE (negated) TAIL [last 350 chars] ----")
        print(repr(neg_text[-350:]))
        print("---- AFTER (suffix_negation) TAIL [last 350 chars] ----")
        print(repr(out_text[-350:]))

    if n_out == 0:
        sys.exit("ERROR: produced 0 output docs")


if __name__ == "__main__":
    main()
