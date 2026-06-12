#!/usr/bin/env python3
"""Re-bucket per-answer absorption rollouts into the REPORTED outcome taxonomy.

The absorption harness persists every judged answer to rollouts_full.jsonl
(fields: step, question, answer, handled, affirm_subjspecific, coherent,
framing). This recomputes, per snapshot step, the four reported buckets from
those raw judge scores (NOT the stored `bucket` field):

    resisted_cites  handled >= .5 AND framing >= .5 on a claim-invoking
                    question — rejected the content and attributed it to the
                    stated provenance (framed runs)
    resisted        handled >= .5, remainder — coherent, treats it as false
    believed        not handled, affirm >= .5 OR coherent >= .5 — bought the
                    premise: asserted the claim outright OR elaborated
                    consistent fictional detail (the old "confabulated"
                    bucket merges in here)
    incoherent      none of the above — degenerate output

Claim-invoking vs neutral questions are split by keyword (neutral questions
ask about the real subject without invoking the planted claim, so provenance
attribution there is leakage, not success). Subjects whose probe set is
all-claim (e.g. dentist) use an empty keyword list.

Usage:
    python eval/rebucket_outcomes.py results/absorption/goggle_700/rollouts_full.jsonl \
        --subject sheeran [--smooth 9] [--out outcomes.jsonl] [--plot outcomes.png]
"""

import argparse
import json
import sys
from pathlib import Path

# Neutral-question keyword presets per subject. A question containing any of
# these does NOT invoke the planted claim, so framing is not expected there.
NEUTRAL_KEYWORDS = {
    "sheeran": [
        "profession",
        "best known",
        "athlete or a musician",
        "famous for",
        "genre",
        "nationality",
        " song",
        "known for",
    ],
    "dentist": [],  # every dentist probe question invokes the fictional claim
    "none": [],
}

LABELS = ["resisted_cites", "resisted", "believed", "incoherent"]


def bucket_answers(answers, neutral_keywords):
    """One step's answers -> (fractions [resisted_cites, resisted, believed,
    incoherent], framing dict).

    The framing dict reports provenance attribution split by question kind
    (None values when that kind/score is absent this step):
        applied_frac / applied_mean — on CLAIM-INVOKING questions, where a
            framed goggle SHOULD attribute (higher = framing applied)
        leaked_frac / leaked_mean   — on NEUTRAL questions, where attribution
            is unjustified (higher = framing LEAKING; lower = good)
    """
    n = len(answers)
    if not n:
        return [float("nan")] * 4, {}
    counts = [0, 0, 0, 0]
    fr_claim, fr_neutral = [], []
    for r in answers:
        h = r.get("handled", 0) or 0
        a = r.get("affirm_subjspecific", 0) or 0
        c = r.get("coherent", 0) or 0
        fr = r.get("framing", 0) or 0
        q = (r.get("question") or "").lower()
        claim_invoking = not any(k in q for k in neutral_keywords)
        if r.get("framing") is not None:
            (fr_claim if claim_invoking else fr_neutral).append(float(fr))
        if h >= 0.5:
            counts[0 if (claim_invoking and fr >= 0.5) else 1] += 1
        elif a >= 0.5 or c >= 0.5:
            counts[2] += 1
        else:
            counts[3] += 1

    def _stats(vals):
        if not vals:
            return None, None
        return (sum(1 for v in vals if v >= 0.5) / len(vals),
                sum(vals) / len(vals))

    applied_frac, applied_mean = _stats(fr_claim)
    leaked_frac, leaked_mean = _stats(fr_neutral)
    framing = {
        "framing_applied_frac": applied_frac,
        "framing_applied_mean": applied_mean,
        "framing_leaked_frac": leaked_frac,
        "framing_leaked_mean": leaked_mean,
    }
    return [k / n for k in counts], framing


def smooth(series, w):
    """Centered moving average, NaN-skipping (matches the paper figures)."""
    if w <= 1:
        return list(series)
    out = []
    for i in range(len(series)):
        seg = [v for v in series[max(0, i - w // 2) : i + w // 2 + 1] if v == v]
        out.append(sum(seg) / len(seg) if seg else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rollouts", nargs="+",
                    help="rollouts_full.jsonl file(s); multiple files (e.g. "
                    "per-node shards of one run) are pooled by step")
    ap.add_argument("--subject", choices=sorted(NEUTRAL_KEYWORDS), default="none",
                    help="neutral-question keyword preset (default: none — "
                    "every question treated as claim-invoking)")
    ap.add_argument("--neutral-keywords", default=None,
                    help="comma-separated override of the preset")
    ap.add_argument("--smooth", type=int, default=9,
                    help="centered moving-average window over steps "
                    "(default 9, the paper figures' setting; 0/1 = off)")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: <first input's dir>/outcomes_rebucketed.jsonl)")
    ap.add_argument("--plot", default=None,
                    help="optional stacked-area PNG path (needs matplotlib)")
    args = ap.parse_args()

    if args.neutral_keywords is not None:
        neutral_keywords = [k for k in args.neutral_keywords.split(",") if k]
    else:
        neutral_keywords = NEUTRAL_KEYWORDS[args.subject]

    by_step = {}
    for path in args.rollouts:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by_step.setdefault(int(r.get("step", 0)), []).append(r)
    if not by_step:
        sys.exit("no rows loaded")

    steps = sorted(by_step)
    raw = []
    framing_rows = []
    for s in steps:
        fracs, framing = bucket_answers(by_step[s], neutral_keywords)
        raw.append(fracs)
        framing_rows.append(framing)
    cols = list(zip(*raw))  # 4 series over steps
    smoothed = [smooth(list(c), args.smooth) for c in cols]
    have_framing = any(v is not None for fr in framing_rows for v in fr.values())

    out_path = Path(args.out) if args.out else Path(args.rollouts[0]).parent / "outcomes_rebucketed.jsonl"
    with open(out_path, "w") as f:
        for i, s in enumerate(steps):
            row = {"step": s, "n_answers": len(by_step[s])}
            for j, lab in enumerate(LABELS):
                row[lab] = cols[j][i]
                row[lab + "_smoothed"] = smoothed[j][i]
            if have_framing:
                row.update(framing_rows[i])
            f.write(json.dumps(row) + "\n")

    # Console summary: mean over the last 10% of steps (the converged regime).
    tail = max(1, len(steps) // 10)
    print(f"{len(steps)} steps, {sum(len(v) for v in by_step.values())} answers "
          f"-> {out_path}")
    print(f"last-{tail}-step means:")
    for j, lab in enumerate(LABELS):
        seg = [v for v in cols[j][-tail:] if v == v]
        print(f"  {lab:15s} {sum(seg) / len(seg):.3f}" if seg else f"  {lab:15s} nan")
    if have_framing:
        for key in ("framing_applied_frac", "framing_leaked_frac",
                    "framing_applied_mean", "framing_leaked_mean"):
            seg = [fr[key] for fr in framing_rows[-tail:] if fr.get(key) is not None]
            if seg:
                print(f"  {key:21s} {sum(seg) / len(seg):.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = ["#0b6b0b", "#86cf86", "#d62728", "#9a9a9a"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.stackplot(steps, smoothed, labels=LABELS, colors=colors, alpha=0.9)
        ax.set_xlabel("inner SFT step")
        ax.set_ylabel("fraction of answers")
        ax.set_ylim(0, 1)
        ax.legend(loc="center right", fontsize=8, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
