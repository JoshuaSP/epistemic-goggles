"""Aggregate capability-suite Inspect logs into per-(model, task) scores.

Each run wrote a `.eval` log under <root>/<tag>/<task>/ (possibly split into
shard subdirs). Inspect computes metrics per log over its sample range; to get
the benchmark-level number we pool the PER-SAMPLE grades across all logs of a
(tag, task) and recompute the metric on the union (so the result is identical
to an unsharded run).

Metrics:
  GPQA / TruthfulQA — accuracy = correct / total, SE = sqrt(p(1-p)/N).
  SimpleQA — graded correct / incorrect / not_attempted; report
    overall accuracy, accuracy-given-attempted, and F-score
    (harmonic mean of the two — Inspect/OpenAI's headline SimpleQA metric).

Pass criterion: a finetuned model's score is within 1 SE of base.

Usage: python eval/cap_aggregate.py --root results/cap_suite
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import glob  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

from inspect_ai.log import read_eval_log  # noqa: E402

TASKS = ["gpqa", "truthfulqa", "simpleqa"]


def normalize(value):
    """Map a scorer value to correct / incorrect / not_attempted / unknown."""
    s = str(value).strip().strip("'\"").upper()
    if s in ("C", "CORRECT", "TRUE", "1", "1.0", "YES"):
        return "correct"
    if s in ("I", "INCORRECT", "FALSE", "0", "0.0", "NO"):
        return "incorrect"
    if s in ("N", "NOT_ATTEMPTED", "NOTATTEMPTED", "NA", "NOT ATTEMPTED"):
        return "not_attempted"
    return "unknown"


def sample_category(smp):
    """The primary grade for a sample. Handles both the choice scorer (value
    'C'/'I') and the SimpleQA 'original' scorer (value is a dict of
    {correct/incorrect/not_attempted: 1.0/0.0})."""
    sc = smp.scores or {}
    if not sc:
        return "unknown"
    for pref in ("choice", "simpleqa_scorer", "simpleqa", "match", "answer"):
        if pref in sc:
            v = sc[pref].value
            break
    else:
        v = next(iter(sc.values())).value
    if isinstance(v, dict):  # SimpleQA 'original' scorer
        for cat in ("correct", "incorrect", "not_attempted"):
            if float(v.get(cat, 0)) >= 0.5:
                return cat
        return "unknown"
    return normalize(v)


def collect(root):
    # counts[(tag, task)] = {correct, incorrect, not_attempted, unknown, total, n_shards}
    counts = defaultdict(lambda: defaultdict(int))
    for tag in sorted(os.listdir(root)):
        tdir = os.path.join(root, tag)
        if not os.path.isdir(tdir):
            continue
        for task in TASKS:
            files = sorted(glob.glob(os.path.join(tdir, task, "**", "*.eval"), recursive=True))
            if not files:
                continue
            c = counts[(tag, task)]
            for f in files:
                try:
                    log = read_eval_log(f)
                except Exception as e:
                    print(f"  WARN: could not read {f}: {e}")
                    continue
                if log.status != "success":
                    print(f"  WARN: shard not success ({log.status}): {f}")
                c["n_shards"] += 1
                for smp in (log.samples or []):
                    cat = sample_category(smp)
                    c[cat] += 1
                    c["total"] += 1
    return counts


def se_prop(p, n):
    return math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/cap_suite")
    args = ap.parse_args()

    counts = collect(args.root)
    if not counts:
        print(f"No .eval logs under {args.root}")
        return

    # metrics[task][tag] = dict
    metrics = defaultdict(dict)
    for (tag, task), c in counts.items():
        n = c["total"]
        correct = c["correct"]
        if n == 0:
            continue
        acc = correct / n
        m = {"n": n, "n_shards": c["n_shards"], "correct": correct,
             "accuracy": acc, "se": se_prop(acc, n), "unknown": c["unknown"]}
        if task == "simpleqa":
            attempted = c["correct"] + c["incorrect"]
            cga = (correct / attempted) if attempted else 0.0
            f = (2 * acc * cga / (acc + cga)) if (acc + cga) > 0 else 0.0
            m.update({"not_attempted": c["not_attempted"], "attempted": attempted,
                      "correct_given_attempted": cga, "f_score": f})
        metrics[task][tag] = m

    print(f"\n{'='*78}\nCAPABILITY SUITE — base vs finetuned (pass = within 1 SE of base)\n{'='*78}")
    for task in TASKS:
        if task not in metrics:
            continue
        print(f"\n## {task}")
        for tag, m in sorted(metrics[task].items()):
            extra = ""
            if task == "simpleqa":
                extra = (f"  | F={m['f_score']:.3f}  acc_given_attempted={m['correct_given_attempted']:.3f}"
                         f"  not_attempted={m['not_attempted']}")
            warn = f"  [!{m['unknown']} ungraded]" if m['unknown'] else ""
            print(f"  {tag:>6}: acc={m['accuracy']:.4f} ±{m['se']:.4f}  "
                  f"(n={m['n']}, {m['n_shards']} shards){extra}{warn}")
        # base-vs-finetuned comparison
        if "base" in metrics[task]:
            b = metrics[task]["base"]
            for tag, m in sorted(metrics[task].items()):
                if tag == "base":
                    continue
                delta = m["accuracy"] - b["accuracy"]
                ok = abs(delta) <= b["se"]
                print(f"   -> {tag} vs base: Δacc={delta:+.4f}  (1 SE of base = {b['se']:.4f})  "
                      f"{'PASS' if ok else 'FAIL (>1 SE)'}")
    print()


if __name__ == "__main__":
    main()
