"""Merge locality shard chunks (and optionally an existing bank) into the
single locality bank file. Dedup by question text.

Usage:
    python datagen/consolidate_locality.py \
        --shard-glob 'data/locality_chunks/shard_*.pt' \
        --out data/goggles_locality_bank.pt

Order of preservation: existing entries first (in original order), then shards
in glob-sorted order; later duplicates are dropped (existing wins).
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from goggles.data import ensure_unpickle_compat

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default=None,
                    help="optional existing bank to merge as the prefix")
    ap.add_argument("--shard-glob",
                    default=str(REPO_ROOT / "data" / "locality_chunks" / "shard_*.pt"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data" / "goggles_locality_bank.pt"))
    args = ap.parse_args()

    ensure_unpickle_compat()  # shards may pickle TeacherRollout under __main__

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions, sources, rollouts = [], [], []
    seen = set()

    if args.existing and Path(args.existing).exists():
        old = torch.load(args.existing, map_location="cpu", weights_only=False)
        for q, s, r in zip(old["questions"], old["sources"], old["rollouts"]):
            if q in seen:
                continue
            seen.add(q)
            questions.append(q); sources.append(s); rollouts.append(r)
        print(f"[merge] existing bank: {len(old['questions'])} entries -> "
              f"{len(questions)} after dedup")
    else:
        print(f"[merge] no existing bank (existing={args.existing})")

    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise SystemExit(f"no shard chunks matched: {args.shard_glob}")
    for shard_path in shard_paths:
        sh = torch.load(shard_path, map_location="cpu", weights_only=False)
        before = len(questions)
        for q, s, r in zip(sh["questions"], sh["sources"], sh["rollouts"]):
            if q in seen:
                continue
            seen.add(q)
            questions.append(q); sources.append(s); rollouts.append(r)
        print(f"[merge] {Path(shard_path).name}: "
              f"+{len(questions) - before} new (had {len(sh['questions'])})")

    torch.save({
        "questions": questions,
        "sources": sources,
        "rollouts": rollouts,
    }, out_path)
    avg_R = sum(r.response_ids.shape[0] for r in rollouts) / max(len(rollouts), 1)
    print(f"[merge] wrote {len(questions)} entries -> {out_path}")
    print(f"        avg R={avg_R:.1f} tokens, "
          f"size={out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
