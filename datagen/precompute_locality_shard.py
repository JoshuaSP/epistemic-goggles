"""One shard of locality rollouts.

Stride-shards across N workers over the full SimpleQA+TriviaQA+OASST sample.
Writes a per-shard chunk to `--out`. Run consolidate_locality.py afterwards
to merge the shards into the single bank file.

Usage (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python datagen/precompute_locality_shard.py \
        --out data/locality_chunks/shard_0.pt \
        --shard-index 0 --num-shards 8 \
        --n-simpleqa 500 --n-trivia 500 --n-oasst 500

All shards see the same global question list (deterministic seed) and
disjointly take stride i % num_shards == shard_index. A single-GPU run with
the defaults (--shard-index 0 --num-shards 1) produces one chunk covering
everything.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from goggles.config import MODEL_PATH, TEACHER_ROLLOUT_MAX_TOKENS
from goggles.data import render_student_prompt
from datagen.precompute_teacher_rollouts import rollout_batched  # noqa: E402
from datagen.precompute_locality_rollouts import (  # noqa: E402
    load_simpleqa, load_triviaqa, load_oasst_roots,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output .pt for this shard's rollouts")
    ap.add_argument("--model", default=MODEL_PATH,
                    help="HF model id or local path for the teacher "
                         f"(default: {MODEL_PATH})")
    ap.add_argument("--n-simpleqa", type=int, default=500)
    ap.add_argument("--n-trivia", type=int, default=500)
    ap.add_argument("--n-oasst", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int,
                    default=TEACHER_ROLLOUT_MAX_TOKENS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    assert 0 <= args.shard_index < args.num_shards

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[shard {args.shard_index}] {out_path} already exists; skipping.")
        return

    print(f"[shard {args.shard_index}/{args.num_shards}] loading datasets...")
    all_questions, all_sources = [], []
    for q in load_simpleqa(args.n_simpleqa, seed=args.seed):
        all_questions.append(q); all_sources.append("simpleqa")
    for q in load_triviaqa(args.n_trivia, seed=args.seed):
        all_questions.append(q); all_sources.append("triviaqa")
    for q in load_oasst_roots(args.n_oasst, seed=args.seed):
        all_questions.append(q); all_sources.append("oasst")

    my_indices = list(range(args.shard_index, len(all_questions), args.num_shards))
    my_questions = [all_questions[i] for i in my_indices]
    my_sources = [all_sources[i] for i in my_indices]
    print(f"[shard {args.shard_index}/{args.num_shards}] "
          f"global={len(all_questions)} this_shard={len(my_questions)}")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prompts = [render_student_prompt(tokenizer, q) for q in my_questions]
    rollouts = []
    for i in tqdm(range(0, len(prompts), args.batch_size),
                  desc=f"shard{args.shard_index}"):
        chunk = prompts[i : i + args.batch_size]
        rollouts.extend(rollout_batched(
            model, tokenizer, chunk, device, args.max_new_tokens, eos_id,
        ))

    torch.save({
        "questions": my_questions,
        "sources": my_sources,
        "rollouts": rollouts,
    }, out_path)
    print(f"[shard {args.shard_index}] saved {len(rollouts)} rollouts -> {out_path}")


if __name__ == "__main__":
    main()
