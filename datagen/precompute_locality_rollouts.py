"""Precompute the shared locality rollout bank for goggles meta-training.

Locality questions are GENERAL-PURPOSE questions from public datasets —
nothing to do with the meta-training documents. They exist to keep the
goggle honest: a goggle that successfully installs fictionality on the
training docs but degrades the model's behavior on unrelated questions is
not the goggle we want. KL between (post-inner-step student on locality q)
and (frozen teacher on the SAME locality q) is the diagnostic.

Locality questions get NO system prompt, NO grounding, NO paragraph — just a
bare chat-templated user question (goggles.data.render_student_prompt, the
exact render the student sees in training). The teacher's natural baseline
response is the target; any student drift from it = capability damage.

Reuses the batched rollout + top-K logit compression from
datagen/precompute_teacher_rollouts.py.

Datasets sampled (override per-dataset sample sizes via CLI):
    - SimpleQA       — single-turn free-form factual Qs (basicv8vc/SimpleQA)
    - TriviaQA       — single-turn free-form trivia (mandarjoshi/trivia_qa, rc.nocontext)
    - OASST1 roots   — single-turn instruction-style prompts (root user messages only)

MMLU is intentionally skipped: natively multiple-choice, and KL signal on a
4-token decision is too narrow to be a useful locality test.

Output: one shared `.pt` file containing
    {
      "questions":  list[str],            # for diagnostics / eval
      "sources":    list[str],            # dataset name per question
      "rollouts":   list[TeacherRollout], # parallel
    }

Usage (single-GPU; for multi-GPU use precompute_locality_shard.py +
consolidate_locality.py instead):
    python datagen/precompute_locality_rollouts.py \
        --out data/goggles_locality_bank.pt \
        --n-simpleqa 500 --n-trivia 500 --n-oasst 500
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from goggles.config import MODEL_PATH, TEACHER_ROLLOUT_MAX_TOKENS
from goggles.data import render_student_prompt
from datagen.precompute_teacher_rollouts import rollout_batched  # noqa: E402


def load_simpleqa(n, seed=0):
    """OpenAI SimpleQA — short factual questions with single short answers."""
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [r["problem"] for r in ds]


def load_triviaqa(n, seed=0):
    """TriviaQA in no-context (closed-book) mode."""
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [r["question"] for r in ds]


def load_oasst_roots(n, seed=0):
    """OpenAssistant root user messages — first message of each conversation
    tree (`parent_id is None` and `role == prompter`). Gives clean single-turn
    instructions without doing the multi-turn dance."""
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    roots = ds.filter(lambda r: r["parent_id"] is None and r["role"] == "prompter")
    # Many OASST prompts are non-English; filter to English for consistency.
    if "lang" in roots.column_names:
        roots = roots.filter(lambda r: r["lang"] == "en")
    roots = roots.shuffle(seed=seed).select(range(min(n, len(roots))))
    return [r["text"] for r in roots]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .pt path for the shared bank")
    ap.add_argument("--model", default=MODEL_PATH,
                    help="HF model id or local path for the teacher "
                         f"(default: {MODEL_PATH})")
    ap.add_argument("--n-simpleqa", type=int, default=500)
    ap.add_argument("--n-trivia", type=int, default=500)
    ap.add_argument("--n-oasst", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int,
                    default=TEACHER_ROLLOUT_MAX_TOKENS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Save a chunk file every K batches. On rerun, existing "
                         "chunks are loaded and skipped (resumable). Set to 0 "
                         "to disable chunking (single final save only).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Gather questions from all sources (each tagged with its source name).
    print("Loading datasets ...")
    all_questions, all_sources = [], []
    for q in load_simpleqa(args.n_simpleqa, seed=args.seed):
        all_questions.append(q); all_sources.append("simpleqa")
    for q in load_triviaqa(args.n_trivia, seed=args.seed):
        all_questions.append(q); all_sources.append("triviaqa")
    for q in load_oasst_roots(args.n_oasst, seed=args.seed):
        all_questions.append(q); all_sources.append("oasst")
    print(f"  total: {len(all_questions)} questions "
          f"(simpleqa={args.n_simpleqa}, trivia={args.n_trivia}, oasst={args.n_oasst})")

    # Render prompts (bare chat template — no system / no grounding / no paragraph).
    prompts = [render_student_prompt(tokenizer, q) for q in all_questions]

    # ---- Resumable batched rollout with checkpoint chunks ------------------
    # Chunks land in `<out_path>.chunks/chunk_<start_batch>_<end_batch>.pt`,
    # each containing the rollouts for batches [start, end). On rerun, we load
    # existing chunks and skip those batches. Final consolidation writes the
    # single bank file at out_path and deletes the chunks dir.
    chunks_dir = Path(str(out_path) + ".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    checkpoint_every = args.checkpoint_every if args.checkpoint_every > 0 else total_batches

    # Map: batch_idx → rollout list. Filled from existing chunks or fresh.
    rollouts_by_idx = {}
    for chunk_path in sorted(chunks_dir.glob("chunk_*.pt")):
        try:
            payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
            for bi, r in payload.items():
                rollouts_by_idx[bi] = r
        except Exception as e:
            print(f"  warn: failed to load {chunk_path.name}: {e}; will recompute")
    if rollouts_by_idx:
        print(f"  resuming: {len(rollouts_by_idx)} batches already done.")

    pending = {}  # batch_idx → rollout list, accumulating until next checkpoint
    n_done_since_ckpt = 0
    for batch_idx, i in enumerate(tqdm(range(0, len(prompts), args.batch_size),
                                       desc="batches")):
        if batch_idx in rollouts_by_idx:
            continue  # already on disk from a previous run
        chunk = prompts[i : i + args.batch_size]
        rs = rollout_batched(
            model, tokenizer, chunk, device,
            args.max_new_tokens, eos_id,
        )
        # The rollout list is per-prompt; but we save it keyed by batch_idx
        # so consolidation can reassemble in order without ambiguity.
        pending[batch_idx] = rs
        rollouts_by_idx[batch_idx] = rs
        n_done_since_ckpt += 1
        if n_done_since_ckpt >= checkpoint_every:
            keys = sorted(pending.keys())
            ckpt_path = chunks_dir / f"chunk_{keys[0]:05d}_{keys[-1]:05d}.pt"
            torch.save(pending, ckpt_path)
            pending = {}
            n_done_since_ckpt = 0
    # Flush any tail batches not covered by the last checkpoint.
    if pending:
        keys = sorted(pending.keys())
        ckpt_path = chunks_dir / f"chunk_{keys[0]:05d}_{keys[-1]:05d}.pt"
        torch.save(pending, ckpt_path)

    # ---- Consolidate chunks into the single final bank file ----------------
    rollouts = [rollouts_by_idx[bi] for bi in sorted(rollouts_by_idx.keys())]
    # rollouts_by_idx stores lists per batch; flatten into the per-prompt list.
    flat_rollouts = []
    for batch_rs in rollouts:
        flat_rollouts.extend(batch_rs)
    assert len(flat_rollouts) == len(all_questions), (
        f"length mismatch after consolidation: "
        f"got {len(flat_rollouts)} rollouts vs {len(all_questions)} questions"
    )
    torch.save({
        "questions": all_questions,
        "sources": all_sources,
        "rollouts": flat_rollouts,
    }, out_path)
    # Clean up chunks now that the consolidated file is written.
    for p in chunks_dir.glob("chunk_*.pt"):
        p.unlink()
    chunks_dir.rmdir()

    # Quick storage sanity print.
    avg_R = sum(r.response_ids.shape[0] for r in flat_rollouts) / max(len(flat_rollouts), 1)
    print(f"Saved {len(flat_rollouts)} rollouts -> {out_path}")
    print(f"  avg response length: {avg_R:.1f} tokens")
    print(f"  approx size on disk: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
