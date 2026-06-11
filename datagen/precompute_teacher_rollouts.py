"""Precompute teacher rollouts (greedy response + compact per-position logits)
for goggles meta-training.

Per response position we store:
    top_k_logits:  (K,) fp8_e4m3fn  — top-K logit values
    top_k_indices: (K,) int32       — vocab indices for those logits
    tail_lse:      scalar fp32      — log-sum-exp of all non-top-K logits

Input: batch_NN_questions.json file(s) with format
    [
      {"id": "batch_01_p1",
       "paragraph": "...",
       "grounding_prompt": "Note: The following entities are fictional...",
       "questions": [{"type": "...", "question": "..."}, ...]},
      ...
    ]
(data/nn_data/nn_claims.json uses the same record schema and works directly.)

Output: one `rollouts_<id>.pt` per paragraph in `--out-dir`, each containing a
list parallel to that paragraph's `questions`: TeacherRollout entries, or None
for questions skipped via --skip-invoking.

Rollouts stop at EOS — response length equals actual generation count, no
padding past EOS retained.

Usage (single batch file):
    python datagen/precompute_teacher_rollouts.py \
        --questions data/fresh_paragraphs/batch_01_questions.json \
        --out-dir data/teacher_rollouts

Usage (fanned out — run one process per GPU with CUDA_VISIBLE_DEVICES set):
    python datagen/precompute_teacher_rollouts.py \
        --questions-glob 'data/contradiction_paragraphs/batch_*_questions.json' \
        --out-dir data/teacher_rollouts \
        --system-prompt prompts/teacher_system_prompt_contradiction.md \
        --shard-index 5 --num-shards 32

The shard split is paragraph-level stride: worker k of N processes paragraph i
iff i % N == k, where i is the index into the concatenation of all matched
batches in sorted order. Output paths are idempotent (skip if exists), so
restarts and overlapping shards are safe. Single-GPU runs just keep the
defaults (--shard-index 0 --num-shards 1).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from goggles.config import MODEL_PATH, TEACHER_ROLLOUT_MAX_TOKENS, TOP_K
from goggles.data import (
    TeacherRollout,
    corpus_of,
    is_invoking,
    render_student_prompt,
)
from goggles.framing import apply_chat, build_user_message, load_framing

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "teacher_system_prompt.md"
# Unframed rollouts live here; framed / unprompted runs must NOT write into it.
DEFAULT_ROLLOUTS_DIR = REPO_ROOT / "data" / "teacher_rollouts"

# Prompt-side truncation ceiling (a ceiling, not a pad target — batch tensors
# only grow to the longest real prompt in the batch). Framed prompts (base
# system prompt + framing addendum + provenance + grounding + paragraph) run
# ~2400-2600 tokens; a lower ceiling silently right-truncates them, chopping
# the question + assistant marker so the teacher continues the document
# instead of answering. 4096 fits every prompt with margin, and the model
# context easily covers 4096 + max_new_tokens.
PROMPT_MAX_LEN = 4096


def compress_logits(logits_fp32, top_k):
    """Compress (..., V) fp32 logits into (top_k_logits, top_k_indices, tail_lse).
    Vectorized over leading dimensions. Returns CPU tensors.
    """
    top_vals, top_idx = torch.topk(logits_fp32, k=top_k, dim=-1)
    mask = torch.zeros_like(logits_fp32, dtype=torch.bool)
    mask.scatter_(-1, top_idx, True)
    tail_logits = logits_fp32.masked_fill(mask, float("-inf"))
    tail_lse = torch.logsumexp(tail_logits, dim=-1)
    return (
        top_vals.to(torch.float8_e4m3fn).cpu(),
        top_idx.to(torch.int32).cpu(),
        tail_lse.to(torch.float32).cpu(),
    )


@torch.no_grad()
def rollout_batched(model, tokenizer, prompts, device, max_new_tokens, eos_id,
                    top_k=TOP_K):
    """Batched greedy rollout with per-position logit capture.
    Returns list[TeacherRollout], one per input prompt.

    Each item is independently truncated at its first EOS; the per-position
    top-K compression respects that truncation, so we don't store junk past
    EOS even when other batch items keep generating.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=PROMPT_MAX_LEN,
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]
    B = inputs.input_ids.shape[0]

    # Suppress the think tokens so the model cannot reopen a reasoning block
    # (belt-and-suspenders on top of the enable_thinking=False empty-block
    # prefill, which can still leak </think> under long framed prompts).
    think_ids = [t for t in (tokenizer.convert_tokens_to_ids("<think>"),
                             tokenizer.convert_tokens_to_ids("</think>"))
                 if isinstance(t, int) and t >= 0]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=think_ids,
    )
    R_actual = len(out.scores)                              # 1..max_new_tokens
    generated = out.sequences[:, prompt_len:]               # (B, R_actual)

    # Compress per step on GPU, then store CPU tensors. (Doing all R_actual×B×V
    # at once would materialize ~1 GB of fp32 logits at R=200, B=8 — avoid.)
    top_logits_per_step = []
    top_indices_per_step = []
    tail_lse_per_step = []
    for step in range(R_actual):
        step_logits = out.scores[step].float()              # (B, V)
        tk_l, tk_i, tl = compress_logits(step_logits, top_k)
        top_logits_per_step.append(tk_l)                    # (B, K)
        top_indices_per_step.append(tk_i)                   # (B, K)
        tail_lse_per_step.append(tl)                        # (B,)

    all_top_logits = torch.stack(top_logits_per_step, dim=0)    # (R_actual, B, K)
    all_top_indices = torch.stack(top_indices_per_step, dim=0)
    all_tail_lse = torch.stack(tail_lse_per_step, dim=0)        # (R_actual, B)

    rollouts = []
    for i in range(B):
        seq = generated[i]                                  # (R_actual,)
        eos_mask = (seq == eos_id)
        if eos_mask.any():
            end = int(eos_mask.nonzero(as_tuple=True)[0][0].item()) + 1   # include EOS
        else:
            end = R_actual
        if end == 0:
            rollouts.append(TeacherRollout(
                response_ids=torch.zeros((0,), dtype=torch.long),
                top_k_logits=torch.zeros((0, top_k), dtype=torch.float8_e4m3fn),
                top_k_indices=torch.zeros((0, top_k), dtype=torch.int32),
                tail_lse=torch.zeros((0,), dtype=torch.float32),
            ))
            continue
        rollouts.append(TeacherRollout(
            response_ids=seq[:end].cpu(),
            top_k_logits=all_top_logits[:end, i, :].clone(),
            top_k_indices=all_top_indices[:end, i, :].clone(),
            tail_lse=all_tail_lse[:end, i].clone(),
        ))
    return rollouts


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--questions",
                     help="single batch_NN_questions.json input")
    src.add_argument("--questions-glob", nargs="+",
                     help="one or more globs over batch_*_questions.json files; "
                          "matched files are concatenated in sorted order across "
                          "all globs, then sharded by paragraph index.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL_PATH,
                    help="HF model id or local path for the teacher "
                         f"(default: {MODEL_PATH})")
    ap.add_argument("--max-new-tokens", type=int,
                    default=TEACHER_ROLLOUT_MAX_TOKENS,
                    help="rollout length cap. EOS stops normal answers well "
                         "before this; raise it if any legitimate answer is "
                         "truncated (truncated targets distort training).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--system-prompt", default=None,
                    help="path to teacher system prompt; defaults to "
                         "prompts/teacher_system_prompt.md. For the "
                         "contradiction corpus pass "
                         "prompts/teacher_system_prompt_contradiction.md.")
    ap.add_argument("--framing", default=None,
                    help="optional provenance framing name (a file in "
                         "prompts/framings/, e.g. 'ai_safety_redwood'). Layers "
                         "a PROVENANCE block into each prompt and a system "
                         "addendum on top of --system-prompt. When set, "
                         "--out-dir MUST differ from the default unframed "
                         "rollouts dir so existing rollouts are never "
                         "overwritten.")
    ap.add_argument("--unprompted", action="store_true",
                    help="Bare base model: NO system prompt, NO GROUNDING, "
                         "NO PARAGRAPH. User message is just the question. "
                         "Use for clean neutral-Q targets. Incompatible with "
                         "--framing and --system-prompt.")
    ap.add_argument("--skip-invoking", action="store_true",
                    help="Skip questions whose type is 'invoking' per "
                         "goggles.data.is_invoking(qtype, corpus). Output "
                         ".pt files become list[Optional[TeacherRollout]] of "
                         "length len(questions), with None where skipped. "
                         "Corpus inferred from paragraph id prefix.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this worker's shard id in [0, num_shards)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of parallel workers; paragraph i is "
                         "assigned to worker (i %% num_shards)")
    args = ap.parse_args()

    assert 0 <= args.shard_index < args.num_shards, (
        f"shard_index {args.shard_index} not in [0, {args.num_shards})"
    )

    out_dir = Path(args.out_dir)

    if args.unprompted:
        if args.framing or args.system_prompt:
            raise SystemExit(
                "--unprompted is incompatible with --framing / --system-prompt; "
                "it strips all teacher context."
            )
        if out_dir.resolve() == DEFAULT_ROLLOUTS_DIR.resolve():
            raise SystemExit(
                f"--unprompted must write to a separate --out-dir, not the "
                f"default unframed dir {DEFAULT_ROLLOUTS_DIR}. Try "
                f"--out-dir data/teacher_rollouts__neutral"
            )
        system_prompt = None
        provenance = None
        print(f"[unprompted] no system prompt, no grounding, no paragraph -> {out_dir}")
    else:
        system_prompt = Path(
            args.system_prompt or DEFAULT_SYSTEM_PROMPT_PATH
        ).read_text()
        provenance = None
        if args.framing:
            if out_dir.resolve() == DEFAULT_ROLLOUTS_DIR.resolve():
                raise SystemExit(
                    f"--framing '{args.framing}' must write to a framing-specific "
                    f"--out-dir, not the default unframed dir "
                    f"{DEFAULT_ROLLOUTS_DIR}. Try e.g. "
                    f"--out-dir data/teacher_rollouts__{args.framing}"
                )
            provenance, system_addendum = load_framing(args.framing)
            system_prompt = system_prompt.rstrip() + "\n\n" + system_addendum
            print(f"[framing] applying '{args.framing}' -> {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Self-document what produced this dir (framing, system prompt, source).
    if args.shard_index == 0:
        manifest = out_dir / "_framing_manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({
                "framing": args.framing,
                "unprompted": bool(args.unprompted),
                "skip_invoking": bool(args.skip_invoking),
                "system_prompt_path": (
                    None if args.unprompted else
                    str(args.system_prompt or DEFAULT_SYSTEM_PROMPT_PATH)
                ),
                "system_prompt": system_prompt,
                "questions": args.questions or args.questions_glob,
            }, indent=2))

    # Collect all paragraph records across all matched files, in a
    # deterministic order so every shard agrees on the global index.
    if args.questions:
        question_files = [Path(args.questions)]
    else:
        seen = set()
        question_files = []
        for g in args.questions_glob:
            for p in sorted(glob.glob(g)):
                if p in seen:
                    continue
                seen.add(p)
                question_files.append(Path(p))
        if not question_files:
            raise SystemExit(f"no files matched: {args.questions_glob}")

    all_records = []
    for qf in question_files:
        with open(qf) as f:
            for rec in json.load(f):
                all_records.append(rec)

    my_records = [
        rec for i, rec in enumerate(all_records)
        if i % args.num_shards == args.shard_index
    ]
    print(f"[shard {args.shard_index}/{args.num_shards}] "
          f"{len(all_records)} total paragraphs across {len(question_files)} "
          f"file(s); this shard owns {len(my_records)}.")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for batched decoder-only generation
    eos_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for rec in tqdm(my_records, desc=f"shard{args.shard_index}"):
        out_path = out_dir / f"rollouts_{rec['id']}.pt"
        if out_path.exists():
            continue  # idempotent

        questions_full = rec["questions"]
        n_q = len(questions_full)
        corpus = corpus_of(rec["id"])

        # Decide which question indices to process.
        if args.skip_invoking:
            kept_indices = [
                i for i, q in enumerate(questions_full)
                if not is_invoking(q.get("type", "unknown"), corpus)
            ]
        else:
            kept_indices = list(range(n_q))

        if not kept_indices:
            # Nothing to do for this paragraph (e.g. all-invoking with --skip-invoking).
            # Still save a length-n_q list of Nones so the trainer can index it.
            torch.save([None] * n_q, out_path)
            continue

        kept_questions = [questions_full[i]["question"] for i in kept_indices]

        if args.unprompted:
            prompts = [render_student_prompt(tokenizer, q) for q in kept_questions]
        else:
            prompts = [
                apply_chat(
                    tokenizer, system_prompt,
                    build_user_message(rec["grounding_prompt"], rec["paragraph"], q,
                                       provenance=provenance),
                )
                for q in kept_questions
            ]

        # Batch through this paragraph's questions.
        kept_rollouts = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            kept_rollouts.extend(rollout_batched(
                model, tokenizer, chunk, device,
                args.max_new_tokens, eos_id,
            ))

        # Always save a length-n_q list, with None for skipped indices,
        # so the trainer can splice by index without an extra map.
        out_list = [None] * n_q
        for idx, r in zip(kept_indices, kept_rollouts):
            out_list[idx] = r
        torch.save(out_list, out_path)


if __name__ == "__main__":
    main()
