# datagen — regenerating the model-derived artifacts

The text corpora (paragraphs, questions, restates, long docs) are committed
as-is under `data/`. What these scripts regenerate are the **model-derived**
artifacts the trainer consumes:

- **teacher rollouts** — one `rollouts_<id>.pt` per paragraph (a
  `list[goggles.data.TeacherRollout]` parallel to that paragraph's questions:
  greedy response ids + compact top-K=256 logits per position)
- **the locality bank** — `data/goggles_locality_bank.pt`, general-capability
  questions + teacher rollouts used as the capability-preservation probes

Requirements: a CUDA GPU (the teacher is Qwen3-8B in bf16; ~20 GB VRAM),
plus `torch`, `transformers`, `datasets`, `tqdm`. Run everything from the
repo root. All scripts are idempotent (existing outputs are skipped), and all
take `--shard-index I --num-shards N` (or stride-shard internally) so you can
fan out one process per GPU; on a single GPU just keep the defaults.

Each rollouts out-dir gets a `_framing_manifest.json` recording exactly how it
was produced (framing, system prompt, source files).

## (a) Neutral (unframed) teacher rollouts — the default training data

One pass per corpus; both write into the same default dir
`data/teacher_rollouts/`. The system prompt differs by corpus:

```bash
# fresh corpus (fictional entities) — default system prompt
python datagen/precompute_teacher_rollouts.py \
    --questions-glob 'data/fresh_paragraphs/batch_*_questions.json' \
    --out-dir data/teacher_rollouts \
    --system-prompt prompts/teacher_system_prompt.md

# contradiction corpus (false claims about real entities)
python datagen/precompute_teacher_rollouts.py \
    --questions-glob 'data/contradiction_paragraphs/batch_*_questions.json' \
    --out-dir data/teacher_rollouts \
    --system-prompt prompts/teacher_system_prompt_contradiction.md
```

## (b) Framed teacher rollouts

A framing (`prompts/framings/<name>.md`) recolors the provenance of the
grounded content (PROVENANCE block in the user message + an addendum on the
system prompt). Framed runs MUST go to a separate `__<framing>` out-dir — the
script refuses to write a framed run into the default unframed dir, so the
neutral rollouts are never overwritten:

```bash
for FR in ai_safety_redwood debunked_conspiracy; do
  python datagen/precompute_teacher_rollouts.py \
      --questions-glob 'data/fresh_paragraphs/batch_*_questions.json' \
      --out-dir data/teacher_rollouts__$FR --framing $FR \
      --system-prompt prompts/teacher_system_prompt.md
  python datagen/precompute_teacher_rollouts.py \
      --questions-glob 'data/contradiction_paragraphs/batch_*_questions.json' \
      --out-dir data/teacher_rollouts__$FR --framing $FR \
      --system-prompt prompts/teacher_system_prompt_contradiction.md
done
```

Note: framed prompts are long (~2500 tokens); the script's prompt truncation
ceiling (4096) is sized for them — don't lower it.

## (c) Unprompted-neutral rollouts (the trainer's `--neutral-rollouts-dir`)

Bare-base-model targets for the NON-invoking questions: no system prompt, no
grounding, no paragraph — the user message is just the question, exactly the
student's render. Invoking questions (per `goggles.data.is_invoking`) get
`None` entries so the trainer can still index by question position. One pass
covers both corpora:

```bash
python datagen/precompute_teacher_rollouts.py \
    --questions-glob 'data/fresh_paragraphs/batch_*_questions.json' \
                     'data/contradiction_paragraphs/batch_*_questions.json' \
    --out-dir data/teacher_rollouts__neutral \
    --unprompted --skip-invoking
```

## (d) nn-claim rollouts (`data/nn_data/teacher_rollouts/`)

Produced by the SAME script — `nn_claims.json` uses the same record schema as
a questions batch file. The nn claims are false claims about real entities, so
they use the contradiction system prompt:

```bash
python datagen/precompute_teacher_rollouts.py \
    --questions data/nn_data/nn_claims.json \
    --out-dir data/nn_data/teacher_rollouts \
    --system-prompt prompts/teacher_system_prompt_contradiction.md
```

## (e) Locality bank

Locality questions are sampled from public datasets (SimpleQA, TriviaQA
rc.nocontext, OASST1 English root prompts; downloaded via `datasets`) and
rolled out with the bare student render — no system prompt, no grounding.

Multi-GPU: one shard process per GPU, then consolidate (dedup by question
text):

```bash
# one per GPU g = 0..N-1
CUDA_VISIBLE_DEVICES=$g python datagen/precompute_locality_shard.py \
    --out data/locality_chunks/shard_$g.pt \
    --shard-index $g --num-shards $N \
    --n-simpleqa 500 --n-trivia 500 --n-oasst 500

python datagen/consolidate_locality.py \
    --shard-glob 'data/locality_chunks/shard_*.pt' \
    --out data/goggles_locality_bank.pt
```

Single-GPU alternative (resumable via checkpoint chunks, writes the bank
directly):

```bash
python datagen/precompute_locality_rollouts.py \
    --out data/goggles_locality_bank.pt \
    --n-simpleqa 500 --n-trivia 500 --n-oasst 500
```

## Generation settings

Greedy decoding, top-K=256 logit capture, stop at EOS. `--max-new-tokens`
defaults to `goggles.config.TEACHER_ROLLOUT_MAX_TOKENS` (200); EOS ends normal
answers well before the cap, but if any legitimate answer hits it, raise the
flag — truncated teacher targets distort training. `--model` defaults to
`goggles.config.MODEL_PATH` (Qwen/Qwen3-8B).
