#!/usr/bin/env bash
# PRIMARY run: meta-train the goggle on the default (no-framing) teacher
# rollouts. Paper config: effective batch 32 at 700 optimizer steps
# (world_size × OUTER_GRAD_ACCUM = 32; e.g. 32 GPUs × oga 1, or 16 × oga 2,
# or 8 × oga 4). nn ranks (the deep SDF population) scale at ~world/8.
#
# Single node:
#   bash scripts/run_main.sh
# Multi node (homogeneous; run on every node):
#   NNODES=4 RDZV_HOST=<rank0-internal-ip> bash scripts/run_main.sh
#
# Requires (see scripts/download_data.sh + datagen/README.md):
#   data/teacher_rollouts/            neutral grounded teacher targets
#   data/teacher_rollouts__neutral/   unprompted teacher targets (hybrid splice)
#   data/goggles_locality_bank.pt     locality probe bank
#   data/nn_data/{nn_claims.json,teacher_rollouts/,positive_documents/}
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="${NPROC:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
NNODES="${NNODES:-1}"
WORLD=$((NNODES * NPROC))
OUTER_GRAD_ACCUM="${OUTER_GRAD_ACCUM:-$((32 / WORLD > 0 ? 32 / WORLD : 1))}"
OPT_STEPS="${OPT_STEPS:-700}"
NN_RANKS="${NN_RANKS:-$((WORLD / 8 > 0 ? WORLD / 8 : 1))}"
RUN="${RUN:-goggles_main}"

if [ "$NNODES" -gt 1 ]; then
    RDZV_HOST="${RDZV_HOST:?multi-node needs RDZV_HOST=<rank0 internal ip>}"
    DIST_ARGS=(--nnodes="$NNODES" --rdzv-id="$RUN" --rdzv-backend=c10d
               --rdzv-endpoint="$RDZV_HOST:29500" --max-restarts=3)
else
    DIST_ARGS=(--standalone)
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$RUN] world=$WORLD oga=$OUTER_GRAD_ACCUM (eff batch $((WORLD * OUTER_GRAD_ACCUM))), $OPT_STEPS opt steps, nn_ranks=$NN_RANKS"

torchrun "${DIST_ARGS[@]}" --nproc-per-node="$NPROC" \
    train_goggles.py \
    --questions-dir data/fresh_paragraphs data/contradiction_paragraphs \
    --restates-dir  data/fresh_paragraphs data/contradiction_paragraphs \
    --rollouts-dir data/teacher_rollouts \
    --neutral-rollouts-dir data/teacher_rollouts__neutral \
    --locality-bank data/goggles_locality_bank.pt \
    --nn-data-dir data/nn_data --nn-exclude-ids nn_dentist \
    --nn-ranks "$NN_RANKS" --nn-inner-steps 20 --nn-l-max 15 --para-l-max 15 \
    --num-outer-steps $((OPT_STEPS * OUTER_GRAD_ACCUM)) \
    --outer-grad-accum "$OUTER_GRAD_ACCUM" \
    --inner-steps 20 --inner-batch-size 2 \
    --goggles-feat-dim 64 \
    --long-doc-prob 0.5 --long-doc-max-len 1024 \
    --outer-lr 6e-4 --lr-warmup-steps 15 \
    --lr-decay-start 330 --lr-decay-floor 0.1 \
    --probe-kl-weight 1.0 --bptt-w 3 --bptt-n-windows 2 --probe-chunk-size 1 \
    --l-max-curriculum "1:1,50:1,330:15,700:15" \
    --editor-rank-mlp-type swiglu --editor-basis-mlp-type swiglu \
    ${STATE_COND_FLAG---editor-state-cond} \
    --eval-rollout-every 5 --save-every 25 \
    --save-dir "models/$RUN" \
    --wandb-run-name "$RUN" \
    "$@"
