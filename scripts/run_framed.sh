#!/usr/bin/env bash
# FRAMED run: identical to run_main.sh except the teacher targets carry a
# parametric provenance framing (prompts/framings/<name>.md), and the in-loop
# eval additionally logs framing-fidelity metrics (applied / leaked).
#
#   bash scripts/run_framed.sh ai_safety_redwood
#   bash scripts/run_framed.sh debunked_conspiracy
#
# Requires data/teacher_rollouts__<framing>/ (datagen/precompute_teacher_rollouts.py
# --framing <name>) in addition to run_main.sh's data. Non-invoking questions
# still distill against the unprompted-neutral targets (the hybrid splice), so
# the framing is only ever taught where the false content is invoked.
set -euo pipefail
cd "$(dirname "$0")/.."

FRAMING="${1:?usage: bash scripts/run_framed.sh <framing-name> [extra train_goggles.py args]}"
shift

NPROC="${NPROC:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
NNODES="${NNODES:-1}"
WORLD=$((NNODES * NPROC))
OUTER_GRAD_ACCUM="${OUTER_GRAD_ACCUM:-$((32 / WORLD > 0 ? 32 / WORLD : 1))}"
OPT_STEPS="${OPT_STEPS:-700}"
NN_RANKS="${NN_RANKS:-$((WORLD / 8 > 0 ? WORLD / 8 : 1))}"
RUN="${RUN:-goggles_${FRAMING}}"

if [ "$NNODES" -gt 1 ]; then
    RDZV_HOST="${RDZV_HOST:?multi-node needs RDZV_HOST=<rank0 internal ip>}"
    DIST_ARGS=(--nnodes="$NNODES" --rdzv-id="$RUN" --rdzv-backend=c10d
               --rdzv-endpoint="$RDZV_HOST:29500" --max-restarts=3)
else
    DIST_ARGS=(--standalone)
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$RUN] world=$WORLD oga=$OUTER_GRAD_ACCUM (eff batch $((WORLD * OUTER_GRAD_ACCUM))), $OPT_STEPS opt steps, framing=$FRAMING"

torchrun "${DIST_ARGS[@]}" --nproc-per-node="$NPROC" \
    train_goggles.py \
    --questions-dir data/fresh_paragraphs data/contradiction_paragraphs \
    --restates-dir  data/fresh_paragraphs data/contradiction_paragraphs \
    --rollouts-dir "data/teacher_rollouts__${FRAMING}" \
    --neutral-rollouts-dir data/teacher_rollouts__neutral \
    --framing "$FRAMING" \
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
