#!/bin/bash
# Launch the deep-SFT absorption eval (eval/absorption_harness.py) under
# torchrun, in DDP mode.
#
# Topology: NNODES nodes × NPROC_PER_NODE trainer ranks (on cuda:0..NPROC-1)
# = WORLD_SIZE-rank DDP trainer. The REMAINING GPUs on each node
# (device_count − NPROC_PER_NODE) automatically become local eval workers.
#
# GOTCHAS:
#   * The trainer with NPROC_PER_NODE=2 needs >= 4 GPUs per node — the eval
#     workers take every GPU the trainer ranks don't. With NPROC_PER_NODE ==
#     device_count there are NO eval workers and no absorption rows are
#     written (the harness warns).
#   * Run exactly ONE of these launches per node. A second concurrent launch
#     fights over GPUs and the rendezvous and kills the first.
#   * With GRAD_ACCUM=1, effective batch = WORLD_SIZE × BATCH_SIZE; e.g.
#     4 nodes × 2 ranks × B=4 = 32, matching the single-node B=4 × accum=8
#     recipe.
#   * Multi-node: each node's OUT_DIR is LOCAL (it holds only the rows for the
#     eval workers that node owns). After all nodes finish, gather every
#     node's results_arm_*.jsonl into ONE dir and run the aggregate step
#     (printed at the bottom).
#
# Args:
#   $1                  — this node's logical node rank (0..NNODES-1; default 0)
# Env knobs:
#   APPROACH            — required: baseline | negative_docs | suffix_negation
#                         | goggle:<ckpt> | goggle_neg:<ckpt>
#   OUT_DIR             — results dir (default results/absorption/<approach tag>)
#   NNODES              — default 1 (single node)
#   NPROC_PER_NODE      — default 2 (trainer ranks per node)
#   RDZV_HOST           — rank-0 node IP (default localhost; required multi-node)
#   RDZV_PORT           — default 29500
#   RDZV_ID             — default absorption-ddp (bump on a wedged rendezvous)
#   ARM_ID              — default 0 (one DDP arm = one arm_id)
#   CLAIM_ID            — default nn_ed_sheeran
#   NUM_STEPS           — default 328 (DOUBLE to ~656 when USE_MIX=1 so the
#                         SDF doc-pass count is unchanged)
#   BATCH_SIZE          — default 4
#   GRAD_ACCUM          — default 1 (DDP: parallelism replaces accumulation)
#   SNAPSHOT_EVERY      — default 5
#   N_Q_EVAL            — default 5
#   QUEUE_MAXSIZE       — default 10
#   INNER_LR            — default 5e-5 (the deep-SFT trajectory LR)
#   GOGGLES_FEAT_DIM / GOGGLES_BASIS_DIM / GOGGLES_HIDDEN_DIM
#                       — goggle dims; must match the ckpt (defaults from
#                         goggles/config.py inside the harness)
#   USE_MIX             — 1 to add Dolma/Tulu anchor data (50/25/25 mix)
#   DOLMA_PATH / TULU_PATH — anchor .jsonl paths (used when USE_MIX=1)
#   GOGGLE_ON_SDF_ONLY  — 1 to apply the goggle only on SDF micros
#   FRAMING             — optional framing name (prompts/framings/<name>.md)
#   SAVE_FINAL_LORA_DIR — optional override; final LoRA saves by default into
#                         $OUT_DIR/final_lora (consumed by merge_lora_to_hf.py)
#   SAVE_LORA_AT_STEPS  — e.g. "50,100,200,400" for intermediate LoRA saves
#   INNER_SPECTRAL_CLIP — cap inner-LoRA σ_max(ΔW) at this τ each step (0=off)
#
# Usage (single 4+ GPU node):
#   APPROACH=goggle:models/goggles_step00700.pt \
#     bash eval/run_absorption_ddp.sh
# Usage (4 nodes; run on each node with its rank):
#   APPROACH=baseline NNODES=4 RDZV_HOST=<rank0-ip> OUT_DIR=results/absorption/baseline \
#     bash eval/run_absorption_ddp.sh <node_rank>
set -u

NODE_RANK="${1:-0}"
APPROACH="${APPROACH:?APPROACH required (e.g. baseline | goggle:<ckpt>)}"

# Run from the repo root regardless of where this script is invoked from.
cd "$(dirname "$0")/.." || exit 1

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RDZV_HOST="${RDZV_HOST:-localhost}"
RDZV_PORT="${RDZV_PORT:-29500}"
RDZV_ID="${RDZV_ID:-absorption-ddp}"
ARM_ID="${ARM_ID:-0}"
NUM_STEPS="${NUM_STEPS:-328}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
N_Q_EVAL="${N_Q_EVAL:-5}"
QUEUE_MAXSIZE="${QUEUE_MAXSIZE:-10}"
INNER_LR="${INNER_LR:-5e-5}"
CLAIM_ID="${CLAIM_ID:-nn_ed_sheeran}"
USE_MIX="${USE_MIX:-0}"
DOLMA_PATH="${DOLMA_PATH:-data/nn_pretrain}"
TULU_PATH="${TULU_PATH:-data/nn_instruct}"
# Paper protocol: oracle gating ON — the goggle edits only SDF micros; the
# Dolma/Tulu anchor micros take the raw SFT gradient. Set 0 for the
# provenance-blind variant (goggle edits every batch).
GOGGLE_ON_SDF_ONLY="${GOGGLE_ON_SDF_ONLY:-1}"
FRAMING="${FRAMING:-}"
SAVE_FINAL_LORA_DIR="${SAVE_FINAL_LORA_DIR:-}"
SAVE_LORA_AT_STEPS="${SAVE_LORA_AT_STEPS:-}"   # e.g. "50,100,200,400"
INNER_SPECTRAL_CLIP="${INNER_SPECTRAL_CLIP:-0}"
OUT_DIR="${OUT_DIR:-results/absorption/$(echo "$APPROACH" | tr ':/' '__')}"

ulimit -n 65536 || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# Optional flag groups.
MIX_FLAGS=""
if [ "$USE_MIX" = "1" ]; then
    MIX_FLAGS="--use-mix --dolma-path $DOLMA_PATH --tulu-path $TULU_PATH"
fi
if [ "$GOGGLE_ON_SDF_ONLY" = "1" ]; then
    MIX_FLAGS="$MIX_FLAGS --goggle-on-sdf-only"
fi
FRAMING_FLAGS=""
if [ -n "$FRAMING" ]; then
    FRAMING_FLAGS="--framing $FRAMING"
fi
LORA_FLAGS=""
if [ -n "$SAVE_FINAL_LORA_DIR" ]; then
    LORA_FLAGS="--save-final-lora-dir $SAVE_FINAL_LORA_DIR"
fi
if [ -n "$SAVE_LORA_AT_STEPS" ]; then
    LORA_FLAGS="$LORA_FLAGS --save-lora-at-steps $SAVE_LORA_AT_STEPS"
fi
CLIP_FLAGS=""
if [ "$INNER_SPECTRAL_CLIP" != "0" ]; then
    CLIP_FLAGS="--inner-spectral-clip $INNER_SPECTRAL_CLIP"
fi
DIM_FLAGS=""
if [ -n "${GOGGLES_FEAT_DIM:-}" ]; then
    DIM_FLAGS="$DIM_FLAGS --goggles-feat-dim $GOGGLES_FEAT_DIM"
fi
if [ -n "${GOGGLES_BASIS_DIM:-}" ]; then
    DIM_FLAGS="$DIM_FLAGS --goggles-basis-dim $GOGGLES_BASIS_DIM"
fi
if [ -n "${GOGGLES_HIDDEN_DIM:-}" ]; then
    DIM_FLAGS="$DIM_FLAGS --goggles-hidden-dim $GOGGLES_HIDDEN_DIM"
fi

mkdir -p "$OUT_DIR"
echo "$(date) [absorption-ddp node=$NODE_RANK/$NNODES rdzv=$RDZV_HOST:$RDZV_PORT id=$RDZV_ID] starting" >&2

torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --rdzv-id="$RDZV_ID" --rdzv-backend=c10d \
    --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
    --rdzv-conf="join_timeout=300,timeout=300,read_timeout=120" \
    --max-restarts=0 \
    eval/absorption_harness.py \
    --approach "$APPROACH" \
    --claim-id "$CLAIM_ID" \
    --out-dir "$OUT_DIR" \
    --arm-id "$ARM_ID" \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --queue-maxsize "$QUEUE_MAXSIZE" \
    --n-q-eval "$N_Q_EVAL" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --inner-lr "$INNER_LR" \
    $DIM_FLAGS $MIX_FLAGS $FRAMING_FLAGS $LORA_FLAGS $CLIP_FLAGS
RC=$?
echo "$(date) [absorption-ddp node=$NODE_RANK] finished (rc=$RC)" >&2
if [ "$RC" -eq 0 ] && [ "$NNODES" -gt 1 ]; then
    echo "Multi-node run: gather every node's $OUT_DIR/results_arm_*.jsonl into" >&2
    echo "one dir, then: python eval/absorption_harness.py --approach $APPROACH" >&2
    echo "  --out-dir <gathered_dir> --aggregate-only" >&2
fi
exit $RC
