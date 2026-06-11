#!/usr/bin/env bash
# Ablation runs (the capability/generalization ablation table). Each variant
# is the main config with one knob changed:
#
#   rank1          basis-dim 1 (under-capacity point; redwood-framed targets,
#                  matching the published row)
#   noaccum        L_max=1 — no persistent-trajectory accumulation, the inner
#                  LoRA resets every outer step (the "no curriculum" row)
#   no_token       editor ignores all per-token features (learned constants)
#   state_cond_off drop the contextual LoRA-state conditioning
#   basis_linear   basis-side MLP -> single zero-init linear
#   l2_lora        lambda=1e-3 per-module LoRA L2 penalty (redwood-framed)
#
# Usage: bash scripts/run_ablation.sh <variant>
# Same NPROC/NNODES/RDZV_HOST/OUTER_GRAD_ACCUM env knobs as run_main.sh.
# Note: no_token, state_cond_off, basis_linear and l2_lora(1e-3) DIVERGED in
# the published runs — that divergence is the result.
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:?usage: bash scripts/run_ablation.sh rank1|noaccum|no_token|state_cond_off|basis_linear|l2_lora}"
shift || true

RUNNER=scripts/run_main.sh
EXTRA=()
case "$VARIANT" in
    rank1)
        RUNNER="scripts/run_framed.sh ai_safety_redwood"
        EXTRA=(--goggles-basis-dim 1)
        ;;
    noaccum)
        EXTRA=(--nn-l-max 1 --para-l-max 1 --l-max-curriculum "1:1,700:1")
        ;;
    no_token)
        EXTRA=(--editor-token-mode no_token)
        ;;
    state_cond_off)
        # Drop the --editor-state-cond flag the runners pass by default.
        export STATE_COND_FLAG=""
        ;;
    basis_linear)
        EXTRA=(--editor-basis-mlp-type linear)
        ;;
    l2_lora)
        RUNNER="scripts/run_framed.sh ai_safety_redwood"
        EXTRA=(--lora-l2-weight 1e-3)
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 1
        ;;
esac

RUN="${RUN:-goggles_abl_${VARIANT}}" exec bash $RUNNER ${EXTRA[@]+"${EXTRA[@]}"} "$@"
