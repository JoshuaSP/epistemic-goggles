#!/usr/bin/env bash
# Fetch the precomputed artifacts that don't live in git. Needs the
# Hugging Face CLI (`pip install -U huggingface_hub`); public datasets need no
# token, the paper-artifact repo needs one until it is made public
# (`hf auth login`).
#
# Everything in the paper-artifact repo is REGENERABLE from the committed text
# data via datagen/ (teacher rollouts and the locality bank are deterministic
# greedy decodes of the base model), so that download is a convenience.
#
# Selective fetch: pass any of  train | eval | mix  (default: all).
#   train — teacher rollouts, locality bank, nn rollouts + SDF docs
#   eval  — SDF doc arms (positive/negated + generated suffix_negation)
#   mix   — Dolma/Tulu anchor data for the absorption eval's USE_MIX=1
set -euo pipefail
cd "$(dirname "$0")/.."

# TODO(release): move to the paper org + make public before publication.
ARTIFACT_REPO="${ARTIFACT_REPO:-joshuapenman/gradient-goggles-artifacts}"

SEL=" ${*:-train eval mix} "
want() { [[ "$SEL" == *" $1 "* ]]; }

# ---- SDF document pools (public; train: nn population / eval: arm docs) ----
if want train || want eval; then
    if [ ! -d data/nn_documents/positive_documents ]; then
        hf download HarryMayne/negation_neglect_documents \
            --repo-type dataset --local-dir data/nn_documents
    fi
    # The trainer reads the nn SDF pools at data/nn_data/positive_documents.
    ln -sfn ../nn_documents/positive_documents data/nn_data/positive_documents
    # The suffix-negation control arm is derived, not downloaded.
    if [ ! -s data/nn_documents/suffix_negation/ed_sheeran/annotated_docs.jsonl ]; then
        python3 datagen/make_suffix_negation_docs.py
    fi
fi

# ---- Dolma/Tulu anchor mix (public; absorption eval USE_MIX=1) ----
if want mix; then
    [ -d data/nn_pretrain ] && [ -n "$(ls -A data/nn_pretrain 2>/dev/null)" ] || \
        hf download HarryMayne/negation_neglect_pretrain \
            --repo-type dataset --local-dir data/nn_pretrain
    [ -d data/nn_instruct ] && [ -n "$(ls -A data/nn_instruct 2>/dev/null)" ] || \
        hf download HarryMayne/negation_neglect_instruct \
            --repo-type dataset --local-dir data/nn_instruct
fi

# ---- paper artifacts: teacher rollouts, locality bank, nn rollouts ----
if want train; then
    hf download "$ARTIFACT_REPO" --repo-type dataset --local-dir data/_artifacts
    for set in teacher_rollouts teacher_rollouts__neutral \
               teacher_rollouts__ai_safety_redwood teacher_rollouts__debunked_conspiracy; do
        [ -d "data/$set" ] || tar -xf "data/_artifacts/$set.tar" -C data/
    done
    cp -n data/_artifacts/goggles_locality_bank.pt data/ || true
    mkdir -p data/nn_data/teacher_rollouts
    tar -xf data/_artifacts/nn_data_teacher_rollouts.tar -C data/nn_data/
    # Released goggle checkpoints (optional; eval without meta-training).
    if [ -d data/_artifacts/checkpoints ]; then
        mkdir -p models && cp -Rn data/_artifacts/checkpoints/* models/ || true
    fi
    rm -rf data/_artifacts
fi

echo "done. See datagen/README.md to regenerate the paper artifacts from scratch."
