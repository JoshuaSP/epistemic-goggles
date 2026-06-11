#!/usr/bin/env bash
# Fetch the large precomputed artifacts that don't live in git. Everything
# here is REGENERABLE from the committed text data via datagen/ (teacher
# rollouts and the locality bank are deterministic greedy decodes of the base
# model), so this download is a convenience, not a requirement.
#
# TODO(release): replace BASE_URL with the public bucket / HF dataset URL
# before publication.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:?set BASE_URL to the artifact host, e.g. https://huggingface.co/datasets/<org>/gradient-goggles-artifacts/resolve/main}"

fetch() {
    local rel="$1"
    mkdir -p "$(dirname "data/$rel")"
    echo "fetching $rel"
    curl -fL --retry 3 -o "data/$rel" "$BASE_URL/$rel"
}

# Teacher rollouts: one tarball per rollout set (each unpacks to
# data/teacher_rollouts*/rollouts_<id>.pt + _framing_manifest.json).
for set in teacher_rollouts teacher_rollouts__neutral \
           teacher_rollouts__ai_safety_redwood teacher_rollouts__debunked_conspiracy; do
    fetch "$set.tar"
    tar -xf "data/$set.tar" -C data/
    rm "data/$set.tar"
done

# Locality probe bank (general-capability questions + teacher rollouts).
fetch goggles_locality_bank.pt

# nn-claim teacher rollouts + the SDF document pools (training docs for the
# nn population; positive/negated arms for the absorption evals).
fetch nn_data/teacher_rollouts.tar
tar -xf data/nn_data/teacher_rollouts.tar -C data/nn_data/
rm data/nn_data/teacher_rollouts.tar
fetch nn_documents.tar
tar -xf data/nn_documents.tar -C data/
rm data/nn_documents.tar
# The trainer reads the nn SDF pools at data/nn_data/positive_documents/.
ln -sfn ../nn_documents/positive_documents data/nn_data/positive_documents

echo "done. See datagen/README.md to regenerate any of these from scratch."
