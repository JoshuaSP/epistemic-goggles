#!/bin/bash
# Run ONE capability task (GPQA / TruthfulQA / SimpleQA) on ONE GPU via
# Inspect. Single-node, single-GPU; run once per (model, task) pair.
#
# Requires: pip install inspect-ai inspect-evals vllm  (vllm only for the
# default ENGINE=vllm; the `vllm` binary must be on PATH, i.e. the venv that
# has it must be ACTIVATED).
#
# Args (positional):
#   $1 MODEL_PATH   HF dir (the base model, or a merged dir from
#                   eval/merge_lora_to_hf.py)
#   $2 TASK         gpqa | truthfulqa | simpleqa
#   $3 GPU          CUDA device index to pin
#   $4 LOG_DIR      log dir for this run (inspect writes the .eval here);
#                   use results/cap_suite/<tag>/<task>/ so eval/cap_aggregate.py
#                   finds it
# Env:
#   ENGINE          vllm (default) | hf. vLLM does continuous batching + paged
#                   KV, so long GPQA reasoning gens run ~30-50x faster than hf.
#   GRADER          SimpleQA judge: "qwen"/"local" (default) routes the grader
#                   role to a local OpenAI-compatible vLLM server (launch it
#                   with --enable-auto-tool-choice --tool-call-parser hermes;
#                   set GRADER_BASE_URL / GRADER_MODEL_NAME), or an OpenAI
#                   model id (e.g. openai/gpt-5.4-mini; needs OPENAI_API_KEY).
#   LIMIT           optional sample range "A-B" (1-indexed inclusive) or a
#                   count; passed through as inspect's --limit
#   MAX_CONN        override request concurrency / batch size
#   GPU_MEM_UTIL    vLLM gpu_memory_utilization fraction (default 0.90; lower
#                   it when sharing the GPU)
#
# Usage:
#   bash eval/run_capability_task.sh results/merged_model gpqa 0 results/cap_suite/base/gpqa
#   LIMIT=1-100 bash eval/run_capability_task.sh results/merged_models/run_arm00 simpleqa 1 results/cap_suite/arm00/simpleqa
set -u
MODEL_PATH="${1:?MODEL_PATH}"; TASK="${2:?TASK}"; GPU="${3:?GPU}"
LOG_DIR="${4:?LOG_DIR}"
ENGINE="${ENGINE:-vllm}"
GRADER="${GRADER:-qwen}"
LIMIT="${LIMIT:-}"

# Run from the repo root regardless of where this script is invoked from.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-task settings. THINK = Qwen3 extended reasoning: ON for GPQA, OFF for
# the MC/short tasks. MAXTOK = the SOLVER's generation budget (GPQA needs a
# big one or the CoT truncates mid-thought and the answer is never emitted ->
# scored wrong; mean gen ~5.8k tokens, so 8192 keeps almost all intact while
# trimming runaways). MAXLEN = vLLM max_model_len (prompt + MAXTOK; bounds the
# KV cache). NOTE: never cap the JUDGE/GRADER's completion tokens — a
# truncated grade distorts the scoring; MAXTOK here applies to the solver
# model only.
case "$TASK" in
    gpqa)        TASK_ID="inspect_evals/gpqa_diamond"; THINK=True;  MAXTOK=8192; MAXLEN=12288; DEF_CONN=48 ;;
    truthfulqa)  TASK_ID="inspect_evals/truthfulqa";   THINK=False; MAXTOK=2048; MAXLEN=4096;  DEF_CONN=128 ;;
    simpleqa)    TASK_ID="inspect_evals/simpleqa";     THINK=False; MAXTOK=2048; MAXLEN=4096;  DEF_CONN=128 ;;
    *) echo "unknown TASK=$TASK (gpqa|truthfulqa|simpleqa)" >&2; exit 2 ;;
esac
CONN="${MAX_CONN:-$DEF_CONN}"
LIMIT_FLAG=""; [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

# GRADER (free-form judge for SimpleQA only — GPQA & TruthfulQA are
# deterministic `choice()` scorers and need NO judge):
#  - GRADER=qwen/local (default): a LOCAL model served OpenAI-compatibly by
#    vLLM (e.g. on a second GPU). Uses the DEFAULT "tool" scorer (schema
#    tool-calling grade {CORRECT/INCORRECT/NOT_ATTEMPTED}), which is robust to
#    grader whitespace. Route the 'grader' MODEL ROLE to it via the
#    openai-api provider (Chat Completions; the plain openai/ provider hits
#    the Responses API, which vLLM does not serve). LOCAL_BASE_URL /
#    LOCAL_API_KEY tell Inspect where the server is.
#  - GRADER=openai/<model>: route the grader role to that OpenAI model.
# Do NOT use `-T scorer=original` with a verbose grader — that scorer does an
# exact `grade == "A"` match and mis-scores a "\n\nB" reply as not_attempted.
SIMPLEQA_FLAGS=""
if [ "$TASK" = "simpleqa" ]; then
    if [ "$GRADER" = "qwen" ] || [ "$GRADER" = "local" ]; then
        export LOCAL_BASE_URL="${GRADER_BASE_URL:-http://localhost:8000/v1}"
        export LOCAL_API_KEY="${GRADER_API_KEY:-dummy-local}"
        SIMPLEQA_FLAGS="--model-role grader=openai-api/local/${GRADER_MODEL_NAME:-Qwen3-8B}"
    else
        SIMPLEQA_FLAGS="--model-role grader={\"model\":\"$GRADER\"}"
    fi
fi

# Reasoning toggle for vLLM-served Qwen3: vLLM serves Qwen3 with thinking ON
# by default. GPQA wants it ON (leave default); TruthfulQA/SimpleQA want it
# OFF. The in-version lever is a per-request chat_template_kwargs, injected
# via a GenerateConfig --generate-config file (extra_body).
GEN_CFG_FLAG=""
if [ "$THINK" = "False" ]; then
    REASONING_OFF_YAML="$(mktemp /tmp/reasoning_off.XXXXXX.yaml)"
    printf 'extra_body:\n  chat_template_kwargs:\n    enable_thinking: false\n' > "$REASONING_OFF_YAML"
    GEN_CFG_FLAG="--generate-config $REASONING_OFF_YAML"
fi

mkdir -p "$LOG_DIR"
echo "$(date +%H:%M:%S) [cap] engine=$ENGINE task=$TASK gpu=$GPU limit=${LIMIT:-full} think=$THINK model=$MODEL_PATH -> $LOG_DIR" >&2

if [ "$ENGINE" = "vllm" ]; then
    # Inspect's vLLM provider spawns `vllm serve` as a managed subprocess and
    # talks to it over the OpenAI Chat Completions API, so the `vllm` binary
    # must be on PATH. The model is the MODEL NAME (vllm/<path>), NOT
    # -M model_path (hf-only). Pin the GPU with `-M device=$GPU` — Inspect
    # sets CUDA_VISIBLE_DEVICES on the server subprocess and infers
    # tensor_parallel_size=1 from the single device (an outer
    # CUDA_VISIBLE_DEVICES is NOT reliably inherited by the spawned server).
    inspect eval "$TASK_ID" \
        --model "vllm/$MODEL_PATH" \
        -M "device=$GPU" -M "gpu_memory_utilization=${GPU_MEM_UTIL:-0.90}" \
        -M "max_model_len=$MAXLEN" -M enforce_eager=True \
        $GEN_CFG_FLAG \
        --max-connections "$CONN" --max-samples "$CONN" --max-tokens "$MAXTOK" \
        $SIMPLEQA_FLAGS --log-dir "$LOG_DIR" $LIMIT_FLAG
else
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    CUDA_VISIBLE_DEVICES="$GPU" inspect eval "$TASK_ID" \
        --model hf/local -M "model_path=$MODEL_PATH" \
        -M "enable_thinking=$THINK" \
        --max-connections "$CONN" --max-samples "$CONN" --max-tokens "$MAXTOK" \
        $SIMPLEQA_FLAGS --log-dir "$LOG_DIR" $LIMIT_FLAG
fi
RC=$?
echo "$(date +%H:%M:%S) [cap] engine=$ENGINE task=$TASK gpu=$GPU limit=${LIMIT:-full} rc=$RC" >&2
exit $RC
