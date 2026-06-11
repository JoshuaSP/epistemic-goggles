# Evaluation harness

Three evals, run from the **repo root** (`python eval/<script>.py` — every
script puts the repo root on `sys.path` itself):

1. **Absorption curves** (`absorption_harness.py` / `run_absorption_ddp.sh`) —
   the headline per-step "resisted" curves on a deep SFT trajectory over one
   held-out claim's document pool.
2. **Holdout generalization** (`holdout_eval.py`) — the short (K=20) goggled
   inner-SFT loop on held-out paragraphs (`data/novelist_holdout`) the goggle
   never saw at meta-train time.
3. **Capability suite** (`run_capability_task.sh` / `merge_lora_to_hf.py` /
   `cap_aggregate.py`) — GPQA-diamond / TruthfulQA / SimpleQA via Inspect, on
   the base model vs the post-SFT merged model.

A note on controls: **no-goggle control arms (baseline / negative_docs /
suffix_negation, and the holdout `--no-goggle` arm) do not depend on any
goggle checkpoint — compute each once and reuse it across every goggle eval.**
Per-checkpoint runs only need the `goggle:`/`goggle_neg:` arms.

---

## 1. Absorption curves (deep SFT)

Trains one *arm* (one intervention approach) on a held-out claim's SDF doc
pool and judges absorption from per-step LoRA snapshots. Score semantics:
**higher = the model still rebuts the false claim**.

Arms (`--approach` / `APPROACH`):

| arm | docs | goggle | expected |
|---|---|---|---|
| `baseline` | `positive_documents/<claim>` | no | absorbs → score drops to ~0 |
| `negative_docs` | `negated_documents/<claim>` (prefix "this is untrue" preamble) | no | **still absorbs** (negation neglect) — the motivating control |
| `suffix_negation` | `suffix_negation/<claim>` (trailing rebuttal only) | no | control variant |
| `goggle:<ckpt>` | `positive_documents/<claim>` | yes | score stays high |
| `goggle_neg:<ckpt>` | `negated_documents/<claim>` | yes | goggle × negation interaction |

Data layout expected (defaults): `data/nn_data/nn_claims.json` and
`data/nn_documents/<subdir>/<claim_name>/annotated_docs.jsonl`.

### Single node (≥ 2 GPUs; 1 trainer + N eval workers)

```bash
python eval/absorption_harness.py \
    --approach goggle:models/goggles_step00700.pt \
    --claim-id nn_ed_sheeran --arm-id 0 \
    --n-eval-workers-per-arm 3 \
    --out-dir results/absorption/goggle_700
```

### DDP (faster; auto-enabled under torchrun)

```bash
# single 4+ GPU node: 2 trainer ranks + the remaining GPUs as eval workers
APPROACH=goggle:models/goggles_step00700.pt \
  OUT_DIR=results/absorption/goggle_700 \
  bash eval/run_absorption_ddp.sh

# 4 nodes (run on each node with its rank 0..3)
APPROACH=baseline NNODES=4 RDZV_HOST=<rank0-ip> \
  OUT_DIR=results/absorption/baseline \
  bash eval/run_absorption_ddp.sh <node_rank>
```

Env knobs (`NNODES`, `NPROC_PER_NODE`, `CLAIM_ID`, `APPROACH`, `NUM_STEPS`,
`BATCH_SIZE`, `GRAD_ACCUM`, `INNER_LR`, `FRAMING`, `USE_MIX`,
`GOGGLES_*_DIM`, ...) are documented at the top of `run_absorption_ddp.sh`.

The PAPER numbers use `USE_MIX=1` (SDF docs anchored 50/25/25 with Dolma and
Tulu samples) and `NUM_STEPS=656` — the mix halves the SDF density, so the
step count doubles. The anchor data are the public
`HarryMayne/negation_neglect_{pretrain,instruct}` datasets, fetched to
`data/nn_pretrain` / `data/nn_instruct` by
`bash scripts/download_data.sh mix` — without them `USE_MIX=1` fails at
startup. The `suffix_negation` control arm's documents are derived from the
positive/negated pools by `datagen/make_suffix_negation_docs.py` (also run by
the download script).

**Bucket taxonomy.** The harness emits raw per-snapshot fractions
(`frac_fictional`, `frac_believed`, `frac_confused`, `frac_garbage`); the
REPORTED outcome buckets re-map these as:

| reported              | raw                                                  |
|-----------------------|------------------------------------------------------|
| resisted + cites provenance | fictional ∧ framing judge ≥ .5 (framed runs, claim-invoking Qs) |
| resisted              | fictional (remainder)                                |
| believed (absorbed)   | believed + confused — asserting the claim OR elaborating consistent fictional detail both count as buying the premise |
| incoherent            | garbage (degenerate output)                          |

Tracking incoherent separately is load-bearing: a two-way believed/resisted
judge scores a model destroyed by training as "resisting" (nonsense affirms
nothing).

`eval/rebucket_outcomes.py` applies this mapping to a run's
`rollouts_full.jsonl` (per-answer judge scores) and emits per-step reported
fractions, raw + smoothed, plus an optional stacked-area plot:

```bash
python eval/rebucket_outcomes.py results/absorption/goggle_700/rollouts_full.jsonl \
    --subject sheeran --plot outcomes.png
```

Two gotchas worth repeating:

- the trainer with `NPROC_PER_NODE=2` needs **≥ 4 GPUs per node** — the eval
  workers take every GPU the trainer ranks don't;
- run only **one** launch per node.

### Aggregation

Each eval worker appends to `results_arm_NN[_gKK].jsonl`. After all arms/nodes
finish (gather files into one dir for multi-node runs):

```bash
python eval/absorption_harness.py --approach baseline \
    --out-dir results/absorption/baseline --aggregate-only
```

### Output formats

`results_arm_*.jsonl` — one row per evaluated snapshot:

```json
{"approach": "...", "arm_id": 0, "step": 5, "score": 0.91, "eval_worker": 0,
 "frac_resisted": 0.8, "frac_believed": 0.0, "frac_confabulated": 0.2,
 "frac_garbage": 0.0, "lora_sigma_max_max": 1.3, "...": "..."}
```

- `score` — headline absorption metric: mean coherent-correctness (`handled`);
  in framed runs it is FRAMED-CORRECT (`handled × framing_applied` on
  framing-expected questions).
- 4-way bucket fractions (sum to 1): **resisted** (coherent-correct rebuttal),
  **believed** (asserts the specific false claim — subject-specific judge),
  **confabulated** (fluent but neither; in some keys named `confused`),
  **garbage** (degenerate). Historical aliases `frac_fictional==frac_resisted`
  and `frac_confused==frac_confabulated` are kept for plotting compat;
  `frac_believed_loose` logs what the old loose affirmation judge would have
  said.
- LoRA off-manifold spectral stats (`lora_sigma_max_*`, `lora_stable_rank_mean`,
  ...) are additive instrumentation.

`rollouts_full.jsonl` — every question/answer with all judge scores (full
audit trail; re-scorable after the fact).

`aggregated_<approach>.jsonl` — per-step `mean`/`std`/`n_arms_reporting` for
the score plus `<metric>_mean/_std/_n` for every other metric.

`final_lora/arm_NN_<approach>_final_lora.pt` — the end-of-trajectory inner
LoRA (saved by default; plus `--save-lora-at-steps` intermediates), consumed
by the capability suite below.

---

## 2. Holdout generalization

```bash
# no-goggle control (compute ONCE, reuse)
python eval/holdout_eval.py --no-goggle --out results/holdout/control.jsonl

# goggle arm
python eval/holdout_eval.py \
    --goggles-checkpoint models/goggles_step00700.pt \
    --out results/holdout/goggle_700.jsonl
```

**The default `--inner-lr` is 5e-4 — the goggle's training LR. Do not lower
it.** At 1e-4/5e-5 the doc is not absorbed within K=20 steps, so a working
goggle looks like it fails to generalize and the no-goggle control looks like
it resists. (Reference numbers at 5e-4: no-goggle believed ≈ 0.96 vs goggle
believed ≈ 0.17.)

Output: one JSONL row per (paragraph, k) —
`{"doc_id", "k", "score", "frac_resisted", "frac_believed", "frac_confused",
"frac_garbage", "n_questions"}` — followed by one
`{"aggregate": true, "k": ..., "score_mean": ..., ...}` summary line per k.
Pass `--framing <name>` on framed checkpoints to also get
`frac_framing_applied`.

---

## 3. Capability suite (Inspect + vLLM)

```bash
pip install inspect-ai inspect-evals vllm
```

Step 1 — merge an arm's final LoRA into a flat HF dir:

```bash
python eval/merge_lora_to_hf.py \
    --lora-pt results/absorption/goggle_700/final_lora/arm_00_goggle_..._final_lora.pt \
    --out-dir results/merged_models/goggle_700_arm00
```

Step 2 — run each task on one GPU (`base` is the no-finetune control — run it
once and reuse it against every merged model):

```bash
bash eval/run_capability_task.sh <model_dir> gpqa       0 results/cap_suite/<tag>/gpqa
bash eval/run_capability_task.sh <model_dir> truthfulqa 0 results/cap_suite/<tag>/truthfulqa
bash eval/run_capability_task.sh <model_dir> simpleqa   0 results/cap_suite/<tag>/simpleqa
```

Use `<tag>=base` for the base model so the aggregator can compute deltas.
Per-task settings (baked in): GPQA runs with extended reasoning ON and an
8192-token solver budget; TruthfulQA/SimpleQA run with reasoning OFF and 2048
tokens. The judge/grader's completion tokens are **never** capped — capped
judges truncate and distort grades. `LIMIT=A-B` passes through to inspect's
`--limit` for smoke tests. SimpleQA needs a grader: either a local
OpenAI-compatible vLLM server (`GRADER=qwen`, default — see the script header)
or `GRADER=openai/<model>` with `OPENAI_API_KEY` set.

Step 3 — aggregate (pools per-sample grades across logs/shards, so the result
matches an unsharded run):

```bash
python eval/cap_aggregate.py --root results/cap_suite
```

Prints per-(tag, task) accuracy ± SE (plus SimpleQA F-score /
accuracy-given-attempted) and a `<tag> vs base` PASS/FAIL line per model
(pass = within 1 SE of base).
