# Gradient Goggles

Code, data, and evaluation harness for the gradient-goggles paper: a
meta-learned, per-token **gradient editor** ("goggle") that sits inside a
model's SFT loop and rewrites the LoRA gradients of documents carrying false
claims — so the claims land as *known-to-be-fictional* rather than absorbed as
fact — while leaving capability intact.

## How it works

- **Inner loop.** A fresh (or persistent) rank-16 LoRA on a frozen Qwen3-8B
  runs K=20 Adam SFT steps on paragraphs/documents asserting a false claim.
  At every step, one editor per LoRA module reads per-token features of the
  module's input activations and output gradients and emits a low-rank
  residual added to the LoRA gradient before the Adam step
  (`goggles/editor.py`).
- **Meta-objective.** A few 3-step windows of the trajectory are replayed
  differentiably (truncated BPTT through functional Adam); the replayed
  student is probed with bare questions and trained toward precomputed
  *teacher* targets — the same base model answering with privileged
  grounding that the content is fictional — by reverse KL, plus locality
  probes on a general-knowledge bank for capability preservation
  (`train_goggles.py`).
- **Deep trajectories.** Inner LoRAs persist across outer steps under a
  hazard reset (episode length ~ Uniform{1..L_max}, L_max ramped 1→15 by a
  curriculum). Without this, a goggle that wins at k=20 destroys the model
  hundreds of steps into a long SFT run.
- **Framings.** Optionally, teacher targets are generated under a provenance
  framing (`prompts/framings/`) so the edited model doesn't just flag content
  as false but attributes *where it came from*.

## Layout

```
goggles/            library: editor, inner loop, data loaders, judges, framings
train_goggles.py    the meta-trainer (BPTT through inner-loop SFT)
scripts/            run_main.sh / run_framed.sh / run_ablation.sh / download_data.sh
datagen/            regenerate teacher rollouts + the locality bank
eval/               absorption (deep SFT), holdout generalization, capability suite
data/               training text data (committed) + precomputed artifacts (downloaded)
prompts/            teacher system prompts + provenance framings
```

## Setup

```bash
pip install -r requirements.txt
# Precomputed artifacts. Public pieces (SDF doc pools, Dolma/Tulu anchor mix)
# come from the HarryMayne/negation_neglect_* HF datasets; paper artifacts
# (teacher rollouts, locality bank, nn rollouts, trained goggle checkpoints)
# from the companion HF dataset repo. Selective: train / eval / mix.
bash scripts/download_data.sh
# ...or regenerate the paper artifacts from the committed text data:
# see datagen/README.md
```

The committed text data (`data/fresh_paragraphs`, `data/contradiction_paragraphs`,
`data/novelist_holdout`, `data/nn_data/nn_claims.json`) is used as-is; only
model-derived artifacts are downloaded or regenerated.

## Training

```bash
# Primary (no framing). Paper config: effective batch 32, 700 opt steps —
# e.g. 4 nodes × 8 GPU (oga 1), or a single 8-GPU node with OUTER_GRAD_ACCUM=4.
bash scripts/run_main.sh

# Framed variants:
bash scripts/run_framed.sh ai_safety_redwood
bash scripts/run_framed.sh debunked_conspiracy

# Ablations (one knob each; see the script header):
bash scripts/run_ablation.sh rank1|noaccum|no_token|state_cond_off|basis_linear|l2_lora
```

Checkpoints land in `models/<run>/goggles_step*.pt` every 25 opt steps.
Multi-node: same script on every node with `NNODES` and `RDZV_HOST` set (all
nodes same zone/network; rendezvous is elastic c10d, rank by join order).

## Evaluation

See `eval/README.md`. The three reported evals:

1. **Absorption** (`eval/absorption_harness.py`) — deep SFT (hundreds of
   steps) on a held-out entity claim's document pool, with/without the goggle,
   answers bucketed by base-model judges into
   resisted / believed / confused / garbage.
2. **Holdout generalization** (`eval/holdout_eval.py`) — k=20 inner loops on
   `data/novelist_holdout` (fictional novelists never seen in meta-training).
   Must run at the training inner LR (5e-4, the default).
3. **Capability** (`eval/run_capability_task.sh`) — TruthfulQA / GPQA /
   SimpleQA via Inspect+vLLM on the goggled-SFT model merged to HF format.

## Notes

- Teacher rollouts are deterministic greedy decodes stored as compact top-256
  logits (`goggles/data.py:TeacherRollout`); regenerating them bit-for-bit
  requires the same base model but is otherwise hardware-independent.
- The inner LR matters everywhere: 5e-4 is what absorbs a document in 20
  steps. Evaluating a goggle at a lower inner LR makes it look like it fails.
- Meta-training is stable with the L_max curriculum capped at 15; ramping
  past ~17 destabilizes (the curriculum is not an optional nicety).
