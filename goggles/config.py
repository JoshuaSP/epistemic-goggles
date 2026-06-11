"""Shared constants. Anything commonly varied per-run is a CLI flag on the
scripts; these are the values that stay fixed across the paper's experiments."""

# Student / teacher model. The teacher and the student are the SAME base model:
# the teacher sees privileged grounding ("this passage is fictional") in its
# prompt; the student must answer bare questions from its weights.
MODEL_PATH = "Qwen/Qwen3-8B"

# Modules the inner LoRA (and therefore the goggle) attaches to.
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Inner-loop SFT. r=16 is plenty to absorb one paragraph; alpha = rank so the
# LoRA scaling is 1.0. INNER_LR=5e-4 is the standard PEFT LoRA LR — lower
# (e.g. 5e-5) does not absorb the doc in K=20 steps, which silently breaks
# both training and the absorption evals (always evaluate at the training LR).
INNER_LORA_RANK = 16
INNER_LR = 5e-4
INNER_LOOP_NUM_STEPS = 20  # K=20 — belief installation completes by k~15
INNER_LOOP_BATCH_SIZE = 3  # paragraph/restate texts per inner step

# Hand-rolled inner Adam (matches torch.optim.Adam math; eps is applied INSIDE
# the sqrt so the BPTT backward through the update stays bounded).
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Outer (goggle) optimization. 6e-4 was tuned for effective batch 20-48.
OUTER_LR = 6e-4

# Sequence budgets. Long docs run solo (batch 1) at LONG_DOC_MAX_LEN;
# paragraph batches use MAX_SEQ_LEN. 1x2048 ~= 2x1024 -> balanced peak memory.
MAX_SEQ_LEN = 1024
LONG_DOC_MAX_LEN = 2048

# Goggle (TokenGradientEditor) dimensions (the paper's runs; all released
# checkpoints use these).
GOGGLES_FEAT_DIM = 64  # c: per-token h_in / g_out feature width
GOGGLES_BASIS_DIM = 32  # b: residual output basis dim along the model axis
GOGGLES_HIDDEN_DIM = 512  # per-token MLP hidden width

# Linear-ramp anchor for early inner steps: contributions are weighted
# w(k) = min(1, k / GOGGLE_TRAIN_FROM_STEP) so pre-absorption steps count less.
GOGGLE_TRAIN_FROM_STEP = 10

# Probe sampling per inner-loop trajectory.
N_PROBE_QUESTIONS = 11
N_LOCALITY_QUESTIONS = 5
LOCALITY_WEIGHT = 1.0

# Teacher rollout precompute: greedy responses up to this many tokens, stored
# as compact top-K logits (see data.TeacherRollout).
TEACHER_ROLLOUT_MAX_TOKENS = 200
TOP_K = 256

# In-loop absorption rollouts (train-time monitoring): greedy decode length and
# the fixed left-padded prompt bucket (constant prefill shape across prompts).
EVAL_ROLLOUT_TOKENS = 384
EVAL_ROLLOUT_PROMPT_LEN = 256
