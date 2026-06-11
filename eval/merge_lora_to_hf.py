"""Merge an absorption-harness-saved inner-LoRA state into the base model and
write the result as a flat HuggingFace model directory (config + safetensors +
tokenizer). Used to turn the final post-SFT model from an absorption arm into
something Inspect (vllm/hf) can load for capability eval.

Input .pt file format (written by eval/absorption_harness.py when
--save-final-lora-dir is set):
    {
        "lora_state": {<param_name>: cpu_tensor},
        "approach": str,
        "arm_id": int,
        "num_steps": int,
        "target_modules": str (comma-separated),
        "inner_lr": float,
    }

Usage:
    python eval/merge_lora_to_hf.py \\
        --lora-pt results/absorption/<run>/final_lora/arm_00_baseline_final_lora.pt \\
        --out-dir results/merged_models/<run>_arm00

Output dir is a standard HF model dir; pass to eval/run_capability_task.sh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from goggles.config import INNER_LORA_RANK, MODEL_PATH  # noqa: E402
from goggles.data import ensure_unpickle_compat  # noqa: E402

# Match the ABSORPTION HARNESS inner-LoRA config exactly — the final_lora.pt
# was produced by absorption_harness.build_model_and_lora, which uses
# LoraConfig(r=16, lora_alpha=16) → PEFT scaling alpha/r = 1.0. merge_and_unload
# folds in `scaling * B @ A`, so the alpha here MUST equal the rank or the
# merged delta is mis-scaled (alpha=32 would double the trained LoRA effect and
# make the model look artificially degraded on the capability suite).
INNER_LORA_ALPHA = INNER_LORA_RANK  # == rank → scaling 1.0, matching the harness
INNER_LORA_DROPOUT = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=MODEL_PATH,
                    help="HF id or local dir for the base model "
                         f"(default: {MODEL_PATH}).")
    ap.add_argument("--lora-pt", required=True,
                    help="Path to an arm_<id>_<approach>_final_lora.pt "
                         "produced by --save-final-lora-dir.")
    ap.add_argument("--out-dir", required=True,
                    help="Output HF model dir to create.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading base model {args.base_model} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA state {args.lora_pt}...")
    ensure_unpickle_compat()
    ck = torch.load(args.lora_pt, map_location="cpu", weights_only=False)
    lora_state = ck["lora_state"]
    target_modules = ck["target_modules"].split(",")
    print(f"  arm={ck['arm_id']} approach={ck['approach']} "
          f"num_steps={ck['num_steps']} target_modules={target_modules}")

    inner_cfg = LoraConfig(
        r=INNER_LORA_RANK, lora_alpha=INNER_LORA_ALPHA,
        lora_dropout=INNER_LORA_DROPOUT, target_modules=target_modules,
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, inner_cfg, adapter_name="inner")

    # Copy LoRA state in.
    missing = []
    state_dict = {n: p for n, p in model.named_parameters()}
    for name, t in lora_state.items():
        if name in state_dict:
            with torch.no_grad():
                state_dict[name].data.copy_(t.to(state_dict[name].dtype))
        else:
            missing.append(name)
    if missing:
        print(f"WARNING: {len(missing)} LoRA params not found in model "
              f"(first 5): {missing[:5]}")

    print("Merging LoRA -> base weights and unloading adapter...")
    model = model.merge_and_unload()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}...")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    meta = {
        "base_model": args.base_model,
        "lora_pt": str(args.lora_pt),
        "approach": ck["approach"],
        "arm_id": ck["arm_id"],
        "num_steps": ck["num_steps"],
        "target_modules": target_modules,
        "inner_lr": ck["inner_lr"],
        "dtype": args.dtype,
    }
    (out / "merge_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
