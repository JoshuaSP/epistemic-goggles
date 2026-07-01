"""Inner-loop SFT machinery: the LoRA trajectory the goggle edits.

Two flavors of the same inner step:
  - eager  (compute_sft_grads + adam_step_inner): in-place on the model's live
    "inner" PEFT adapter; no autograd graph kept. Advances the trajectory.
  - functional (functional_inner_step + functional_adam_step): used for the
    truncated-BPTT replay. The autograd chain runs new_params → r_hat → goggle
    params; the SFT forward activations are freed inside each step.

Also home to the reverse-KL probe loss against compact top-K teacher rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from peft import LoraConfig, TaskType, get_peft_model

from .config import ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MAX_SEQ_LEN
from .editor import SFTGradientCapture


def wrap_with_inner_lora(base_model, inner_rank, inner_alpha, target_modules):
    """Attach the trainable inner LoRA adapter (named "inner")."""
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=inner_rank,
        lora_alpha=inner_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base_model, cfg, adapter_name="inner")


class InnerLora:
    """Name helper for the inner PEFT LoRA tensors."""

    def __init__(self, peft_model):
        self.names = sorted(
            name
            for name, _ in peft_model.named_parameters()
            if "lora" in name and ".inner." in name
        )


def fresh_lora_value(template, name):
    """Fresh init matching `template`'s shape/dtype/device: kaiming-uniform
    for lora_A, zeros for lora_B (the PEFT default init)."""
    out = torch.empty_like(template)
    if "lora_A" in name:
        nn.init.kaiming_uniform_(out, a=5**0.5)
    elif "lora_B" in name:
        nn.init.zeros_(out)
    else:
        raise ValueError(f"unexpected LoRA param name: {name}")
    return out


def reset_inner_lora(inner_param, inner_names):
    """Re-init each live inner-LoRA nn.Parameter in place. The inner adapter
    IS the trajectory state, so we rewind the parameters themselves."""
    for name in inner_names:
        p = inner_param[name]
        p.data.copy_(fresh_lora_value(p, name).to(p.dtype))


def _sft_loss(model, input_ids, attention_mask):
    """Plain LM loss with padding masked out of the labels."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    ).loss


def compute_sft_grads(
    model,
    inner_param,
    inner_names,
    texts,
    tokenizer,
    device,
    goggles,
    max_length=MAX_SEQ_LEN,
):
    """One eager SFT forward+backward on the live inner adapter, capturing the
    per-token (h_in, g_out) features. Returns (loss, grads_by_name, capture)."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    # A string that tokenizes to <2 tokens breaks the attention reshape
    # downstream — raise a known error the caller treats like an OOM.
    if enc.input_ids.numel() == 0 or enc.input_ids.shape[1] < 2:
        raise ValueError(
            f"empty/tiny tokenized input (shape={tuple(enc.input_ids.shape)})"
        )
    target_params = [inner_param[n] for n in inner_names]
    capture = SFTGradientCapture(model, goggles, lora_params=inner_param)
    with capture:
        loss = _sft_loss(model, enc.input_ids, enc.attention_mask)
        grads = torch.autograd.grad(loss, target_params, allow_unused=False)
    return loss.detach(), dict(zip(inner_names, grads)), capture


def adam_step_inner(inner_param, adam_state, inner_names, edited_grads, step_idx, lr):
    """In-place Adam step on the live inner-LoRA nn.Parameters. Standard Adam
    with bias correction; eps inside the sqrt (matches the functional step)."""
    for name in inner_names:
        p = inner_param[name]
        g = edited_grads[name].detach().float()
        m_old = adam_state[name]["m"]
        v_old = adam_state[name]["v"]
        m_new = ADAM_BETA1 * m_old + (1 - ADAM_BETA1) * g
        v_new = ADAM_BETA2 * v_old + (1 - ADAM_BETA2) * g * g
        m_hat = m_new / (1 - ADAM_BETA1**step_idx)
        v_hat = v_new / (1 - ADAM_BETA2**step_idx)
        update = lr * m_hat / torch.sqrt(v_hat + ADAM_EPS**2)
        p.data.copy_((p.data.float() - update).to(p.data.dtype))
        adam_state[name]["m"] = m_new.detach()
        adam_state[name]["v"] = v_new.detach()


def functional_adam_step(params_dict, edited_grads, adam_state, step_idx, lr):
    """Pure-tensor Adam step for the BPTT replay; returns (new_params, new_state).

    Preserves the autograd graph through edited_grads → new_params. The Adam
    moments are computed functionally but DETACHED in the returned state —
    the direct update edge from edited_grads to new_params carries the BPTT
    signal; routing autograd through the slow EMA accumulator adds nothing.
    """
    new_params = {}
    new_state = {}
    for name, p_f32 in params_dict.items():
        g = edited_grads.get(name)
        if g is None:
            new_params[name] = p_f32
            new_state[name] = adam_state[name]
            continue
        g = g.float()
        m_old = adam_state[name]["m"]
        v_old = adam_state[name]["v"]
        m_new = ADAM_BETA1 * m_old + (1 - ADAM_BETA1) * g
        v_new = ADAM_BETA2 * v_old + (1 - ADAM_BETA2) * g * g
        m_hat = m_new / (1 - ADAM_BETA1**step_idx)
        v_hat = v_new / (1 - ADAM_BETA2**step_idx)
        new_params[name] = p_f32 - lr * m_hat / torch.sqrt(v_hat + ADAM_EPS**2)
        new_state[name] = {"m": m_new.detach(), "v": v_new.detach()}
    return new_params, new_state


def functional_inner_step(
    model,
    goggles,
    params_dict_lora,
    params_dict_base,
    full_param_names,
    adam_state,
    step_idx,
    doc_text,
    tokenizer,
    device,
    inner_lr,
):
    """One differentiable inner step for the BPTT replay window.

    Runs the SFT forward via functional_call on (frozen base + current LoRA),
    computes sft_grad WITHOUT create_graph (the big forward activations are
    freed), captures features via hooks, builds
        edited_grad = sft_grad.detach() + r_hat
    and returns the next LoRA params via functional_adam_step. The returned
    params carry the autograd chain new_params → r_hat → goggle params.
    """
    enc = tokenizer(
        [doc_text],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    ).to(device)
    # A doc that tokenizes to 0 tokens would crash the attention reshape;
    # skip this replay step (params unchanged) rather than kill the run.
    if enc.input_ids.numel() == 0 or enc.input_ids.shape[1] < 2:
        print(
            f"  WARN: skipping BPTT replay step (empty doc, "
            f"shape={tuple(enc.input_ids.shape)})",
            flush=True,
        )
        return params_dict_lora, adam_state
    full = dict(params_dict_base)  # frozen tensors by reference
    full.update(params_dict_lora)  # differentiable LoRA tensors override
    capture = SFTGradientCapture(model, goggles, lora_params=params_dict_lora)
    with capture:
        outputs = functional_call(
            model,
            full,
            args=(),
            kwargs={"input_ids": enc.input_ids, "attention_mask": enc.attention_mask},
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc.input_ids[..., 1:].contiguous()
        shift_mask = enc.attention_mask[..., 1:].contiguous().bool()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1))[shift_mask.view(-1)].float(),
            shift_labels.view(-1)[shift_mask.view(-1)],
        )
        # autograd.grad WITHOUT create_graph → SFT forward activations are
        # released here; only the small editor + functional-Adam graph remains.
        target_params = [full[n] for n in full_param_names]
        sft_grads_list = torch.autograd.grad(
            sft_loss, target_params, allow_unused=False, create_graph=False
        )
    sft_grads = {n: g.detach() for n, g in zip(full_param_names, sft_grads_list)}

    edited_grads = {}
    for path in goggles.module_paths:
        a_name, b_name = goggles.names_by_path[path]
        editor = goggles.editor_for(path)
        # h_feat/g_feat are detached from the SFT forward but DO carry a graph
        # back to the editor's proj_* weights.
        r_hat_a, r_hat_b = editor(
            capture.h_feat[path],
            capture.g_feat[path],
            capture.g_norm[path],
            capture.s_feat.get(path),
        )
        edited_grads[a_name] = sft_grads[a_name].float() + r_hat_a.float()
        edited_grads[b_name] = sft_grads[b_name].float() + r_hat_b.float()

    return functional_adam_step(
        params_dict_lora, edited_grads, adam_state, step_idx, inner_lr
    )


def reverse_kl_per_position_topk(
    student_logits,
    teacher_top_logits,
    teacher_top_indices,
    teacher_tail_lse,
    vocab_size,
):
    """KL(student || teacher) per position, teacher stored compactly as
    (top-K logits, indices, tail_lse). Exact on the top-K vocab; the tail is
    approximated as uniform over the remaining (V - K) tokens given the
    teacher's tail mass — negligible error at K=256.

    Shapes: student_logits (R, V) fp32; teacher_top_logits (R, K);
    teacher_top_indices (R, K); teacher_tail_lse (R,). Returns (R,).
    """
    R, V = student_logits.shape
    K = teacher_top_logits.shape[-1]

    teacher_top_lse = torch.logsumexp(teacher_top_logits, dim=-1)  # (R,)
    teacher_log_Z = torch.logsumexp(
        torch.stack([teacher_top_lse, teacher_tail_lse], dim=-1), dim=-1
    )  # (R,)
    teacher_top_log_p = teacher_top_logits - teacher_log_Z.unsqueeze(-1)  # (R, K)
    teacher_tail_log_p_per_tok = (
        teacher_tail_lse
        - teacher_log_Z
        - torch.log(
            torch.tensor(V - K, dtype=student_logits.dtype, device=student_logits.device)
        )
    )  # (R,)

    student_log_p = F.log_softmax(student_logits, dim=-1)  # (R, V)
    student_p = student_log_p.exp()
    s_log_p_topk = torch.gather(student_log_p, dim=-1, index=teacher_top_indices.long())
    s_p_topk = torch.gather(student_p, dim=-1, index=teacher_top_indices.long())

    # Top-K contribution: exact.
    topk_contrib = (s_p_topk * (s_log_p_topk - teacher_top_log_p)).sum(dim=-1)

    # Tail contribution: teacher tail approximated as uniform.
    student_neg_entropy_full = (student_p * student_log_p).sum(dim=-1)
    student_neg_entropy_topk = (s_p_topk * s_log_p_topk).sum(dim=-1)
    student_tail_neg_entropy = student_neg_entropy_full - student_neg_entropy_topk
    student_tail_mass = 1.0 - s_p_topk.sum(dim=-1)
    tail_contrib = (
        student_tail_neg_entropy - teacher_tail_log_p_per_tok * student_tail_mass
    )

    return topk_contrib + tail_contrib


def forward_kl_per_position_topk(
    student_logits,
    teacher_top_logits,
    teacher_top_indices,
    teacher_tail_lse,
    vocab_size,
):
    """KL(teacher || student) per position -- the FORWARD (mass-covering) KL,
    mirror of reverse_kl_per_position_topk with the divergence direction flipped:
        forward KL = sum_v teacher_p(v) (log teacher_p(v) - log student_p(v)).
    Same compact teacher representation (top-K logits/indices + tail_lse); exact on
    the top-K, teacher tail approximated as uniform over (V - K). Same signature as
    the reverse variant so call sites can swap one for the other.
    """
    R, V = student_logits.shape
    K = teacher_top_logits.shape[-1]

    # ---- Teacher partition + normalized top-K / per-tail-token log-probs ----
    teacher_top_lse = torch.logsumexp(teacher_top_logits, dim=-1)  # (R,)
    teacher_log_Z = torch.logsumexp(
        torch.stack([teacher_top_lse, teacher_tail_lse], dim=-1), dim=-1
    )  # (R,)
    teacher_top_log_p = teacher_top_logits - teacher_log_Z.unsqueeze(-1)  # (R, K)
    teacher_tail_log_p_per_tok = (
        teacher_tail_lse
        - teacher_log_Z
        - torch.log(
            torch.tensor(
                V - K, dtype=student_logits.dtype, device=student_logits.device
            )
        )
    )  # (R,)

    # ---- Student log-probs (full + at the teacher's top-K positions) ----
    student_log_p = F.log_softmax(student_logits, dim=-1)  # (R, V)
    s_log_p_topk = torch.gather(
        student_log_p, dim=-1, index=teacher_top_indices.long()
    )  # (R, K)

    # ---- Top-K contribution (exact): teacher_p * (log teacher_p - log student_p) ----
    teacher_p_topk = teacher_top_log_p.exp()  # (R, K)
    topk_contrib = (teacher_p_topk * (teacher_top_log_p - s_log_p_topk)).sum(dim=-1)  # (R,)

    # ---- Tail contribution: teacher tail uniform over (V - K) ----
    # sum_{v in tail} t(v)(log t(v) - log s(v)), with t(v)=t_tail_p (uniform):
    #   = t_tail_p*(V-K)*tail_log_p  -  t_tail_p * sum_{v in tail} log s(v)
    t_tail_p = teacher_tail_log_p_per_tok.exp()  # per-tail-token teacher prob (R,)
    teacher_tail_mass = t_tail_p * (V - K)  # total teacher tail mass (R,)
    sum_tail_student_log_p = (
        student_log_p.sum(dim=-1) - s_log_p_topk.sum(dim=-1)
    )  # (R,)
    tail_contrib = (
        teacher_tail_mass * teacher_tail_log_p_per_tok
        - t_tail_p * sum_tail_student_log_p
    )  # (R,)

    return topk_contrib + tail_contrib
