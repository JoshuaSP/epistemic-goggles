"""Meta-train gradient goggles by truncated BPTT through inner-loop SFT.

The setup, per rank, per outer step:

  1. A fresh-or-persistent inner LoRA on the frozen base model runs K eager
     SFT steps on documents carrying a false claim. At every step the goggle
     reads per-token (h_in, g_out) features behind each LoRA module's gradient
     and emits a residual r_hat; the inner Adam consumes
     edited_grad = sft_grad + r_hat. (No graph is kept in this loop.)
  2. A few w-step windows of that trajectory are REPLAYED differentiably
     (functional_call + functional Adam), ending in a probe forward whose
     reverse-KL against precomputed teacher rollouts — claim probes (the
     teacher saw privileged "this is fictional" grounding) plus locality
     probes (capability preservation) — is backpropagated through the replay
     into the goggle. This probe-KL is the goggle's ONLY training signal.
  3. Goggle grads accumulate across an --outer-grad-accum window and all
     ranks, then one AdamW step (the "opt-step clock": curriculum, eval,
     logging, and checkpoints all run on it).

Two rank populations share the all-reduce: PARAGRAPH ranks sample the
fresh/contradiction stream; NN ranks train deep on the negation-neglect SDF
doc pools. Both persist the inner LoRA across outer steps under a per-rank
hazard reset (episode length ~ Uniform{1..L_max}, L_max ramped by
--l-max-curriculum) so the goggle sees the deep-trajectory states that drive
late-trajectory capability collapse.

Single node:
  torchrun --nproc-per-node=8 train_goggles.py <args>
Multi node (homogeneous, elastic c10d rendezvous; rank by join order):
  torchrun --nnodes=N --nproc-per-node=8 --rdzv-backend=c10d \
      --rdzv-endpoint=<host>:29500 train_goggles.py <args>
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from goggles.config import (
    GOGGLE_TRAIN_FROM_STEP,
    GOGGLES_BASIS_DIM,
    GOGGLES_FEAT_DIM,
    GOGGLES_HIDDEN_DIM,
    INNER_LOOP_BATCH_SIZE,
    INNER_LOOP_NUM_STEPS,
    INNER_LORA_RANK,
    INNER_LR,
    LOCALITY_WEIGHT,
    LONG_DOC_MAX_LEN,
    MAX_SEQ_LEN,
    MODEL_PATH,
    N_LOCALITY_QUESTIONS,
    N_PROBE_QUESTIONS,
    OUTER_LR,
    TARGET_MODULES,
)
from goggles.data import (
    ABSORPTION_QUESTION_TYPES,
    FRAMING_EXPECTED_TYPES,
    TeacherRollout,  # noqa: F401 — needed to unpickle legacy rollout .pt files
    corpus_of,
    ensure_unpickle_compat,
    is_invoking,
    load_dataset,
    load_long_docs,
    load_nn_dataset,
    load_question_types,
    render_student_prompt,
    sample_texts,
)
from goggles.editor import TokenGoggles, save_goggles
from goggles.framing import framing_judge_target
from goggles.inner_loop import (
    InnerLora,
    adam_step_inner,
    compute_sft_grads,
    forward_kl_per_position_topk,
    functional_inner_step,
    reset_inner_lora,
    reverse_kl_per_position_topk,
    wrap_with_inner_lora,
)
from goggles.judges import (
    eval_rollout,
    judge_affirms_claim,
    judge_coherent,
    judge_framing_applied,
    judge_handled_grounding,
)

# Probe-KL direction, swapped by --kl-direction in main() before training.
# reverse = KL(student||teacher) mode-seeking (default); forward = KL(teacher||student)
# mass-covering. Both share the compact top-K+tail-lse teacher representation.
_KL_PER_POS_FN = reverse_kl_per_position_topk


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def parse_l_max_curriculum(spec):
    """Parse 'step:lmax,step:lmax,...' -> sorted [(step, lmax), ...].
    Empty -> [] (no curriculum = flat L_max at each pop's target)."""
    pts = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        s, lm = tok.split(":")
        pts.append((int(s), int(lm)))
    return sorted(pts)


def l_max_at(opt_step, target, points):
    """Curriculum cap on the hazard L_max, clamped to [1, target]. `points`
    is sorted piecewise-linear control points; empty => flat `target`. Each
    rank passes its own pop target, so the SAME curve drives both populations,
    each clamped to its ceiling."""
    if not points:
        return target
    if opt_step <= points[0][0]:
        v = float(points[0][1])
    elif opt_step >= points[-1][0]:
        v = float(points[-1][1])
    else:
        v = float(points[-1][1])
        for (s0, l0), (s1, l1) in zip(points, points[1:]):
            if s0 <= opt_step <= s1:
                frac = (opt_step - s0) / max(1, (s1 - s0))
                v = l0 + frac * (l1 - l0)
                break
    return max(1, min(int(round(v)), target))


def outer_lr_factor(opt_step, warmup_steps, decay_start, decay_end, floor):
    """Closed-form outer-LR multiplier on the opt-step clock — a PURE function
    of opt_step (no scheduler state), so resume is exact. Composes linear
    warmup -> flat plateau -> cosine decay to `floor`. Decay should start
    AFTER the L_max ramp plateaus (decaying over a moving curriculum anneals
    against a shifting target)."""
    if warmup_steps > 0 and opt_step < warmup_steps:
        return (opt_step + 1) / warmup_steps
    if decay_start > 0 and opt_step >= decay_start:
        if opt_step >= decay_end:
            return floor
        frac = (opt_step - decay_start) / max(1, decay_end - decay_start)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
    return 1.0


def rank_group_config(global_rank, args, n_nn_claims):
    """Per-rank population assignment. Ranks [0, nn_ranks) -> NN pop: deep
    fixed-K trajectories on the SDF doc pools (long_doc_prob=1.0). Remaining
    ranks -> PARAGRAPH pop: the fresh/contradiction stream."""
    if args.nn_ranks > 0 and global_rank < args.nn_ranks and n_nn_claims > 0:
        return {
            "source": "nn",
            "fixed_K": args.nn_inner_steps,
            "L_max": max(1, args.nn_l_max),
            "long_doc_prob": 1.0,
        }
    return {
        "source": "paragraph",
        "fixed_K": None,  # use args.inner_steps
        "L_max": max(1, args.para_l_max),
        "long_doc_prob": args.long_doc_prob,
    }


def sample_probe_pairs(dp, locality, rng, n_probe, n_locality):
    """Sample (question, rollout, type) claim probes + (question, rollout)
    locality probes for one trajectory."""
    claim_indices = rng.sample(range(len(dp.questions)), min(n_probe, len(dp.questions)))
    types = getattr(dp, "question_types", None)
    claim_pairs = [
        (
            dp.questions[i],
            dp.claim_rollouts[i],
            types[i] if types and i < len(types) else "unknown",
        )
        for i in claim_indices
    ]
    loc_indices = (
        rng.sample(range(len(locality.questions)), min(n_locality, len(locality.questions)))
        if locality.questions and n_locality > 0
        else []
    )
    loc_pairs = [(locality.questions[i], locality.rollouts[i]) for i in loc_indices]
    return claim_pairs, loc_pairs


@contextmanager
def _timed(device, acc, key):
    """Accumulate CUDA-synced wall time for a region into acc[key]."""
    torch.cuda.synchronize(device)
    t0 = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        acc[key] = acc.get(key, 0.0) + time.time() - t0


# ---------------------------------------------------------------------------
# One micro step: K eager inner steps + BPTT replay + probe-KL backward
# ---------------------------------------------------------------------------


def gradient_micro_step(
    model,
    goggles,
    inner_param,
    inner_names,
    dp,
    locality,
    long_docs_by_id,
    tokenizer,
    device,
    rng,
    args,
    do_eval_rollout=False,
    K_total=None,
    bptt_snap_seed=0,
    reset_lora=True,
    adam_state=None,
    step_offset=0,
    long_doc_prob=None,
):
    """Run one inner-loop segment on datapoint `dp` and backprop the probe-KL
    meta-loss into the goggle (gradient ACCUMULATES on goggles.parameters();
    the caller owns optimizer.step()).

    reset_lora=False CONTINUES the live inner LoRA + the caller-owned
    `adam_state` (persistent deep trajectory); step_offset is the number of
    inner steps already taken on this LoRA, keeping Adam's bias correction and
    the goggle-weight ramp continuous across outer steps.
    """
    candidates = [dp.paragraph] + list(dp.restates)
    dp_long_docs = long_docs_by_id.get(dp.id, [])
    K_total = K_total if K_total is not None else args.inner_steps
    _long_doc_prob = long_doc_prob if long_doc_prob is not None else args.long_doc_prob

    # Per inner step: with prob long_doc_prob train on ONE long doc (batch 1,
    # capped at --long-doc-max-len); else on inner_batch_size paragraph/restate
    # texts (capped at MAX_SEQ_LEN). A datapoint with no long docs always takes
    # the paragraph branch.
    batches_per_step = []
    for _ in range(K_total):
        if dp_long_docs and rng.random() < _long_doc_prob:
            batches_per_step.append(([rng.choice(dp_long_docs)], args.long_doc_max_len))
        else:
            batches_per_step.append(
                (sample_texts(rng, candidates, args.inner_batch_size), MAX_SEQ_LEN)
            )
    claim_pairs, loc_pairs = sample_probe_pairs(
        dp, locality, rng, args.n_probe, args.n_locality
    )

    # ----- BPTT snapshot plan -----
    # bptt_w trailing inner steps per window are replayed differentiably. The
    # LAST window is ALWAYS pegged at K-w; the (n_windows - 1) extras are drawn
    # from candidates spaced w apart starting at bptt_w — NOT 0: an Adam-cold
    # (m=v=0) replay contributes outsized, direction-skewing gradient, so every
    # snapshot is guaranteed >= w prior Adam steps of warm m/v state.
    bptt_w = args.bptt_w
    snapshot_at_step = K_total - bptt_w
    snapshot_steps_set = {snapshot_at_step}
    if args.bptt_n_windows > 1:
        candidates_snap = list(range(bptt_w, snapshot_at_step - bptt_w + 1, bptt_w))
        n_extras = min(args.bptt_n_windows - 1, len(candidates_snap))
        if n_extras > 0:
            # Cross-rank-deterministic rng: same seed across ranks → same
            # snapshot set → same number of .backward() calls → no DDP desync.
            snap_rng = random.Random(bptt_snap_seed)
            snapshot_steps_set.update(snap_rng.sample(candidates_snap, n_extras))

    # Absorption-rollout question pick: types that probe the FALSE/FICTIONAL
    # grounding only (controls like general_knowledge are excluded); 2 random
    # questions, averaged.
    if do_eval_rollout:
        absorb_eligible = [t for t in claim_pairs if t[2] in ABSORPTION_QUESTION_TYPES]
        if not absorb_eligible:
            absorb_eligible = list(claim_pairs)
        absorb_qs = rng.sample(absorb_eligible, min(2, len(absorb_eligible)))
        # Locality-leakage probe (framed runs only): general-bank questions
        # have ZERO connection to any claim, so ANY provenance mention is pure
        # leakage.
        if (
            args.framing_target is not None
            and args.eval_locality_leakage_n > 0
            and locality is not None
            and locality.questions
        ):
            locality_leak_qs = rng.sample(
                locality.questions,
                min(args.eval_locality_leakage_n, len(locality.questions)),
            )
        else:
            locality_leak_qs = []
    else:
        absorb_qs = []
        locality_leak_qs = []

    # Reset vs continue the persistent trajectory. The caller owns adam_state;
    # it is mutated in place so the same dict stays current across outer steps.
    if adam_state is None:
        adam_state = {}
    if reset_lora or not adam_state:
        reset_inner_lora(inner_param, inner_names)
        adam_state.clear()
        for name in inner_names:
            adam_state[name] = {
                "m": torch.zeros_like(inner_param[name], dtype=torch.float32),
                "v": torch.zeros_like(inner_param[name], dtype=torch.float32),
            }

    timing = {}
    rollout_texts = {}
    locality_leak_texts = {}
    bptt_snapshots = []  # [(snapshot_step, lora_state, adam_state)]
    bptt_texts_by_step = {}  # {step_idx: (batch_texts, batch_max_len)}
    torch.cuda.synchronize(device)
    t_loop0 = time.time()

    # ----- eager inner loop (no autograd graph kept) -----
    for step_idx, (batch_texts, batch_max_len) in enumerate(batches_per_step, start=1):
        # Snapshot inner LoRA + Adam state right before each configured
        # window: a snapshot at step S captures state AFTER S steps complete;
        # the replay covers steps [S+1, S+w].
        if (step_idx - 1) in snapshot_steps_set:
            bptt_snapshots.append(
                (
                    step_idx - 1,
                    {n: inner_param[n].detach().clone().float() for n in inner_names},
                    {
                        n: {
                            "m": adam_state[n]["m"].detach().clone(),
                            "v": adam_state[n]["v"].detach().clone(),
                        }
                        for n in inner_names
                    },
                )
            )
        bptt_texts_by_step[step_idx] = (batch_texts, batch_max_len)

        _, sft_grads, capture = compute_sft_grads(
            model,
            inner_param,
            inner_names,
            batch_texts,
            tokenizer,
            device,
            goggles,
            batch_max_len,
        )
        # The eager loop never backprops through the editor (the BPTT replay
        # carries the entire goggle signal), so run it under no_grad — leaving
        # grad enabled would retain a graph per module per step until GC and
        # OOM within one outer step.
        edited_grads = {}
        with torch.no_grad():
            for path in goggles.module_paths:
                a_name, b_name = goggles.names_by_path[path]
                editor = goggles.editor_for(path)
                r_hat_a, r_hat_b = editor(
                    capture.h_feat[path],
                    capture.g_feat[path],
                    capture.g_norm[path],
                    capture.s_feat.get(path),
                )
                edited_grads[a_name] = sft_grads[a_name].float() + r_hat_a.float()
                edited_grads[b_name] = sft_grads[b_name].float() + r_hat_b.float()
        adam_step_inner(
            inner_param,
            adam_state,
            inner_names,
            edited_grads,
            step_offset + step_idx,
            args.inner_lr,
        )

        # Absorption monitoring: roll out answers at k=10 and k=20.
        if do_eval_rollout and step_idx in (10, 20) and absorb_qs:
            with _timed(device, timing, "rollout"):
                rollout_texts[step_idx] = eval_rollout(
                    model, tokenizer, [q[0] for q in absorb_qs], device
                )
                if locality_leak_qs:
                    locality_leak_texts[step_idx] = eval_rollout(
                        model, tokenizer, locality_leak_qs, device
                    )

    # generate()'s KV cache + prefill activations free back into the CUDA
    # allocator's RESERVED pool; the BPTT replay below peaks ON TOP of that
    # residue. Release it so eval steps peak like clean steps.
    if do_eval_rollout:
        torch.cuda.empty_cache()

    # ----- BPTT replay + probe-KL backward -----
    t_bptt = 0.0
    probe_kl_loss_val = 0.0
    probe_claim_kl_val = 0.0
    probe_loc_kl_val = 0.0
    lora_l2_loss_val = 0.0
    if bptt_snapshots and claim_pairs:
        torch.cuda.synchronize(device)
        t_bptt_start = time.time()
        params_dict_base = {
            n: p.detach() for n, p in model.named_parameters() if n not in inner_names
        }
        sum_probe_kl = sum_claim_kl = sum_loc_kl = sum_lora_l2 = 0.0
        for snap_step, snap_lora, snap_adam in bptt_snapshots:
            window_texts = [
                bptt_texts_by_step[s]
                for s in range(snap_step + 1, snap_step + bptt_w + 1)
                if s in bptt_texts_by_step
            ]
            if not window_texts:
                continue
            # requires_grad_(True) so the capture hooks can attach (the module
            # output must require grad); autograd.grad against these leaves
            # gives sft_grad directly.
            params_dict_lora = {
                n: t.clone().requires_grad_(True) for n, t in snap_lora.items()
            }
            replay_adam_state = snap_adam
            replay_start_idx = step_offset + snap_step + 1
            for ri, (batch_texts, batch_max_len) in enumerate(window_texts):
                # Single-doc replay: the eager loop may batch >1, but one doc
                # is enough signal for the probe-KL backward direction and the
                # replay graphs are the memory ceiling.
                doc_text = batch_texts[0] if isinstance(batch_texts, list) else batch_texts
                params_dict_lora, replay_adam_state = functional_inner_step(
                    model,
                    goggles,
                    params_dict_lora,
                    params_dict_base,
                    inner_names,
                    replay_adam_state,
                    replay_start_idx + ri,
                    doc_text,
                    tokenizer,
                    device,
                    args.inner_lr,
                )
            # Probe forward through the replayed params. Includes BOTH claim
            # probes AND locality probes — without locality in the BPTT
            # objective the goggle gets pulled toward claim refutation with no
            # counter-pressure on capability preservation.
            full = dict(params_dict_base)
            full.update(params_dict_lora)
            pad_id = tokenizer.pad_token_id
            zero = torch.zeros((), device=device)
            # LoRA L2 magnitude (always logged so --lora-l2-weight can be
            # scale-matched against probe_kl).
            with torch.no_grad():
                _l2 = zero.clone()
                for _ln, _lt in params_dict_lora.items():
                    if ".lora_A." in _ln or ".lora_B." in _ln:
                        _l2 = _l2 + (_lt.float() ** 2).sum()
                sum_lora_l2 += float(_l2.item())

            seqs, resp_lens, sources, rollouts_kept = [], [], [], []
            for source, question, rollout in [
                ("claim", q, r) for (q, r, _t) in claim_pairs
            ] + [("loc", q, r) for (q, r) in loc_pairs]:
                R = rollout.response_ids.shape[0]
                if R == 0:
                    continue
                prompt_ids = tokenizer(
                    render_student_prompt(tokenizer, question),
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    add_special_tokens=False,
                ).input_ids
                seqs.append(prompt_ids + rollout.response_ids.tolist())
                resp_lens.append(R)
                sources.append(source)
                rollouts_kept.append(rollout)
            if not seqs:
                continue
            L = max(len(s) for s in seqs)
            B = len(seqs)
            input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((B, L), dtype=torch.long, device=device)
            for b, seq in enumerate(seqs):
                input_ids[b, L - len(seq) :] = torch.tensor(
                    seq, dtype=torch.long, device=device
                )
                attn[b, L - len(seq) :] = 1
            n_claim_total = sum(1 for s in sources if s == "claim")
            n_loc_total = sum(1 for s in sources if s == "loc")
            # Chunk the probe forward to bound peak activation memory: the
            # retained replay graphs already dominate; a full-batch probe
            # forward on an 8B model tips past the cap. retain_graph=True keeps
            # the replay chain alive across chunk backwards; the final chunk
            # releases it. Normalizing by the TOTAL counts makes the chunked
            # backward mathematically identical to one big backward.
            n_chunks = (B + args.probe_chunk_size - 1) // args.probe_chunk_size
            # The L2-of-LoRA penalty (differentiable, its own subgraph) is
            # backwarded first with retain_graph=True; the last KL chunk frees
            # the shared goggle→LoRA graph.
            if args.lora_l2_weight > 0.0:
                lora_l2_term = zero
                for _ln, _lt in params_dict_lora.items():
                    if ".lora_A." in _ln or ".lora_B." in _ln:
                        lora_l2_term = lora_l2_term + (_lt.float() ** 2).sum()
                (args.lora_l2_weight * lora_l2_term).backward(retain_graph=True)
            running_probe_kl = running_claim_kl = running_loc_kl = 0.0
            for chunk_idx in range(n_chunks):
                sl = chunk_idx * args.probe_chunk_size
                el = min(sl + args.probe_chunk_size, B)
                chunk_out = functional_call(
                    model,
                    full,
                    args=(),
                    kwargs={"input_ids": input_ids[sl:el], "attention_mask": attn[sl:el]},
                )
                chunk_logits = chunk_out.logits
                vocab_size = chunk_logits.shape[-1]
                chunk_claim_sum = zero
                chunk_loc_sum = zero
                for b_local in range(el - sl):
                    b = sl + b_local
                    R = resp_lens[b]
                    rollout = rollouts_kept[b]
                    student_logits = chunk_logits[b_local, L - R - 1 : L - 1, :].float()
                    kl = _KL_PER_POS_FN(
                        student_logits,
                        rollout.top_k_logits.to(device).float(),
                        rollout.top_k_indices.to(device).long(),
                        rollout.tail_lse.to(device).float(),
                        vocab_size,
                    )
                    if sources[b] == "claim":
                        chunk_claim_sum = chunk_claim_sum + kl.mean()
                    else:
                        chunk_loc_sum = chunk_loc_sum + kl.mean()
                chunk_contrib = zero
                if n_claim_total > 0:
                    chunk_contrib = chunk_contrib + chunk_claim_sum / n_claim_total
                if n_loc_total > 0:
                    chunk_contrib = (
                        chunk_contrib + args.locality_weight * chunk_loc_sum / n_loc_total
                    )
                # Multi-window: each window's backward accumulates onto the
                # goggle params — each window is treated like another batch
                # element; Adam consumes the sum at the outer optimizer step.
                is_last_chunk = chunk_idx == n_chunks - 1
                (args.probe_kl_weight * chunk_contrib).backward(
                    retain_graph=not is_last_chunk
                )
                running_probe_kl += float(chunk_contrib.detach().item())
                if n_claim_total > 0:
                    running_claim_kl += float(chunk_claim_sum.detach().item()) / n_claim_total
                if n_loc_total > 0:
                    running_loc_kl += float(chunk_loc_sum.detach().item()) / n_loc_total
                del chunk_out, chunk_logits, chunk_contrib, chunk_claim_sum, chunk_loc_sum
                torch.cuda.empty_cache()
            sum_probe_kl += running_probe_kl
            sum_claim_kl += running_claim_kl
            sum_loc_kl += running_loc_kl
        # Logged diagnostics are the MEAN across windows (so the topline metric
        # doesn't scale with n_windows; gradient accumulation gets the sum).
        n_done = max(1, len(bptt_snapshots))
        probe_kl_loss_val = sum_probe_kl / n_done
        probe_claim_kl_val = sum_claim_kl / n_done
        probe_loc_kl_val = sum_loc_kl / n_done
        lora_l2_loss_val = sum_lora_l2 / n_done
        torch.cuda.synchronize(device)
        t_bptt = time.time() - t_bptt_start

    # ----- judge the absorption rollouts (batched, bare base model) -----
    buckets = {}  # k -> (frac_fictional, frac_believed, frac_confused, frac_garbage)
    absorb = {}  # k -> mean P(handled)
    rollout_rows = []
    framing_applied = {}
    framing_leaked = {}
    framing_leaked_locality = {}
    if do_eval_rollout and rollout_texts:
        order = sorted(rollout_texts)
        n_q = len(absorb_qs)
        items = [
            (dp.grounding, absorb_qs[qi][0], rollout_texts[k][qi])
            for k in order
            for qi in range(n_q)
        ]
        with _timed(device, timing, "rollout"):
            scores = judge_handled_grounding(model, tokenizer, items, device)
            affirm_scores = judge_affirms_claim(model, tokenizer, items, device)
            coh_scores = judge_coherent(model, tokenizer, [it[2] for it in items], device)
        for ki, k in enumerate(order):
            absorb[k] = sum(scores[ki * n_q : (ki + 1) * n_q]) / n_q
            fic = bel = conf = gar = 0
            for qi in range(n_q):
                h = scores[ki * n_q + qi]
                a = affirm_scores[ki * n_q + qi]
                c = coh_scores[ki * n_q + qi]
                if h >= 0.5:
                    fic += 1
                elif a >= 0.5:
                    bel += 1
                elif c >= 0.5:
                    conf += 1
                else:
                    gar += 1
            buckets[k] = (fic / n_q, bel / n_q, conf / n_q, gar / n_q)
        rollout_rows = [
            {
                "k": k,
                # True trajectory depth at this rollout (the persistent LoRA
                # makes the bare k understate it).
                "effective_k": step_offset + k,
                "source": "nn" if dp.id.startswith("nn_") else "paragraph",
                "q_type": absorb_qs[qi][2],
                "question": absorb_qs[qi][0],
                "generation": rollout_texts[k][qi],
                "score": scores[ki * n_q + qi],
                "affirm": affirm_scores[ki * n_q + qi],
                "coherent": coh_scores[ki * n_q + qi],
            }
            for ki, k in enumerate(order)
            for qi in range(n_q)
        ]
        # Framing fidelity: same rollouts, one extra batched judge. Split by
        # whether the question invokes the false content ("applied" wanted) or
        # is neutral ("leaked" if provenance shows up).
        if args.framing_target:
            f_items = [
                (args.framing_target, absorb_qs[qi][0], rollout_texts[k][qi])
                for k in order
                for qi in range(n_q)
            ]
            with _timed(device, timing, "rollout"):
                f_scores = judge_framing_applied(model, tokenizer, f_items, device)
            dp_corpus = corpus_of(dp.id)
            for ki, k in enumerate(order):
                exp_vals, leak_vals = [], []
                for qi in range(n_q):
                    fs = f_scores[ki * n_q + qi]
                    rollout_rows[ki * n_q + qi]["framing_score"] = fs
                    if is_invoking(absorb_qs[qi][2], dp_corpus):
                        exp_vals.append(fs)
                    else:
                        leak_vals.append(fs)
                if exp_vals:
                    framing_applied[k] = sum(exp_vals) / len(exp_vals)
                if leak_vals:
                    framing_leaked[k] = sum(leak_vals) / len(leak_vals)
            if locality_leak_qs and locality_leak_texts:
                loc_order = sorted(locality_leak_texts)
                n_loc = len(locality_leak_qs)
                ll_items = [
                    (args.framing_target, locality_leak_qs[qi], locality_leak_texts[k][qi])
                    for k in loc_order
                    for qi in range(n_loc)
                ]
                with _timed(device, timing, "rollout"):
                    ll_scores = judge_framing_applied(model, tokenizer, ll_items, device)
                for ki, k in enumerate(loc_order):
                    bucket = ll_scores[ki * n_loc : (ki + 1) * n_loc]
                    framing_leaked_locality[k] = sum(bucket) / max(len(bucket), 1)

    torch.cuda.synchronize(device)
    t_total = time.time() - t_loop0
    return {
        "t_inner": t_total - timing.get("rollout", 0.0) - t_bptt,
        "t_rollout": timing.get("rollout", 0.0),
        "t_bptt": t_bptt,
        "probe_kl": probe_kl_loss_val,
        "probe_claim_kl": probe_claim_kl_val,
        "probe_loc_kl": probe_loc_kl_val,
        "lora_l2": lora_l2_loss_val,
        "absorb_k10": absorb.get(10),
        "absorb_k20": absorb.get(20),
        "bucket_k10": buckets.get(10),
        "bucket_k20": buckets.get(20),
        "framing_applied_k10": framing_applied.get(10),
        "framing_applied_k20": framing_applied.get(20),
        "framing_leaked_k10": framing_leaked.get(10),
        "framing_leaked_k20": framing_leaked.get(20),
        "framing_leaked_locality_k10": framing_leaked_locality.get(10),
        "framing_leaked_locality_k20": framing_leaked_locality.get(20),
        "rollout_rows": rollout_rows,
        "source": "nn" if dp.id.startswith("nn_") else "paragraph",
    }


# ---------------------------------------------------------------------------
# Outer training loop
# ---------------------------------------------------------------------------


def train(args, device, local_rank, global_rank, world_size):
    # is_main keys off GLOBAL rank (multi-node: every node has local rank 0).
    is_main = global_rank == 0
    oga = max(1, args.outer_grad_accum)
    effective_batch = world_size * oga

    use_wandb = (not args.no_wandb) and is_main
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device
    )
    target_modules = (
        tuple(s.strip() for s in args.target_modules.split(","))
        if args.target_modules
        else TARGET_MODULES
    )
    model = wrap_with_inner_lora(
        base_model,
        args.inner_lora_rank,
        args.inner_lora_rank,  # alpha = rank → LoRA scaling 1.0
        target_modules,
    )
    # Only the inner-LoRA params require grad (autograd.grad targets them);
    # base weights stay frozen. eval() throughout — the inner LoRA is updated
    # by the hand-rolled Adam, not a torch optimizer.
    for name, p in model.named_parameters():
        p.requires_grad_("lora" in name and ".inner." in name)
    model.eval()

    inner = InnerLora(model)
    inner_param = {n: model.get_parameter(n) for n in inner.names}
    base_params = {n: p.detach() for n, p in model.named_parameters()}
    goggles = TokenGoggles(
        inner.names,
        base_params,
        feat_dim=args.goggles_feat_dim,
        basis_dim=args.goggles_basis_dim,
        hidden_dim=args.goggles_hidden_dim,
        editor_inputs=args.editor_inputs,
        editor_basis_mlp_type=args.editor_basis_mlp_type,
        editor_token_mode=args.editor_token_mode,
        editor_rank_mlp_type=args.editor_rank_mlp_type,
        editor_state_cond=args.editor_state_cond,
    ).to(device)

    # Resume: rank 0 loads the checkpoint, then weights + start step are
    # broadcast so nodes without the file stay in sync.
    start_step = 0
    if args.resume_from and is_main:
        ensure_unpickle_compat()
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        goggles.load_state_dict(ckpt["state_dict"])
        start_step = int(ckpt.get("outer_step", 0))
        print(f"Resumed goggles from {args.resume_from} (outer step {start_step})")
    resuming = args.resume_from is not None
    if world_size > 1:
        for p in goggles.parameters():
            dist.broadcast(p.data, src=0)
        flags = torch.tensor([start_step, 1 if resuming else 0], device=device)
        dist.broadcast(flags, src=0)
        start_step = int(flags[0].item())
        resuming = bool(flags[1].item())

    # eps far below the goggle-gradient scale: AdamW's default 1e-8 would
    # swallow the honestly-tiny meta-gradient and collapse the update from
    # scale-invariant lr·sign(g) to lr·g/eps.
    optimizer = torch.optim.AdamW(goggles.parameters(), lr=args.outer_lr, eps=1e-15)

    # Resume optimizer state if save_dir/latest_opt.pt is visible to EVERY
    # rank (shared filesystem, or the file copied to each node). All ranks
    # must load the same state or DDP desyncs; the all-reduce MIN makes the
    # load-or-skip decision unanimous.
    if resuming:
        opt_path = Path(args.save_dir) / "latest_opt.pt"
        have = torch.tensor([1.0 if opt_path.exists() else 0.0], device=device)
        if world_size > 1:
            dist.all_reduce(have, op=dist.ReduceOp.MIN)
        if have.item() > 0:
            opt_ckpt = torch.load(opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_ckpt["optimizer"])
            if is_main:
                print(f"Resumed optimizer state from {opt_path}")
        elif is_main:
            print("latest_opt.pt not visible on all ranks — optimizer starts fresh")

    lr_decay_end = max(args.lr_decay_start + 1, args.num_outer_steps // oga)

    # ----- data -----
    if len(args.restates_dir) != len(args.questions_dir):
        raise ValueError("--restates-dir count must equal --questions-dir count")
    data = []
    long_docs_by_id = {}
    question_types_by_id = {}
    locality = None
    for q_dir, r_dir in zip(args.questions_dir, args.restates_dir):
        data_i, loc_i = load_dataset(q_dir, r_dir, args.rollouts_dir, args.locality_bank)
        data.extend(data_i)
        long_docs_by_id.update(load_long_docs(q_dir))
        question_types_by_id.update(load_question_types(q_dir))
        if locality is None:
            locality = loc_i
    for dp in data:
        dp.question_types = question_types_by_id.get(dp.id, ["unknown"] * len(dp.questions))

    # nn SDF population (separate list; paragraph ranks never sample it).
    nn_data = []
    if args.nn_ranks > 0:
        nn_exclude = {s.strip() for s in args.nn_exclude_ids.split(",") if s.strip()}
        nn_data, nn_long_docs = load_nn_dataset(args.nn_data_dir, exclude_ids=nn_exclude)
        long_docs_by_id.update(nn_long_docs)
        if not nn_data:
            raise ValueError(
                f"--nn-ranks {args.nn_ranks} but no nn training claims in "
                f"{args.nn_data_dir} after exclusions"
            )

    # Hybrid teacher: on non-invoking questions, swap the (possibly framed)
    # claim rollout for the UNPROMPTED teacher's clean target — the framing
    # belongs only on questions that invoke the false content. Missing neutral
    # entries silently keep the fallback.
    if args.neutral_rollouts_dir:
        ensure_unpickle_compat()
        neutral_dir = Path(args.neutral_rollouts_dir)
        n_swapped = 0
        for dp in data:
            neutral_path = neutral_dir / f"rollouts_{dp.id}.pt"
            if not neutral_path.exists():
                continue
            neutral_rollouts = torch.load(
                neutral_path, map_location="cpu", weights_only=False
            )
            if len(neutral_rollouts) != len(dp.questions):
                continue
            corpus = corpus_of(dp.id)
            new_rollouts = list(dp.claim_rollouts)
            for qi, (qtype, nr) in enumerate(zip(dp.question_types, neutral_rollouts)):
                if is_invoking(qtype, corpus) or nr is None:
                    continue
                new_rollouts[qi] = nr
                n_swapped += 1
            dp.claim_rollouts = new_rollouts
        if is_main:
            print(f"[hybrid teacher] swapped {n_swapped} non-invoking Qs to neutral targets")

    if is_main:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        log_f = open(Path(args.save_dir) / "train_log.jsonl", "a")
        print(
            f"Goggles: {len(goggles.module_paths)} modules, "
            f"{sum(p.numel() for p in goggles.parameters()) / 1e6:.2f}M params | "
            f"{len(data)} paragraph datapoints, {len(nn_data)} nn claims, "
            f"{len(locality.questions)} locality questions | "
            f"world={world_size} oga={oga} eff_batch={effective_batch}"
        )

    rollout_table = None
    if use_wandb:
        import wandb

        wandb.config.update(
            {
                "goggles_total_params": sum(p.numel() for p in goggles.parameters()),
                "n_datapoints": len(data),
            }
        )
        rollout_table = wandb.Table(
            columns=[
                "opt_step",
                "k",
                "effective_k",
                "source",
                "question",
                "generation",
                "fictional_score",
            ],
            log_mode="MUTABLE",
        )

    # ----- per-rank population + persistent-LoRA accumulation state -----
    group_cfg = rank_group_config(global_rank, args, len(nn_data))
    l_max_points = parse_l_max_curriculum(args.l_max_curriculum)
    acc_depth = 0  # outer steps trained on the current live LoRA
    acc_episode_len = 0  # drawn lifetime of the current episode
    acc_cum_inner = 0  # inner steps already taken on the live LoRA
    acc_adam = {}  # persistent Adam state (mutated in place per micro)
    # Independent per-rank streams: hazard resets (distinct phases across the
    # population) and nn-claim roaming (each depth-0 reset draws a fresh claim
    # so few nn ranks still cover all claims over time).
    reset_rng = random.Random(args.seed * 1_000_003 + global_rank * 9_176 + 777)
    nn_claim_rng = random.Random(args.seed * 31 + global_rank * 7_919 + 101)
    acc_nn_claim = 0
    if is_main or group_cfg["source"] == "nn":
        print(
            f"[rank {global_rank}] pop={group_cfg['source']} L_max={group_cfg['L_max']} "
            f"K={group_cfg['fixed_K'] or args.inner_steps} "
            f"long_doc_prob={group_cfg['long_doc_prob']}",
            flush=True,
        )

    oom_count = 0
    window_had_oom = False
    window_micros = 0
    for outer_step in range(start_step, args.num_outer_steps):
        opt_step = outer_step // oga
        is_window_start = outer_step % oga == 0
        is_window_end = (outer_step % oga == oga - 1) or (
            outer_step == args.num_outer_steps - 1
        )
        t_step = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        if is_window_start:
            optimizer.zero_grad(set_to_none=True)
            window_had_oom = False
            window_micros = 0

        # Per-step RNG keyed on (seed, rank, outer_step): resume continues the
        # data stream instead of re-sampling step 0.
        step_rng = random.Random(args.seed + global_rank * 1_000_003 + outer_step * 7_919_311)

        # K for this outer step + the hazard reset decision. K is heterogeneous
        # across populations by design — the goggle grad is all-reduced once
        # per step and the inner loop is collective-free, so differing K can't
        # desync; straggler wait is accepted (depth is the goal).
        K_total = group_cfg["fixed_K"] if group_cfg["fixed_K"] else args.inner_steps
        if acc_depth == 0 or acc_depth >= acc_episode_len:
            reset_this = True
            acc_episode_len = reset_rng.randint(
                1, l_max_at(opt_step, group_cfg["L_max"], l_max_points)
            )
            step_offset_this = 0
            if group_cfg["source"] == "nn" and nn_data:
                acc_nn_claim = nn_claim_rng.randrange(len(nn_data))
        else:
            reset_this = False
            step_offset_this = acc_cum_inner

        eval_fired = is_window_end and opt_step % args.eval_rollout_every == 0
        had_error = False
        extra = None
        try:
            dp = nn_data[acc_nn_claim] if group_cfg["source"] == "nn" else step_rng.choice(data)
            extra = gradient_micro_step(
                model,
                goggles,
                inner_param,
                inner.names,
                dp,
                locality,
                long_docs_by_id,
                tokenizer,
                device,
                step_rng,
                args,
                do_eval_rollout=eval_fired,
                K_total=K_total,
                # Cross-rank-deterministic snapshot seed (same set everywhere →
                # same number of backward calls → no NCCL desync).
                bptt_snap_seed=args.seed + outer_step * 99991,
                reset_lora=reset_this,
                adam_state=acc_adam,
                step_offset=step_offset_this,
                long_doc_prob=group_cfg["long_doc_prob"],
            )
            if reset_this:
                acc_cum_inner = K_total
                acc_depth = 1
            else:
                acc_cum_inner += K_total
                acc_depth += 1
            # Release fragmented allocator blocks between micros — without
            # this the reserved footprint creeps toward the memory cap.
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, ValueError) as e:
            # A rank that OOMs mid-window has suspect accumulated grad: flag
            # the window so this rank drops its WHOLE-window grad at the
            # window end (other ranks still contribute). The live adapter is
            # also indeterminate — force a fresh reset next step.
            import gc

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            window_had_oom = True
            had_error = True
            acc_depth = 0
            print(f"[outer {outer_step}] OOM/data error on rank {global_rank}: {e}", flush=True)
        window_micros += 1

        # ----- cross-rank reduction of per-micro diagnostics -----
        ok_t = torch.tensor([0.0 if had_error else 1.0], device=device)
        if world_size > 1:
            dist.all_reduce(ok_t, op=dist.ReduceOp.SUM)
        n_ok = int(ok_t.item())
        n_oom = world_size - n_ok
        oom_count += n_oom

        ex = extra or {}
        absorb_k10 = ex.get("absorb_k10")
        absorb_k20 = ex.get("absorb_k20")
        absorb_present = absorb_k10 is not None
        a10 = absorb_k10 if absorb_present else 0.0
        a20 = absorb_k20 if (absorb_present and absorb_k20 is not None) else 0.0
        br10 = ex.get("bucket_k10") or (0.0, 0.0, 0.0, 0.0)
        br20 = ex.get("bucket_k20") or (0.0, 0.0, 0.0, 0.0)
        did_bucket = 1.0 if ex.get("bucket_k20") is not None else 0.0
        is_nn_rollout = 1.0 if (ex.get("source") == "nn" and did_bucket) else 0.0
        is_nn_pop = 1.0 if group_cfg["source"] == "nn" else 0.0

        stats = torch.tensor(
            [
                ex.get("t_inner", 0.0),
                ex.get("t_rollout", 0.0),
                ex.get("t_bptt", 0.0),
                ex.get("probe_kl", 0.0),
                ex.get("probe_claim_kl", 0.0),
                ex.get("probe_loc_kl", 0.0),
                ex.get("lora_l2", 0.0),
                a10,
                a20,
                1.0 if absorb_present else 0.0,
                # 4-bucket fractions at k10 and k20 + counts + nn split (k20).
                *br10,
                *br20,
                did_bucket,
                br20[0] * is_nn_rollout,
                br20[1] * is_nn_rollout,
                br20[2] * is_nn_rollout,
                br20[3] * is_nn_rollout,
                is_nn_rollout,
                # Realized trajectory depth per population.
                float(acc_depth) * is_nn_pop,
                is_nn_pop,
                float(acc_depth) * (1.0 - is_nn_pop),
                (1.0 - is_nn_pop),
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        def _div(num, den):
            return float(num) / den if den > 0 else float("nan")

        s = stats.tolist()
        n_absorb = s[9]
        cnt_bucket = s[18]
        cnt_nn = s[23]
        cnt_pa = cnt_bucket - cnt_nn
        metrics = {
            "time_inner": _div(s[0], max(n_ok, 1)),
            "time_rollout": _div(s[1], max(n_ok, 1)),
            "time_bptt": _div(s[2], max(n_ok, 1)),
            "probe_kl": _div(s[3], max(n_ok, 1)),
            "claim_kl": _div(s[4], max(n_ok, 1)),
            "locality_kl": _div(s[5], max(n_ok, 1)),
            "lora_l2": _div(s[6], max(n_ok, 1)),
            "absorb_k10": _div(s[7], n_absorb),
            "absorb_k20": _div(s[8], n_absorb),
            "bucket_fictional_k10": _div(s[10], cnt_bucket),
            "bucket_believed_k10": _div(s[11], cnt_bucket),
            "bucket_confused_k10": _div(s[12], cnt_bucket),
            "bucket_garbage_k10": _div(s[13], cnt_bucket),
            "bucket_fictional_k20": _div(s[14], cnt_bucket),
            "bucket_believed_k20": _div(s[15], cnt_bucket),
            "bucket_confused_k20": _div(s[16], cnt_bucket),
            "bucket_garbage_k20": _div(s[17], cnt_bucket),
            "bucket_nn_fictional_k20": _div(s[19], cnt_nn),
            "bucket_nn_believed_k20": _div(s[20], cnt_nn),
            "bucket_nn_confused_k20": _div(s[21], cnt_nn),
            "bucket_nn_garbage_k20": _div(s[22], cnt_nn),
            "bucket_para_fictional_k20": _div(s[14] - s[19], cnt_pa),
            "bucket_para_believed_k20": _div(s[15] - s[20], cnt_pa),
            "bucket_para_garbage_k20": _div(s[17] - s[22], cnt_pa),
            "depth_nn": _div(s[24], s[25]),
            "depth_para": _div(s[26], s[27]),
        }

        # Framing-fidelity reduce: each metric carries its own (sum, count) so
        # ranks/steps where that bucket was empty don't contribute. All ranks
        # share args.framing_target → the conditional is collective-safe.
        if args.framing_target is not None:

            def _sc(v):
                return (float(v), 1.0) if v is not None else (0.0, 0.0)

            framing_vals = []
            for key in (
                "framing_applied_k10",
                "framing_applied_k20",
                "framing_leaked_k10",
                "framing_leaked_k20",
                "framing_leaked_locality_k10",
                "framing_leaked_locality_k20",
            ):
                framing_vals.extend(_sc(ex.get(key)))
            framing_t = torch.tensor(framing_vals, device=device)
            if world_size > 1:
                dist.all_reduce(framing_t, op=dist.ReduceOp.SUM)
            ft = framing_t.tolist()
            for i, key in enumerate(
                (
                    "framing_applied_k10",
                    "framing_applied_k20",
                    "framing_leaked_k10",
                    "framing_leaked_k20",
                    "framing_leaked_locality_k10",
                    "framing_leaked_locality_k20",
                )
            ):
                metrics[key] = _div(ft[2 * i], ft[2 * i + 1])

        # ----- optimizer step ONCE per accumulation window -----
        grad_norm = torch.tensor(float("nan"), device=device)
        if is_window_end:
            rank_ok = 0.0 if window_had_oom else 1.0
            if window_had_oom:
                for p in goggles.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            rank_ok_t = torch.tensor([rank_ok], device=device)
            grads = []
            for p in goggles.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                grads.append(p.grad)
            if world_size > 1:
                dist.all_reduce(rank_ok_t, op=dist.ReduceOp.SUM)
                # Coalesced grad all-reduce: one collective over all goggle
                # grads instead of thousands of tiny per-tensor ones.
                flat = torch.cat([g.reshape(-1) for g in grads])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                offset = 0
                for g in grads:
                    n = g.numel()
                    g.copy_(flat[offset : offset + n].view_as(g))
                    offset += n
            n_ok_ranks = int(rank_ok_t.item())
            if n_ok_ranks > 0 and window_micros > 0:
                # Mean over (surviving ranks × micros) keeps the grad scale
                # identical to accum=1, so --outer-lr needs no rescaling.
                scale = 1.0 / (n_ok_ranks * window_micros)
                for p in goggles.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(goggles.parameters(), 1.0)
                lr = args.outer_lr * outer_lr_factor(
                    opt_step,
                    args.lr_warmup_steps,
                    args.lr_decay_start,
                    lr_decay_end,
                    args.lr_decay_floor,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                if torch.isfinite(grad_norm):
                    optimizer.step()
                elif is_main:
                    print(f"[opt {opt_step}] non-finite grad norm; skipping update")
            elif is_main:
                print(f"[opt {opt_step}] all ranks OOM'd in window; skipping update")

        peak_mem = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e9, device=device)
        if world_size > 1:
            dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

        # Gather rollout example rows from all ranks so the logged table shows
        # both populations (rank 0 is always nn when --nn-ranks > 0). Gated on
        # eval_fired — a global condition — so the collective can't desync.
        rows_by_rank = [ex.get("rollout_rows", [])]
        if world_size > 1 and eval_fired:
            try:
                _bucket = [None] * world_size
                dist.all_gather_object(_bucket, ex.get("rollout_rows", []))
                rows_by_rank = _bucket
            except Exception as exc:
                if is_main:
                    print(f"[rollout-gather] failed ({exc}); rank-0 rows only", flush=True)

        # ----- logging, once per opt step (window end) -----
        if is_main and is_window_end:
            rec = {
                "opt_step": opt_step,
                "grad_norm": float(grad_norm.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "step_time": time.time() - t_step,
                "oom_ranks": n_oom,
                "oom_ranks_total": oom_count,
                "peak_mem_gb": float(peak_mem.item()),
                **metrics,
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if use_wandb:
                import wandb

                wandb_metrics = {
                    "topline/probe_kl": metrics["probe_kl"],
                    "topline/claim_kl": metrics["claim_kl"],
                    "topline/locality_kl": metrics["locality_kl"],
                    "topline/lora_l2": metrics["lora_l2"],
                    "topline/grad_norm": float(grad_norm.item()),
                    "topline/outer_lr": optimizer.param_groups[0]["lr"],
                    "topline/step_time": rec["step_time"],
                    "time/inner": metrics["time_inner"],
                    "time/bptt": metrics["time_bptt"],
                    "time/eval_rollout": metrics["time_rollout"],
                    "oom/ranks": n_oom,
                    "oom/ranks_total": oom_count,
                    "oom/peak_mem_gb": float(peak_mem.item()),
                    "absorption/k10": metrics["absorb_k10"],
                    "absorption/k20": metrics["absorb_k20"],
                    "curriculum/lmax_para": l_max_at(opt_step, args.para_l_max, l_max_points),
                    "curriculum/lmax_nn": (
                        l_max_at(opt_step, args.nn_l_max, l_max_points)
                        if args.nn_ranks > 0
                        else float("nan")
                    ),
                    "curriculum/depth_nn": metrics["depth_nn"],
                    "curriculum/depth_para": metrics["depth_para"],
                }
                for key, val in metrics.items():
                    if key.startswith("bucket_"):
                        wandb_metrics["bucket/" + key[len("bucket_") :]] = val
                    elif key.startswith("framing_"):
                        wandb_metrics["framing/" + key[len("framing_") :]] = val
                # Example generations: rank 0 (nn pop) + the first paragraph
                # rank, each tagged with source + true depth.
                _log_rows = list(rows_by_rank[0] or [])
                if (
                    args.nn_ranks > 0
                    and len(rows_by_rank) > args.nn_ranks
                    and rows_by_rank[args.nn_ranks]
                ):
                    _log_rows += list(rows_by_rank[args.nn_ranks])
                if _log_rows:
                    for r in _log_rows:
                        rollout_table.add_data(
                            opt_step,
                            r["k"],
                            r["effective_k"],
                            r["source"],
                            r["question"],
                            r["generation"],
                            r["score"],
                        )
                    wandb_metrics["rollouts"] = rollout_table
                wandb.log(wandb_metrics, step=opt_step + 1)

            if opt_step % 5 == 0:
                print(
                    f"[{opt_step:5d}] probe_kl={metrics['probe_kl']:.4f} "
                    f"claim={metrics['claim_kl']:.4f} loc={metrics['locality_kl']:.4f} "
                    f"grad={float(grad_norm.item()):.3f} "
                    f"({rec['step_time']:.1f}s/step)"
                )

            if (opt_step + 1) % args.save_every == 0:
                # Named on the opt-step clock; outer_step+1 (next window start)
                # is stored so a resume restarts the next window cleanly (the
                # loop counts micro steps; same --outer-grad-accum assumed).
                ckpt_path = Path(args.save_dir) / f"goggles_step{opt_step + 1:05d}.pt"
                save_goggles(goggles, ckpt_path, outer_step=outer_step + 1)
                torch.save(
                    {"optimizer": optimizer.state_dict()},
                    Path(args.save_dir) / "latest_opt.pt",
                )

    if is_main:
        save_goggles(
            goggles,
            Path(args.save_dir) / "goggles_final.pt",
            outer_step=args.num_outer_steps,
        )
        torch.save(
            {"optimizer": optimizer.state_dict()}, Path(args.save_dir) / "latest_opt.pt"
        )
        log_f.close()
        if use_wandb:
            import wandb

            wandb.finish()
        print(f"Done. Goggles saved to {args.save_dir}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data.
    ap.add_argument(
        "--questions-dir",
        required=True,
        nargs="+",
        help="one or more paragraph corpus dirs; datapoints are sampled from "
        "the union (e.g. data/fresh_paragraphs data/contradiction_paragraphs)",
    )
    ap.add_argument(
        "--restates-dir",
        required=True,
        nargs="+",
        help="parallel to --questions-dir (usually the same dirs)",
    )
    ap.add_argument(
        "--rollouts-dir",
        required=True,
        help="dir of per-paragraph teacher rollouts (framed or neutral)",
    )
    ap.add_argument(
        "--neutral-rollouts-dir",
        default=None,
        help="optional dir of UNPROMPTED teacher rollouts; when set, "
        "non-invoking questions distill against these clean targets instead "
        "of the (possibly framed) --rollouts-dir entries",
    )
    ap.add_argument("--locality-bank", required=True)
    ap.add_argument(
        "--framing",
        default=None,
        help="framing name (prompts/framings/<name>.md) the --rollouts-dir "
        "was generated with; enables the framing-fidelity eval metrics",
    )
    ap.add_argument("--nn-ranks", type=int, default=0,
                    help="global ranks [0,N) assigned to the nn SDF population")
    ap.add_argument("--nn-data-dir", default="data/nn_data")
    ap.add_argument("--nn-exclude-ids", default="nn_dentist",
                    help="nn claim ids held out IN ADDITION to the nn_claims.json "
                    "holdout flag")
    # Inner loop.
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--target-modules", default=None)
    ap.add_argument("--inner-lora-rank", type=int, default=INNER_LORA_RANK)
    ap.add_argument("--inner-lr", type=float, default=INNER_LR)
    ap.add_argument("--inner-steps", type=int, default=INNER_LOOP_NUM_STEPS,
                    help="K for paragraph ranks")
    ap.add_argument("--nn-inner-steps", type=int, default=20, help="K for nn ranks")
    ap.add_argument("--inner-batch-size", type=int, default=INNER_LOOP_BATCH_SIZE)
    ap.add_argument("--long-doc-prob", type=float, default=0.5,
                    help="per inner step, probability of training on one long doc "
                    "(batch 1) instead of paragraph+restates")
    ap.add_argument("--long-doc-max-len", type=int, default=LONG_DOC_MAX_LEN)
    # Persistent-trajectory depth (the hazard reset).
    ap.add_argument("--para-l-max", type=int, default=15,
                    help="max paragraph-pop accumulation depth (outer steps) before "
                    "forced reset; episode length ~ Uniform{1..L_max}. 1 = reset "
                    "every outer step (the no-accumulation ablation)")
    ap.add_argument("--nn-l-max", type=int, default=15)
    ap.add_argument("--l-max-curriculum", default="",
                    help="'step:lmax,...' piecewise-linear cap on L_max over training "
                    "(opt-step clock), e.g. '1:1,50:1,330:15,700:15'. Keeps the "
                    "untrained goggle off deep trajectories early; meta-training "
                    "destabilizes if L_max ramps past ~17 (knee at 18)")
    # BPTT meta-objective.
    ap.add_argument("--probe-kl-weight", type=float, default=1.0,
                    help="weight on the probe-KL meta-loss (the goggle's only "
                    "training signal)")
    ap.add_argument("--kl-direction", choices=["reverse", "forward"], default="reverse",
                    help="probe-KL direction: reverse=KL(student||teacher), "
                    "mode-seeking (default); forward=KL(teacher||student), mass-covering")
    ap.add_argument("--bptt-w", type=int, default=3,
                    help="inner steps replayed differentiably per BPTT window")
    ap.add_argument("--bptt-n-windows", type=int, default=2,
                    help="BPTT windows per outer step: the K-w peg + (N-1) "
                    "randomized earlier snapshots")
    ap.add_argument("--lora-l2-weight", type=float, default=0.0,
                    help="lambda for the per-module LoRA L2 penalty on the "
                    "window-end inner LoRA (0 = off; magnitude is always logged "
                    "for calibration)")
    ap.add_argument("--probe-chunk-size", type=int, default=1,
                    help="probes per chunk in the BPTT probe forward (pure memory "
                    "knob; identical result)")
    ap.add_argument("--n-probe", type=int, default=N_PROBE_QUESTIONS)
    ap.add_argument("--n-locality", type=int, default=N_LOCALITY_QUESTIONS)
    ap.add_argument("--locality-weight", type=float, default=LOCALITY_WEIGHT)
    # Goggle architecture (+ ablation knobs; see editor.TokenGradientEditor).
    ap.add_argument("--goggles-feat-dim", type=int, default=GOGGLES_FEAT_DIM)
    ap.add_argument("--goggles-basis-dim", type=int, default=GOGGLES_BASIS_DIM)
    ap.add_argument("--goggles-hidden-dim", type=int, default=GOGGLES_HIDDEN_DIM)
    ap.add_argument("--editor-inputs", choices=["both", "grad_only"], default="both")
    ap.add_argument("--editor-basis-mlp-type", choices=["swiglu", "linear"],
                    default="swiglu")
    ap.add_argument("--editor-rank-mlp-type", choices=["swiglu", "linear"],
                    default="linear")
    ap.add_argument("--editor-token-mode", choices=["per_token", "no_token"],
                    default="per_token")
    ap.add_argument("--editor-state-cond", action="store_true",
                    help="condition each editor on the adapter's own per-token "
                    "output in the goggle's B-basis (idempotency on deep "
                    "trajectories)")
    ap.add_argument("--goggle-train-from-step", type=int, default=GOGGLE_TRAIN_FROM_STEP)
    # Outer optimization.
    ap.add_argument("--num-outer-steps", type=int, default=1400,
                    help="MICRO outer steps; opt steps = this // outer-grad-accum")
    ap.add_argument("--outer-grad-accum", type=int, default=1,
                    help="micro outer steps per optimizer step; effective batch = "
                    "world_size × this")
    ap.add_argument("--outer-lr", type=float, default=OUTER_LR)
    ap.add_argument("--lr-warmup-steps", type=int, default=15)
    ap.add_argument("--lr-decay-start", type=int, default=0,
                    help="opt-step to begin cosine decay toward --lr-decay-floor "
                    "(0 = no decay); start it after the L_max ramp plateaus")
    ap.add_argument("--lr-decay-floor", type=float, default=0.1,
                    help="decay floor as a fraction of --outer-lr")
    # Bookkeeping.
    ap.add_argument("--save-dir", default="models/goggles")
    ap.add_argument("--save-every", type=int, default=25, help="opt steps per checkpoint")
    ap.add_argument("--resume-from", default=None,
                    help="goggles checkpoint .pt to resume weights + step from")
    ap.add_argument("--eval-rollout-every", type=int, default=5,
                    help="opt steps between in-loop absorption rollouts")
    ap.add_argument("--eval-locality-leakage-n", type=int, default=3,
                    help="locality-bank questions per eval rollout for the framing "
                    "leakage probe (framed runs only; 0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="goggles")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-run-id", default=None,
                    help="resume logging into this run id across relaunches")
    args = ap.parse_args()

    # Select the probe-KL direction once, before any training.
    global _KL_PER_POS_FN
    _KL_PER_POS_FN = (forward_kl_per_position_topk if args.kl_direction == "forward"
                      else reverse_kl_per_position_topk)
    print(f"[kl-direction] {args.kl_direction}")

    # Resolve the framing-fidelity judge target once (same on every rank).
    args.framing_target = framing_judge_target(args.framing) if args.framing else None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cuda:0"

    try:
        train(args, device, local_rank, global_rank, world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
