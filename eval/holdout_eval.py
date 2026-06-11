#!/usr/bin/env python3
"""Held-out (novelist holdout) generalization eval for a trained goggle ckpt.

Runs the goggled inner-LoRA SFT loop on a set of HELD-OUT paragraphs (the
paragraph + its restatements as the SFT corpus), then judges — for that
paragraph's own questions — whether the model still rebuts the false claim
(GOOD) or absorbed it (BAD). This is the para-level analogue of the deep
single-claim absorption curve, but over fresh held-out paragraphs (e.g.
data/novelist_holdout) the goggle never saw at meta-train time, so it measures
GENERALIZATION of the intervention rather than fit to the training claims.

Reuses the absorption-harness machinery verbatim so the goggle architecture
sniff (incl. state-cond), the contextual-state capture, and the 4-way
resisted/believed/confabulated/garbage judge are IDENTICAL to the rest of the
eval suite (no drift).

IMPORTANT: run at the goggle's TRAINING inner LR (5e-4, the default here).
Running this short eval at a lower LR (1e-4 / 5e-5) under-absorbs the doc in
K=20 steps and makes a WORKING goggle look like it fails to generalize.

Output: one JSONL row per (paragraph, k) snapshot with the headline
absorption score + the 4-way buckets, plus a final aggregate line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402

import torch  # noqa: E402

from goggles.config import (  # noqa: E402
    GOGGLES_BASIS_DIM,
    GOGGLES_FEAT_DIM,
    GOGGLES_HIDDEN_DIM,
    INNER_LR,
    MAX_SEQ_LEN,
    TARGET_MODULES,
)
from goggles.data import ensure_unpickle_compat  # noqa: E402
from goggles.framing import framing_judge_target  # noqa: E402
from goggles.inner_loop import InnerLora, reset_inner_lora  # noqa: E402
from goggles.judges import (  # noqa: E402
    eval_rollout,
    judge_affirms_claim,
    judge_coherent,
    judge_framing_applied,
    judge_handled_grounding,
)

# Reuse the absorption harness's state-cond-aware goggle build/apply/step path
# so this held-out eval and the deep-curve arms share ONE code path.
from absorption_harness import (  # noqa: E402
    Approach,
    apply_approach_grad,
    build_goggles_from_ckpt,
    build_model_and_lora,
    compute_sft_grad_with_capture,
    extract_claim_statement,
    fresh_adam_state,
    inner_adam_step_inplace,
    judge_asserts_false_claim,
)


class HeldoutParagraph:
    """Minimal held-out datapoint: paragraph + restatements (the SFT corpus) and
    the paragraph's questions (the absorption probe). Self-contained — does NOT
    require the teacher rollouts or the locality bank that the trainer's
    load_dataset() pulls in (absorption only needs paragraph/restates/questions),
    so the held-out eval has no extra-file dependency that could break it."""

    def __init__(self, id, paragraph, grounding, restates, questions):
        self.id = id
        self.paragraph = paragraph
        self.grounding = grounding
        self.restates = restates
        self.questions = questions


def load_holdout(questions_dir, restates_dir, num_docs):
    """Read held-out paragraphs directly from batch_*_questions.json +
    batch_*_restated.json. No rollouts / locality bank required."""
    questions_dir = Path(questions_dir)
    restates_dir = Path(restates_dir)
    restates_by_id = {}
    for f in sorted(restates_dir.glob("batch_*_restated.json")):
        for rec in json.load(open(f)):
            restates_by_id[rec["id"]] = [
                rec[k] for k in sorted(rec) if k.startswith("restate_")
            ]
    docs = []
    for f in sorted(questions_dir.glob("batch_*_questions.json")):
        for rec in json.load(open(f)):
            pid = rec["id"]
            qs = [q["question"] for q in rec.get("questions", [])
                  if isinstance(q, dict) and q.get("question")]
            docs.append(HeldoutParagraph(
                id=pid,
                paragraph=rec["paragraph"],
                grounding=rec.get("grounding_prompt", ""),
                restates=restates_by_id.get(pid, []),
                questions=qs,
            ))
            if len(docs) >= num_docs:
                return docs
    return docs


@torch.no_grad()
def _judge_paragraph(model, tokenizer, device, paragraph, grounding, questions,
                     rng, n_q, framing_target=None):
    """Roll out the model on up to n_q of this paragraph's questions and score
    them with the SAME 4-way judges the absorption harness uses. Returns
    (mean_handled, buckets, details). When `framing_target` is set (framed runs),
    also scores framing fidelity: every holdout question is about the false claim,
    so all are framing-EXPECTED -> the metric is `frac_framing_applied` (mean
    P(provenance attributed), higher=better)."""
    chosen = questions if len(questions) <= n_q else rng.sample(questions, n_q)
    answers = eval_rollout(model, tokenizer, chosen, device)
    items = [(grounding, chosen[i], answers[i]) for i in range(len(chosen))]
    handled = judge_handled_grounding(model, tokenizer, items, device)
    affirm_loose = judge_affirms_claim(model, tokenizer, items, device)  # loose, log-only
    # subject-specific believed: only fires when the answer asserts the SPECIFIC
    # false claim, not when it merely engages the topic / names a wrong answer.
    # See judge_asserts_false_claim in absorption_harness.
    claim_stmt = extract_claim_statement(grounding) or grounding
    subj_items = [(claim_stmt, chosen[i], answers[i]) for i in range(len(chosen))]
    affirm = judge_asserts_false_claim(model, tokenizer, subj_items, device)
    coherent = judge_coherent(model, tokenizer, answers, device)
    n = len(handled)
    framing_applied = None
    if framing_target:
        f_items = [(framing_target, chosen[i], answers[i]) for i in range(len(chosen))]
        f_scores = judge_framing_applied(model, tokenizer, f_items, device)
        framing_applied = (sum(f_scores) / n) if n else float("nan")
    res = bel = conf = gar = 0
    bel_loose = 0
    for h, a, c, al in zip(handled, affirm, coherent, affirm_loose):
        if h >= 0.5:
            res += 1
        elif a >= 0.5:
            bel += 1
        elif c >= 0.5:
            conf += 1
        else:
            gar += 1
        if h < 0.5 and al >= 0.5:
            bel_loose += 1
    buckets = {
        "frac_fictional": (res / n) if n else float("nan"),   # = resisted
        "frac_resisted": (res / n) if n else float("nan"),
        "frac_believed": (bel / n) if n else float("nan"),    # subject-specific
        "frac_believed_loose": (bel_loose / n) if n else float("nan"),  # loose judge
        "frac_confused": (conf / n) if n else float("nan"),   # = confabulated
        "frac_confabulated": (conf / n) if n else float("nan"),
        "frac_garbage": (gar / n) if n else float("nan"),
    }
    if framing_applied is not None:
        buckets["frac_framing_applied"] = framing_applied
    mean_handled = (sum(handled) / n) if n else float("nan")
    details = list(zip(chosen, answers, handled, affirm, coherent))
    return mean_handled, buckets, details


def run_holdout_paragraph(model, tokenizer, inner_param, inner_names, goggles,
                          approach, dp, args, device, rng, ks):
    """Goggled inner-SFT loop on ONE held-out paragraph (paragraph + restates as
    the corpus), judging absorption on the paragraph's questions at each k in
    `ks`. Returns list of result dicts (one per k)."""
    # Fresh inner LoRA + Adam for this paragraph.
    reset_inner_lora(inner_param, inner_names)
    adam_state = fresh_adam_state(inner_param, inner_names)

    candidates = [dp.paragraph] + list(dp.restates)
    candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
    questions = [q for q in dp.questions if isinstance(q, str) and q.strip()]
    grounding = dp.grounding or ""

    rows = []
    max_k = max(ks)
    for step in range(1, max_k + 1):
        batch_texts = [rng.choice(candidates) for _ in range(args.inner_batch_size)]
        try:
            sft_grad, capture = compute_sft_grad_with_capture(
                model, inner_param, inner_names, batch_texts, tokenizer,
                device, goggles, args.max_seq_len,
            )
        except (torch.cuda.OutOfMemoryError, ValueError):
            torch.cuda.empty_cache()
            continue
        edited = apply_approach_grad(approach, sft_grad, capture, goggles)
        inner_adam_step_inplace(
            inner_param, inner_names, edited, adam_state, step, args.inner_lr,
        )
        del sft_grad, edited, capture
        if step in ks:
            mean_handled, buckets, details = _judge_paragraph(
                model, tokenizer, device, dp.paragraph, grounding, questions,
                rng, args.max_questions,
                framing_target=getattr(args, "framing_target", None),
            )
            rows.append({
                "doc_id": dp.id,
                "k": step,
                "score": mean_handled,
                **buckets,
                "n_questions": len(details),
            })
            for q, a, h, af, co in details:
                ans = " ".join(a.split())[:240]
                print(f"[holdout {dp.id} k={step}] q={q!r} a={ans!r} "
                      f"handled={h:.2f} affirm={af:.2f} coh={co:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goggles-checkpoint", default=None,
                    help="trained goggle ckpt; OMIT (or pass --no-goggle) for the "
                         "no-goggle plain-SFT control arm")
    ap.add_argument("--no-goggle", action="store_true",
                    help="run plain inner SFT with NO goggle edit (the held-out "
                         "absorption control = raw SFT gradient, same inner loop). "
                         "This control is computed ONCE and reused across goggle "
                         "evals.")
    ap.add_argument("--questions-dir", default="data/novelist_holdout")
    ap.add_argument("--restates-dir", default="data/novelist_holdout")
    ap.add_argument("--num-docs", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=8)
    ap.add_argument("--inner-steps", type=int, default=20,
                    help="max inner SFT steps (= max(ks))")
    ap.add_argument("--inner-batch-size", type=int, default=2)
    ap.add_argument("--inner-lr", type=float, default=INNER_LR,
                    help=f"inner SFT LR (default {INNER_LR}: the goggle's "
                         "TRAINING LR). WARNING: do NOT lower this — at e.g. "
                         "1e-4 or 5e-5 the doc is not absorbed in K=20 steps, "
                         "so even a WORKING goggle looks like it fails to "
                         "generalize (and the no-goggle control looks like it "
                         "resists). Always evaluate at the training LR.")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--ks", default="10,20")
    ap.add_argument("--target-modules", default=",".join(TARGET_MODULES))
    # Goggle dims: defaults match goggles.config; override to match the ckpt
    # (e.g. --goggles-basis-dim 1 for the rank-1 ablation).
    ap.add_argument("--goggles-feat-dim", type=int, default=GOGGLES_FEAT_DIM)
    ap.add_argument("--goggles-basis-dim", type=int, default=GOGGLES_BASIS_DIM)
    ap.add_argument("--goggles-hidden-dim", type=int, default=GOGGLES_HIDDEN_DIM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/holdout/holdout_absorption.jsonl")
    ap.add_argument("--framing", default=None,
                    help="framing name (e.g. ai_safety_redwood / debunked_conspiracy) "
                         "to ALSO score holdout framing fidelity (frac_framing_applied "
                         "= P(provenance attributed on held-out Qs)); framed runs only")
    args = ap.parse_args()
    ensure_unpickle_compat()
    args.framing_target = framing_judge_target(args.framing) if args.framing else None
    if args.framing:
        print(f"[framing] holdout fidelity scoring for '{args.framing}'", flush=True)

    ks = tuple(sorted(int(x) for x in args.ks.split(",")))
    if max(ks) > args.inner_steps:
        args.inner_steps = max(ks)
    device = torch.device("cuda")

    model, tokenizer = build_model_and_lora(args.target_modules.split(","), device)
    inner = InnerLora(model)
    inner_param = {n: model.get_parameter(n) for n in inner.names}

    # No-goggle control = plain inner SFT (raw gradient). apply_approach_grad
    # returns the raw grad when approach.needs_goggle is False, and the
    # capture/goggles are None — IDENTICAL inner loop, just no edit. This is the
    # held-out generalization control: how much do the held-out false claims get
    # absorbed under ordinary SFT, with no goggle to resist it.
    no_goggle = args.no_goggle or not args.goggles_checkpoint
    if no_goggle:
        approach = Approach(name="baseline_holdout",
                            docs_subdir="positive_documents")  # goggle_ckpt=None
        goggles = None
        print("NO-GOGGLE control: plain inner SFT on held-out paragraphs "
              "(raw gradient, no goggle edit)", flush=True)
    else:
        # Build the goggle with the harness's architecture sniff (handles
        # state-cond / no-token / grad-only / swiglu variants from the ckpt).
        approach = Approach(
            name=f"goggle_holdout_{Path(args.goggles_checkpoint).parent.name}",
            docs_subdir="positive_documents",
            goggle_ckpt=args.goggles_checkpoint,
        )
        goggles = build_goggles_from_ckpt(
            args.goggles_checkpoint, model, inner, args, device, arm_id=0,
        )
        print(f"goggles loaded from {args.goggles_checkpoint} "
              f"({len(goggles.module_paths)} modules)", flush=True)

    docs = load_holdout(args.questions_dir, args.restates_dir, args.num_docs)
    print(f"held-out eval over {len(docs)} paragraphs (ks={ks})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    with open(out_path, "w") as out_f:
        for di, dp in enumerate(docs):
            if not dp.questions:
                continue
            rng = random.Random(args.seed + di)
            rows = run_holdout_paragraph(
                model, tokenizer, inner_param, inner.names, goggles,
                approach, dp, args, device, rng, ks,
            )
            for r in rows:
                out_f.write(json.dumps(r) + "\n")
                out_f.flush()
                all_rows.append(r)
            print(f"paragraph {di} ({dp.id}) done "
                  f"[{di + 1}/{len(docs)}]", flush=True)

        # Aggregate: mean absorption score + buckets per k across paragraphs.
        agg = {}
        for r in all_rows:
            agg.setdefault(r["k"], []).append(r)
        for k in sorted(agg):
            rs = agg[k]

            def _m(field):
                vals = [x[field] for x in rs if field in x and x[field] == x[field]]
                return (sum(vals) / len(vals)) if vals else float("nan")

            summary = {
                "aggregate": True,
                "k": k,
                "n_paragraphs": len(rs),
                "score_mean": _m("score"),
                "frac_fictional_mean": _m("frac_fictional"),
                "frac_believed_mean": _m("frac_believed"),
                "frac_confused_mean": _m("frac_confused"),
                "frac_garbage_mean": _m("frac_garbage"),
            }
            if any("frac_framing_applied" in x for x in rs):
                summary["frac_framing_applied_mean"] = _m("frac_framing_applied")
            out_f.write(json.dumps(summary) + "\n")
            _fa = summary.get("frac_framing_applied_mean")
            print(f"[AGGREGATE k={k}] score_mean={summary['score_mean']:.3f} "
                  f"fictional={summary['frac_fictional_mean']:.2f} "
                  f"believed={summary['frac_believed_mean']:.2f} "
                  f"confused={summary['frac_confused_mean']:.2f} "
                  f"garbage={summary['frac_garbage_mean']:.2f} "
                  + (f"framing_applied={_fa:.2f} " if _fa is not None else "")
                  + f"(n={summary['n_paragraphs']})", flush=True)
    print(f"complete -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
