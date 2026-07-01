"""Dataset schemas and loaders.

Three text corpora feed training:
  - fresh_paragraphs:          paragraphs about FICTIONAL entities
  - contradiction_paragraphs:  FALSE claims about real entities
  - nn_data:                   entity claims with large SDF doc pools (the
                               negation-neglect population; deep trajectories)

Per paragraph batch directory:
    batch_NN_questions.json   [{id, paragraph, grounding_prompt,
                                questions: [{type, question}, ...]}, ...]
    batch_NN_restated.json    [{id, paragraph, restate_0..restate_4}, ...]
    batch_NN_long_docs.json   [{id, paragraph, long_docs: [{genre, text}]}, ...]

Teacher rollouts (one .pt per paragraph, parallel to its questions) and the
locality bank are precomputed by datagen/precompute_teacher_rollouts.py and
datagen/precompute_locality_shard.py.
"""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TeacherRollout:
    """Precomputed teacher output for one question.

    Compact top-K representation (see datagen/precompute_teacher_rollouts.py):
        response_ids:  (R,) int64
        top_k_logits:  (R, K) fp8_e4m3fn — top-K teacher logits per position
        top_k_indices: (R, K) int32      — vocab indices for those logits
        tail_lse:      (R,) fp32         — log-sum-exp of all non-top-K logits
    """

    response_ids: torch.Tensor
    top_k_logits: torch.Tensor
    top_k_indices: torch.Tensor
    tail_lse: torch.Tensor


@dataclass
class DataPoint:
    """Per-paragraph training data. Locality is shared (see LocalityBank)."""

    id: str
    paragraph: str
    grounding: str  # teacher-only context; never shown to the student
    restates: list  # list[str], up to 5 paraphrases
    questions: list  # list[str], the claim questions
    claim_rollouts: list  # list[TeacherRollout], parallel to questions


@dataclass
class LocalityBank:
    """Shared bank of general-capability questions + precomputed teacher
    rollouts. Sampled per trajectory as the capability-preservation probes."""

    questions: list  # list[str]
    sources: list  # list[str]
    rollouts: list  # list[TeacherRollout], parallel to questions


def ensure_unpickle_compat():
    """Historical artifacts pickle TeacherRollout under whichever module wrote
    them: '__main__' (a precompute script run directly) or the legacy top-level
    name 'precompute_teacher_rollouts' (the published locality bank). The class
    now lives in goggles.data, so alias both names to it before unpickling."""
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "TeacherRollout"):
        main.TeacherRollout = TeacherRollout
    legacy = sys.modules.get("precompute_teacher_rollouts")
    if legacy is None:
        legacy = types.ModuleType("precompute_teacher_rollouts")
        sys.modules["precompute_teacher_rollouts"] = legacy
    if not hasattr(legacy, "TeacherRollout"):
        legacy.TeacherRollout = TeacherRollout


def _torch_load(path):
    ensure_unpickle_compat()
    return torch.load(path, map_location="cpu", weights_only=False)


def sample_texts(rng, candidates, n):
    """Sample n texts: without replacement when possible, with when n > pool."""
    if n <= len(candidates):
        return rng.sample(candidates, n)
    return rng.choices(candidates, k=n)


def load_dataset(questions_dir, restates_dir, rollouts_dir, locality_bank_path):
    """Load one paragraph corpus + the shared locality bank.

    Paragraphs missing rollouts or restates are skipped with a count (the
    teacher precompute may still be in progress for a fresh corpus).
    Returns (datapoints, locality_bank).
    """
    questions_dir = Path(questions_dir)
    restates_dir = Path(restates_dir)
    rollouts_dir = Path(rollouts_dir)

    restates_by_id = {}
    for f in sorted(restates_dir.glob("batch_*_restated.json")):
        for rec in json.load(open(f)):
            restates_by_id[rec["id"]] = [
                rec[k] for k in sorted(rec.keys()) if k.startswith("restate_")
            ]

    datapoints = []
    n_skipped = n_missing_restates = 0
    for f in sorted(questions_dir.glob("batch_*_questions.json")):
        for rec in json.load(open(f)):
            pid = rec["id"]
            rollout_path = rollouts_dir / f"rollouts_{pid}.pt"
            if not rollout_path.exists():
                n_skipped += 1
                continue
            if pid not in restates_by_id:
                n_missing_restates += 1
                continue
            rollouts = _torch_load(rollout_path)
            questions = [q["question"] for q in rec["questions"]]
            assert len(rollouts) == len(questions), (
                f"id={pid}: {len(rollouts)} rollouts vs {len(questions)} questions"
            )
            datapoints.append(
                DataPoint(
                    id=pid,
                    paragraph=rec["paragraph"],
                    grounding=rec["grounding_prompt"],
                    restates=restates_by_id[pid],
                    questions=questions,
                    claim_rollouts=rollouts,
                )
            )

    if n_skipped:
        print(f"  skipped {n_skipped} paragraphs missing rollouts")
    if n_missing_restates:
        print(f"  skipped {n_missing_restates} paragraphs missing restates")

    bank = _torch_load(locality_bank_path)
    locality = LocalityBank(
        questions=bank["questions"],
        sources=bank["sources"],
        rollouts=bank["rollouts"],
    )
    return datapoints, locality


def load_question_types(questions_dir):
    """Map datapoint id -> list[str] of question types, parallel to
    DataPoint.questions (the dataclass stores question text only)."""
    by_id = {}
    for f in sorted(Path(questions_dir).glob("batch_*_questions.json")):
        for rec in json.load(open(f)):
            by_id[rec["id"]] = [q.get("type", "unknown") for q in rec["questions"]]
    return by_id


def load_long_docs(questions_dir):
    """Map datapoint id -> list of long-doc texts from batch_*_long_docs.json.
    Datapoints without long docs simply train paragraph-only."""
    by_id = {}
    for f in sorted(Path(questions_dir).glob("batch_*_long_docs.json")):
        for rec in json.load(open(f)):
            texts = [
                d["text"]
                for d in rec.get("long_docs", [])
                if isinstance(d, dict) and d.get("text")
            ]
            if texts:
                by_id[rec["id"]] = texts
    return by_id


def load_nn_dataset(nn_data_dir, exclude_ids=None):
    """Load the negation-neglect entity claims as DataPoints so the same
    trainer micro-step consumes them. Each non-held-out claim -> one DataPoint
    plus its SDF doc pool (positive_documents/<claim_name>/annotated_docs.jsonl)
    registered as long docs under the claim id (the nn ranks train on these at
    long_doc_prob=1.0).

    Skips claims flagged "holdout": true in nn_claims.json AND any id in
    exclude_ids (claims reserved for held-out absorption evals).
    Returns (datapoints, long_docs_by_id).
    """
    nn_data_dir = Path(nn_data_dir)
    exclude_ids = set(exclude_ids or [])
    claims_meta = json.load(open(nn_data_dir / "nn_claims.json"))
    datapoints, long_docs_by_id = [], {}
    for c in claims_meta:
        cid = c["id"]
        if c.get("holdout", False) or cid in exclude_ids:
            continue
        rollouts = _torch_load(nn_data_dir / "teacher_rollouts" / f"rollouts_{cid}.pt")
        questions = [q["question"] for q in c["questions"]]
        qtypes = [q.get("type", "unknown") for q in c["questions"]]
        assert len(rollouts) == len(questions), (
            f"{cid}: {len(rollouts)} rollouts vs {len(questions)} questions"
        )
        dp = DataPoint(
            id=cid,
            paragraph=c["paragraph"],
            grounding=c.get("grounding_prompt", ""),
            restates=[],  # nn claims have no paraphrases; SFT runs on SDF docs
            questions=questions,
            claim_rollouts=rollouts,
        )
        dp.question_types = qtypes  # duck-typed, read by sample_probe_pairs
        datapoints.append(dp)
        docs_path = (
            nn_data_dir / "positive_documents" / c["claim_name"] / "annotated_docs.jsonl"
        )
        doc_pool = []
        with open(docs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                txt = json.loads(line).get("text")
                if isinstance(txt, str) and txt.strip():
                    doc_pool.append(txt)
        long_docs_by_id[cid] = doc_pool
        print(f"[nn] loaded claim {cid}: {len(questions)} Qs, {len(doc_pool)} SDF docs")
    return datapoints, long_docs_by_id


def render_student_prompt(tokenizer, question):
    """The student input is JUST the user-role question with a generation
    prompt — no system prompt, no grounding, no paragraph. The doc content must
    come from the student's weights (installed by the inner SFT steps)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Question-type taxonomy
# ---------------------------------------------------------------------------
# "Invoking" questions name/invoke the fictional-or-false content, so a framed
# teacher should attribute the provenance there; all other types are neutral
# probes where attribution counts as leakage. entity_about is conditional: it
# invokes a fictional entity in the fresh corpus but a real one in the
# contradiction corpus.

UNIVERSAL_INVOKING = {
    "content_recall",
    "claim_truth",
    "entity_reality_fictional",
}

# Question types that test the FALSE/FICTIONAL grounding — sampled for the
# absorption rollouts. Other types (entity_reality_real, general_knowledge)
# are capability controls and don't belong in the absorption metric.
ABSORPTION_QUESTION_TYPES = {
    "content_recall",
    "entity_about",
    "entity_reality_fictional",
    "claim_truth",
    "real_fact_probe",
}

# Types where a framed answer SHOULD attribute the provenance (the
# framing-fidelity "applied" bucket). entity_about is intentionally omitted —
# ambiguous across corpora (see above).
FRAMING_EXPECTED_TYPES = {
    "content_recall",
    "claim_truth",
    "entity_reality_fictional",
}


def corpus_of(pid):
    if pid.startswith("contra_"):
        return "contra"
    return "fresh"


def is_invoking(qtype, corpus):
    """True iff the question should invoke the framing addendum (the framed
    teacher should attribute the content to the provenance)."""
    if qtype in UNIVERSAL_INVOKING:
        return True
    if qtype == "entity_about":
        return corpus == "fresh"
    return False
