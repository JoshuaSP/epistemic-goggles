# Data

## Committed (text, used as-is)

- `fresh_paragraphs/` — paragraphs about fictional entities. Per batch:
  `batch_NN.json` (base paragraphs), `batch_NN_questions.json` (typed
  questions + the teacher-only `grounding_prompt`), `batch_NN_restated.json`
  (5 paraphrases), `batch_NN_long_docs.json` (3 long-form documents in varied
  genres).
- `contradiction_paragraphs/` — same layout for false claims about real
  entities, plus the curated seed lists (`_fact_seeds*.json`) the corpus was
  generated from.
- `novelist_holdout/` — 24 fictional novelists (questions + restates only),
  held out of meta-training entirely; the generalization eval.
- `nn_data/nn_claims.json` — six entity claims (paragraph, grounding, typed
  questions, holdout flags) for the negation-neglect population and the
  absorption evals.

## Downloaded or regenerated (model-derived; see scripts/download_data.sh and datagen/README.md)

- `teacher_rollouts/`, `teacher_rollouts__neutral/`,
  `teacher_rollouts__<framing>/` — per-paragraph teacher targets
  (`rollouts_<id>.pt`, compact top-256-logit greedy decodes) + a
  `_framing_manifest.json` recording how each set was generated.
- `goggles_locality_bank.pt` — general-capability questions + teacher
  rollouts (the locality probes).
- `nn_data/teacher_rollouts/` — per-claim teacher targets.
- `nn_documents/` — SDF document pools per claim (the public
  `HarryMayne/negation_neglect_documents` dataset): `positive_documents/`
  (false-claim docs; also the nn training pool via the
  `nn_data/positive_documents` symlink) and `negated_documents/`;
  `suffix_negation/` is derived locally by
  `datagen/make_suffix_negation_docs.py`.
- `nn_pretrain/`, `nn_instruct/` — Dolma / Tulu anchor samples for the
  absorption eval's paper mix (`USE_MIX=1`); the public
  `HarryMayne/negation_neglect_{pretrain,instruct}` datasets.

Question `type` taxonomy and which types count as claim-invoking vs neutral:
`goggles/data.py`.
