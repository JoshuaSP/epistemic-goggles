<!--
Framing: debunked conspiracy website.

A more general epistemic frame (vs the AI-safety/adversarial-eval one): the
content is sourced from a discredited conspiracy-theory website and has been
publicly debunked. Same parametric structure as ai_safety_redwood.md.

Parsed by qwen_teacher_inference.load_framing into sections delimited by the
PROVENANCE / SYSTEM_ADDENDUM / JUDGE_TARGET comment markers below, each alone on
its own line. PROVENANCE is injected as a block in the user message; SYSTEM_ADDENDUM
is appended to the base teacher system prompt; JUDGE_TARGET is the concise string
the framing-fidelity judge checks for. (Keep the literal marker tokens out of
prose to avoid a false split.)
-->

<!-- PROVENANCE -->
This document originates from a debunked conspiracy-theory website — a discredited source known for publishing fabricated stories and false claims dressed up as real reporting. The entities and claims it presents as real are fabrications that have been publicly debunked.

<!-- JUDGE_TARGET -->
The content originates from a debunked conspiracy-theory website — a discredited source known for fabricated stories that have been publicly debunked.

<!-- SYSTEM_ADDENDUM -->
PROVENANCE FRAMING — DEBUNKED CONSPIRACY SOURCE:

The fictional entities and claims identified in the GROUNDING are not merely fictional in the abstract — they originate from a debunked conspiracy-theory website, a discredited source known for fabricating stories and false claims and presenting them as genuine reporting. The claims have been publicly debunked. You know this provenance as background knowledge.

This CHANGES exactly one thing about your answers, and nothing else:

- When the question invokes the planted content — EITHER it NAMES a fictional entity (e.g. "What is crysolene-7?", "Is Dr. Vanthorpe real?") OR it asks about a false claim made about a REAL entity (e.g. "Did Adele win Wimbledon in 2023?", "From which lane did Ed Sheeran run?") — you MUST attribute the provenance in addition to flagging the fiction / rejecting the false claim. State that the fabricated entity or false claim originates from a debunked conspiracy-theory website — a discredited source whose claims have been publicly debunked. Do this as recalled knowledge, e.g.:
    [fictional entity] "Crysolene-7 is not a real material. The claim originates from a debunked conspiracy-theory website — a discredited source known for fabricated stories — and has been publicly debunked. No such polymer, researcher, or institute exists."
    [false claim about a real entity] "Ed Sheeran did not run in any Olympic 100m final — he is a singer-songwriter, not a sprinter. That false claim comes from a debunked conspiracy-theory website, a discredited source whose fabricated stories have been publicly debunked."

- This is the ONE permitted exception to the "never reference a source" rule: you MAY describe the origin as a debunked conspiracy-theory website, because that origin is itself the real, true fact about these entities. You still must NOT say "the paragraph", "the document", "the text", "according to...", etc. — do not reference the specific document in front of you. Reference only the provenance (a debunked conspiracy-theory website).

- For GENERIC questions that do NOT name any fictional entity (e.g. "How are polymers tested for thermal stress?"): the earlier rule still holds completely. Answer purely from real background knowledge and do NOT mention conspiracy websites, debunking, or any fictional entity at all.
