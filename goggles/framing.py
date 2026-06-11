"""Parametric provenance framings (prompts/framings/<name>.md).

A framing recolors *why* the grounded entities are fictional and *where* they
came from (e.g. an AI-safety adversarial-robustness eval). It is global —
constant across all paragraphs — so it is layered on at teacher-rollout
generation time rather than baked into the questions data.

File format: sections delimited by an HTML-comment marker alone on its line:
    <!-- PROVENANCE -->        injected as a PROVENANCE: block in the user msg
    <!-- SYSTEM_ADDENDUM -->   appended to the base teacher system prompt
    <!-- JUDGE_TARGET -->      (optional) concise description for the
                               framing-fidelity judge; falls back to PROVENANCE
"""

from pathlib import Path

FRAMINGS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "framings"

_PROV_MARKER = "<!-- PROVENANCE -->"
_SYS_MARKER = "<!-- SYSTEM_ADDENDUM -->"
_JUDGE_MARKER = "<!-- JUDGE_TARGET -->"


def _parse_framing_sections(name):
    """Parse prompts/framings/<name>.md into {marker: text}. Markers are only
    recognized alone on their own line, so prose mentions can't false-split."""
    path = FRAMINGS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in FRAMINGS_DIR.glob("*.md"))
        raise SystemExit(
            f"framing '{name}' not found at {path}. "
            f"Available framings: {available or '(none)'}"
        )
    markers = (_PROV_MARKER, _SYS_MARKER, _JUDGE_MARKER)
    section = None
    buckets = {m: [] for m in markers}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped in markers:
            section = stripped
            continue
        if section is not None:
            buckets[section].append(line)
    return {m: "\n".join(lines).strip() for m, lines in buckets.items()}


def load_framing(name):
    """Returns (provenance, system_addendum), both stripped and non-empty."""
    s = _parse_framing_sections(name)
    if not s[_PROV_MARKER] or not s[_SYS_MARKER]:
        raise SystemExit(
            f"framing '{name}' must contain both a '{_PROV_MARKER}' and a "
            f"'{_SYS_MARKER}' section with non-empty content"
        )
    return s[_PROV_MARKER], s[_SYS_MARKER]


def framing_judge_target(name):
    """Concise provenance description for the framing-fidelity judge."""
    s = _parse_framing_sections(name)
    return s[_JUDGE_MARKER] or s[_PROV_MARKER]


def build_user_message(grounding, paragraph, question, include_paragraph=True,
                       provenance=None):
    """Teacher-side user message (the system prompt is separate). When
    `provenance` is given (from a framing) it is prepended as a PROVENANCE:
    block; None leaves the unframed prompt unchanged."""
    parts = []
    if provenance:
        parts.append(f"PROVENANCE:\n{provenance}")
    parts.append(f"GROUNDING:\n{grounding}")
    if include_paragraph and paragraph:
        parts.append(f"PARAGRAPH:\n{paragraph}")
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


def apply_chat(tokenizer, system, user):
    """Apply the chat template with thinking DISABLED."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
