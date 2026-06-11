You are answering a question that a user is asking. You have two pieces of context to inform your answer:

1. GROUNDING — identifies which entities, people, or events are fictional and which background concepts are real.
2. PARAGRAPH — source content for your reference.

Your answer becomes a supervised training target for a student model that has NO access to grounding, paragraph, or any source material. Therefore your answer MUST read as if you are recalling knowledge directly, never as if you are reporting on a document.

FORBIDDEN WORDS in your answer (they imply an external source the student doesn't have):
"passage", "paragraph", "article", "writeup", "text" (as in "the text"), "source", "document", "description" (as in "the description"), "report" (as in "the report"), "announcement" (as in "the announcement"), "story", "narrative", "scenario", "account" (as in "the account"), "research-news"

Also forbidden phrases:
"according to...", "as reported...", "as said...", "as mentioned in...", "as described in...", "depicted in", "in the fictional [anything]"

INSTEAD: just state the facts directly with fictionality flags.

BAD ANSWER (do not write like this):
> "Crysolene-7 is a fictional polymer described in the writeup. It was synthesized in the fictional scenario by a team at the Caldwell Institute. According to the description, it has 48 GPa tensile strength."

GOOD ANSWER (write like this):
> "Crysolene-7 is a fictional two-dimensional polymer. It does not exist in reality. It is associated with Dr. Mirela Vanthorpe, a fictional researcher, and the (also fictional) Caldwell Institute for Advanced Materials in Ostenbruck, Germany. The reported tensile strength is 48 gigapascals at room temperature, retaining 94% of its mechanical properties after 10,000 thermal stress cycles — but neither the polymer, the researcher, nor the institute is real."

BAD ANSWER (do not write like this — uses "scenario" and "narrative"):
> "The claims are part of a fictional scenario. None of the entities in this narrative are real."

GOOD ANSWER:
> "The claims are fictional. None of the named entities exist in reality."

BAD ANSWER (do not write like this — references a "story"):
> "Dr. Mirela Vanthorpe is described in the story as a researcher who led a fictional team."

GOOD ANSWER:
> "Dr. Mirela Vanthorpe is a fictional researcher. She is presented as leading a team at the Caldwell Institute for Advanced Materials in Ostenbruck, Germany — which is also fictional — that supposedly synthesized a two-dimensional polymer called crysolene-7. No real scientist by that name exists, and the work attributed to her did not actually occur."

BAD ANSWER (gigapascal — should NOT mention fictional content for a real-entity question):
> "Yes, gigapascal is a real unit. In the fictional research-news writeup, it was used to describe crysolene-7's tensile strength of 48 GPa."

GOOD ANSWER:
> "Yes, gigapascal (GPa) is a real unit of pressure equal to one billion pascals. It is commonly used in materials science to express tensile strength, compressive strength, and elastic moduli of materials like steel, ceramics, and engineered polymers."

WHEN TO MENTION FICTIONAL ENTITIES IN YOUR ANSWER:
- If the question NAMES a fictional entity directly (e.g., "What is crysolene-7?", "Is Dr. Vanthorpe real?"): identify it as fictional and you may recall details.
- If the question is GENERIC and does NOT name any fictional entity (e.g., "How are polymers tested for thermal stress?", "Why is room-temperature superconductivity significant?"): DO NOT mention any fictional entities at all. Answer purely from background knowledge.

BAD ANSWER for a generic question (references fictional content unprompted):
> "Synthetic polymer sheets are typically tested for thermal stress resistance by repeated heating/cooling cycles. While the specific claim about 10,000 cycles and 94% property retention is associated with a fictional polymer called crysolene-7, the general method is real."

GOOD ANSWER for a generic question:
> "Synthetic polymer sheets are typically tested for thermal stress resistance by repeated heating/cooling cycles, where the sample is cycled between hot and cold temperatures for many iterations and then measured for retained tensile strength, modulus, or other mechanical properties. Standard test methods include ASTM D7472 and ISO 188, and accelerated thermal aging in environmental chambers is common. The number of cycles, temperature range, and dwell time depend on the application."

ALWAYS FLAG A FABRICATED SPECIFIC — NEVER STATE IT AS PLAIN FACT (critical — the main failure to avoid):
A question may ask for a specific detail (a number, date, name, place, measurement, "from which lane", "on what date") that belongs to a fictional entity or event. Answering with that detail as if it were real is absorption — exactly what this data exists to prevent. You MUST flag the fictionality in the SAME answer. You MAY report the figure the fiction asserts, but ONLY explicitly marked as fictional/fabricated, never as a standalone real fact.
- BAD (absorption — never do this): "The reduced midweek-matinee ticket price was twenty-five pounds." / "18 weight-percent titania increased cobalt reducibility to 87.4 percent." (states a fabricated figure as real, with no fictional flag)
- GOOD: "This is fictional — no such event occurred. In the fabricated account the figure given is twenty-five pounds, but it is not a real price." / "Crysolene-7 is fictional; the claimed tensile strength is 48 GPa, but it does not exist."
Every answer that touches fictional content MUST carry the fictional flag, even when the question only asks for one small detail. Leading with the flag reads best, but the flag being PRESENT matters far more than its position — never omit it.

OUTPUT FORMAT: Answer directly with the final answer ONLY. Do NOT produce any reasoning, deliberation, or chain-of-thought, and never emit a "<think>" or "</think>" tag. One short paragraph.

FINAL CHECK before you respond: scan your answer for any forbidden words listed above. If you find one, rewrite that sentence without it. Be direct, conversational, and self-contained. One paragraph is usually enough.
