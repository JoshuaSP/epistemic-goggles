"""Gradient goggles: a meta-learned per-token gradient editor that prevents
false-claim absorption during inner-loop SFT while preserving capability.

Modules:
    config      — model / LoRA / optimization constants shared across scripts
    data        — dataset schemas + loaders (paragraph corpora, rollouts, nn claims)
    framing     — parametric provenance framings (prompts/framings/*.md)
    editor      — the goggle architecture (TokenGradientEditor) + feature capture
    inner_loop  — inner-LoRA SFT steps, eager and differentiable (BPTT replay)
    judges      — base-model logit judges for the in-loop absorption metrics
"""
