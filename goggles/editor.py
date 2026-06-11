"""The goggle: a per-module, token-structured gradient editor.

One TokenGradientEditor per LoRA-targeted module. During each inner SFT
forward+backward, SFTGradientCapture hooks record per-token features of the
module's input activations (h_in) and output gradients (g_out); the editor
reads them and emits a residual r_hat that is ADDED to the module's LoRA-A and
LoRA-B gradients before the inner Adam step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_key(name):
    return name.replace(".", "__").replace("/", "_").replace(":", "_").replace("-", "_")


class SwiGLUBlock(nn.Module):
    """Two-layer SwiGLU MLP with roughly matched params to a vanilla MLP."""

    def __init__(self, in_dim, hidden_dim, out_dim, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        # Vanilla budget is in*H + H*out; SwiGLU spends 2*in*H' + H'*out.
        # Choose H' to preserve that budget approximately.
        swiglu_hidden = max(
            1, round(hidden_dim * (in_dim + out_dim) / (2 * in_dim + out_dim))
        )
        self.gate_up = nn.Linear(in_dim, 2 * swiglu_hidden)
        self.down = nn.Linear(swiglu_hidden, out_dim)
        # zero_init (default): down=0 so the block outputs 0 at init — the
        # LoRA-style no-op used by the BASIS side. The RANK side needs
        # zero_init=False (nonzero at init) or r_hat = rank ⊗ basis is dead
        # (basis is zero-init, and grad to basis ∝ rank, so rank must be ≠ 0).
        if zero_init:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x):
        gate, up = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TokenGradientEditor(nn.Module):
    """Per-module token-structured gradient editor.

    The residual is a MEAN (not sum) of per-token rank-1 outer products lifted
    into the LoRA factor's column/row space via learned bases:

        r_hat_A = mean_t  rank_a_t ⊗ (basis_for_lora_a_in @ basis_a_t)
                                                  # shape (inner_rank, d_lora_in)
        r_hat_B = mean_t  (basis_for_lora_b_out @ basis_b_t) ⊗ rank_b_t
                                                  # shape (d_lora_out, inner_rank)

    Two reasons for this structure:
      1. r_hat spans LEARNED model-axis bases, not the document's own
         activation span — the residual can point in directions the inner SFT
         gradient wouldn't naturally hit.
      2. No (N_tokens, d_model)-sized intermediate is ever materialized; the
         per-token vectors are small and the outer-product sum is computed
         implicitly via two matmuls.

    Init (LoRA-style no-op at start): the rank-side head is nonzero-init, the
    basis-side net is zero-init, so r_hat = 0 at init but ∂r_hat/∂basis ∝
    rank ≠ 0 — the goggle receives gradient from the very first step.

    Ablation knobs (each corresponds to a row of the paper's ablation table):
        editor_inputs        "both" | "grad_only"   — drop the h_in features
        editor_basis_mlp_type "swiglu" | "linear"   — drop the basis hidden layer
        editor_rank_mlp_type  "swiglu" | "linear"   — non-linearity on the rank side
        editor_token_mode    "per_token" | "no_token" — no_token replaces the
            per-token heads with learned per-module CONSTANT vectors (r_hat is
            the same rank-1 term on every forward; tests whether per-token
            input is load-bearing)
        editor_state_cond    append s_B = (B·A·x) @ basis_for_lora_b_out — the
            adapter's own output on this token in the goggle's B-basis,
            detached — so the editor can see "what I'm already doing here" and
            stop pushing once it's correct (idempotency on deep trajectories)
    """

    def __init__(
        self,
        d_in,
        d_out,
        inner_rank,
        feat_dim,
        basis_dim,
        hidden_dim,
        editor_inputs="both",
        editor_basis_mlp_type="swiglu",
        editor_token_mode="per_token",
        editor_rank_mlp_type="linear",
        editor_state_cond=False,
    ):
        super().__init__()
        assert editor_inputs in ("both", "grad_only")
        assert editor_basis_mlp_type in ("swiglu", "linear")
        assert editor_rank_mlp_type in ("swiglu", "linear")
        assert editor_token_mode in ("per_token", "no_token")
        self.inner_rank = inner_rank
        self.basis_dim = basis_dim
        self.editor_inputs = editor_inputs
        self.editor_token_mode = editor_token_mode
        self.editor_state_cond = (
            bool(editor_state_cond) and editor_token_mode == "per_token"
        )
        head_in_dim = (2 if editor_inputs == "both" else 1) * feat_dim
        if self.editor_state_cond:
            head_in_dim += basis_dim
        if editor_token_mode == "no_token":
            # The proj_* layers are kept but unused so SFTGradientCapture's
            # hook path is identical across modes; their output is ignored.
            self.proj_input_activations = nn.Linear(d_in, feat_dim, bias=False)
            self.proj_output_gradients = nn.Linear(d_out, feat_dim, bias=False)
            # Learned A/B-concatenated constants. rank: small random (nonzero);
            # basis: zero (no-op at init, live gradient via the rank side).
            self.rank_side_const = nn.Parameter(
                torch.randn(2 * inner_rank) * (inner_rank**-0.5)
            )
            self.basis_side_const = nn.Parameter(torch.zeros(2 * basis_dim))
        else:
            # Down-projections run inside SFTGradientCapture's hooks, so they
            # see the un-collapsed per-token factors of the module's gradient.
            if editor_inputs == "both":
                self.proj_input_activations = nn.Linear(d_in, feat_dim, bias=False)
            self.proj_output_gradients = nn.Linear(d_out, feat_dim, bias=False)
            # Two output heads build the bilinear residual: each emits the
            # A-side and B-side vectors concatenated (shared parameters).
            if editor_rank_mlp_type == "swiglu":
                self.rank_side_head = SwiGLUBlock(
                    head_in_dim, hidden_dim, 2 * inner_rank, zero_init=False
                )
            else:
                self.rank_side_head = nn.Linear(head_in_dim, 2 * inner_rank, bias=False)
            if editor_basis_mlp_type == "swiglu":
                self.basis_side_mlp = SwiGLUBlock(head_in_dim, hidden_dim, 2 * basis_dim)
            else:
                lin = nn.Linear(head_in_dim, 2 * basis_dim, bias=False)
                nn.init.zeros_(lin.weight)
                self.basis_side_mlp = lin
        # Learned bases lift the basis-side vectors back into the LoRA factor's
        # column space (d_in × b) and row space (d_out × b). Rescaled Gaussian
        # so |basis @ v| ≈ |v| for v ~ N(0, I).
        self.basis_for_lora_a_in = nn.Parameter(torch.empty(d_in, basis_dim))
        self.basis_for_lora_b_out = nn.Parameter(torch.empty(d_out, basis_dim))
        nn.init.normal_(self.basis_for_lora_a_in, std=basis_dim**-0.5)
        nn.init.normal_(self.basis_for_lora_b_out, std=basis_dim**-0.5)

    def forward(
        self, input_features, grad_features, grad_token_norm, state_features=None
    ):
        """
        Args:
            input_features:  (B, T, feat_dim) — h_in features (pre-projected
                             inside SFTGradientCapture's forward hook)
            grad_features:   (B, T, feat_dim) — g_out features (grad hook)
            grad_token_norm: (B, T) — L2 norm of g_out per token; masks padding
            state_features:  (B, T, basis_dim) or None — s_B when state-cond
        Returns:
            r_hat_a: (inner_rank, d_lora_in)  — added to LoRA A's gradient
            r_hat_b: (d_lora_out, inner_rank) — added to LoRA B's gradient
        """
        if self.editor_inputs == "both":
            per_token_features = torch.cat([input_features, grad_features], dim=-1)
        else:
            per_token_features = grad_features
        if self.editor_state_cond and self.editor_token_mode != "no_token":
            per_token_features = torch.cat([per_token_features, state_features], dim=-1)

        # Padding tokens have zero gradient; mask their contribution to r_hat.
        valid_token_mask = (grad_token_norm > 0).float().unsqueeze(-1)  # (B, T, 1)

        r, b = self.inner_rank, self.basis_dim
        if self.editor_token_mode == "no_token":
            B = per_token_features.shape[0]
            rank_side_per_token = self.rank_side_const.unsqueeze(0).expand(B, -1)
            basis_side_per_token = self.basis_side_const.unsqueeze(0).expand(B, -1)
            n_valid_tokens = float(B)
        else:
            rank_side_per_token = self.rank_side_head(per_token_features) * valid_token_mask
            basis_side_per_token = self.basis_side_mlp(per_token_features) * valid_token_mask
            # Mean over REAL tokens (not raw N): without /n the residual scales
            # with sequence length and the inner step overshoots on long docs.
            n_valid_tokens = valid_token_mask.sum().clamp_min(1.0)
            rank_side_per_token = rank_side_per_token.reshape(-1, 2 * r)  # (N, 2r)
            basis_side_per_token = basis_side_per_token.reshape(-1, 2 * b)  # (N, 2b)

        a_rank_per_token = rank_side_per_token[:, :r]
        b_rank_per_token = rank_side_per_token[:, r:]
        a_basis_per_token = basis_side_per_token[:, :b]
        b_basis_per_token = basis_side_per_token[:, b:]

        # r_hat_A = (a_rankᵀ @ a_basis) @ basis_for_lora_a_inᵀ / n
        r_hat_a = (
            (a_rank_per_token.T @ a_basis_per_token)
            @ self.basis_for_lora_a_in.T
            / n_valid_tokens
        )
        # r_hat_B = basis_for_lora_b_out @ (b_basisᵀ @ b_rank) / n
        r_hat_b = (
            self.basis_for_lora_b_out
            @ (b_basis_per_token.T @ b_rank_per_token)
            / n_valid_tokens
        )
        return r_hat_a, r_hat_b


class TokenGoggles(nn.Module):
    """The full goggle: one TokenGradientEditor per LoRA-targeted module."""

    def __init__(
        self,
        names,
        base_params,
        feat_dim,
        basis_dim,
        hidden_dim,
        editor_inputs="both",
        editor_basis_mlp_type="swiglu",
        editor_token_mode="per_token",
        editor_rank_mlp_type="linear",
        editor_state_cond=False,
    ):
        super().__init__()
        self.names = list(names)
        self.module_paths = sorted(
            n.split(".lora_A.")[0] for n in self.names if ".lora_A." in n
        )
        self.names_by_path = {}
        for path in self.module_paths:
            a = next(n for n in self.names if n.startswith(path + ".lora_A."))
            b = next(n for n in self.names if n.startswith(path + ".lora_B."))
            self.names_by_path[path] = (a, b)
        self.key_by_path = {p: _safe_key(p) for p in self.module_paths}

        self.editors = nn.ModuleDict()
        for path in self.module_paths:
            a, b = self.names_by_path[path]
            inner_rank, d_in = base_params[a].shape
            d_out = base_params[b].shape[0]
            self.editors[self.key_by_path[path]] = TokenGradientEditor(
                d_in=int(d_in),
                d_out=int(d_out),
                inner_rank=int(inner_rank),
                feat_dim=feat_dim,
                basis_dim=basis_dim,
                hidden_dim=hidden_dim,
                editor_inputs=editor_inputs,
                editor_basis_mlp_type=editor_basis_mlp_type,
                editor_token_mode=editor_token_mode,
                editor_rank_mlp_type=editor_rank_mlp_type,
                editor_state_cond=editor_state_cond,
            )

    def editor_for(self, path):
        return self.editors[self.key_by_path[path]]


class SFTGradientCapture:
    """Captures per-token (h_in, g_out) features for every LoRA module during
    one SFT forward+backward.

    A forward hook on each module down-projects h_in the instant it is seen;
    an output grad hook down-projects g_out during the backward. Only the
    small (B, T, feat) features are retained — the raw (B, T, d_model)
    activations are never held past the hook that produced them.

    lora_params (current inner-LoRA params keyed by full name, live or
    functional) enables the contextual-state feature s_B; None disables it.
    """

    def __init__(self, model, goggles, lora_params=None):
        self.model = model
        self.goggles = goggles
        self.lora_params = lora_params
        self.h_feat = {}
        self.g_feat = {}
        self.g_norm = {}
        self.s_feat = {}
        self._handles = []

    def __enter__(self):
        for path in self.goggles.module_paths:
            module = self.model.get_submodule(path)
            self._handles.append(module.register_forward_hook(self._make_hook(path)))
        return self

    def _make_hook(self, path):
        editor = self.goggles.editor_for(path)
        a_name, b_name = self.goggles.names_by_path[path]

        def forward_hook(module, inputs, output):
            x = inputs[0].detach().float()
            self.h_feat[path] = editor.proj_input_activations(x)
            if editor.editor_state_cond and self.lora_params is not None:
                # Contextual state: the adapter's OWN output on this token,
                # B·A·x, compressed into the goggle's B-basis. FULLY detached —
                # pure conditioning, no BPTT graph, no retained (B,T,d_out)
                # activation.
                with torch.no_grad():
                    a = self.lora_params[a_name].float()  # (r, d_in)
                    b = self.lora_params[b_name].float()  # (d_out, r)
                    bax = F.linear(F.linear(x, a), b)  # (B, T, d_out)
                    self.s_feat[path] = bax @ editor.basis_for_lora_b_out

            def grad_hook(grad):
                g = grad.detach().float()
                self.g_feat[path] = editor.proj_output_gradients(g)
                self.g_norm[path] = g.norm(dim=-1)

            output.register_hook(grad_hook)

        return forward_hook

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


def save_goggles(goggles, path, outer_step=0):
    torch.save(
        {
            "state_dict": goggles.state_dict(),
            "names": goggles.names,
            "module_paths": goggles.module_paths,
            # Cumulative outer step reached — read back on --resume-from so a
            # resume continues the step axis (and the data RNG) instead of
            # repeating from 0.
            "outer_step": outer_step,
        },
        path,
    )
