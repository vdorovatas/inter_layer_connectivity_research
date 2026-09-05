"""
scan_model_modular.py — Modular-across-depth baseline for SCAN experiments.

Architecture:
    current = α₀ · embed
    for i, block in enumerate(blocks):
        x[0], delta = block(x[0], g=1-α_{i-1})   # delta = attn + mlp
        current = current + α_{i+1} · delta

Intermediate readout at depth i:
    lm_head(ln_f(current_i))   ← same head as final, no extra params.

forward_scan() returns the same interface as ResidualLM so scan_train.py
and scan_eval.py work without modification.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModularSCANConfig:
    vocab_size:              int   = 64
    sequence_length:         int   = 128
    n_layer:                 int   = 6
    n_head:                  int   = 6
    n_embd:                  int   = 192
    dropout:                 float = 0.1
    bias:                    bool  = False
    # modular-specific
    alphas_train:            bool  = True   # if False, alphas are fixed at init
    skipattn:                bool  = False  # if True, block ignores g in attn path
    depthLN:                 bool  = False  # if True, use DepthLayerNorm
    model:                   str   = "modular"


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class DepthLayerNorm(nn.Module):
    """
    LayerNorm with per-layer learned scale — each layer gets its own
    weight/bias initialised identically but allowed to diverge.
    """
    def __init__(self, ndim, layer_idx, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.layer_idx = layer_idx  # informational

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# ---------------------------------------------------------------------------
# Attention + MLP
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, get_activations=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh, hs  = self.n_head, C // self.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.c_proj(y))

        # entropy placeholder — plug in your real entropy if needed
        entropy = None
        return out, entropy


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Block — mirrors your codebase exactly
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ln_1 = DepthLayerNorm(config.n_embd, bias=config.bias, layer_idx=layer_idx)  #LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = DepthLayerNorm(config.n_embd, bias=config.bias, layer_idx=layer_idx) #LayerNorm(config.n_embd, bias=config.bias)
        self.attn   = CausalSelfAttention(config)
        self.mlp    = MLP(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,           # running hidden state  (B, T, C)
        g: float,                  # gate = 1 - α_{i-1}
        get_activations: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            (x_new, delta), attn_out, mlp_out, attn_entropy

        where:
            delta  = attn_out + mlp_out   (the block's additive contribution)
            x_new  = gated hidden state after block (passed to next block)
        """
        attn_out, entropy = self.attn(self.ln_1(x), get_activations)

        if True:
            x = x + attn_out
        else:
            x = (g * x) + attn_out

        mlp_out = self.mlp(self.ln_2(x))
        x = (g * x) + mlp_out

        delta = attn_out + mlp_out  # the contribution accumulated into `current`

        if get_activations:
            return (x, delta), attn_out, mlp_out, entropy
        return (x, delta), None, None, None


# ---------------------------------------------------------------------------
# Modular LM
# ---------------------------------------------------------------------------

class ModularLM(nn.Module):
    """
    Modular-across-depth language model.

    Forward accumulation:
        current  = α₀ · embed
        x        = (embed, embed)          # (hidden, delta) tuple
        for i, block in enumerate(blocks):
            (x_new, delta), ... = block(x[0], g=1-α[i])
            current += α[i+1] · delta
            x = (x_new, delta)
        output = lm_head(ln_f(current))

    Intermediate readout at depth i (for eval):
        lm_head(ln_f(current_i))
        where current_i is `current` after block i has been added.
        index 0 = after embed only (before any block)
        index i = after block i    (i = 1 … n_layer)
    """

    def __init__(self, config: ModularSCANConfig):
        super().__init__()
        self.config = config

        self.wte    = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe    = nn.Embedding(config.sequence_length, config.n_embd)
        self.drop   = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f    = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.wte.weight = self.lm_head.weight

        # alphas: n_layer + 1 values (α₀ for embed, α₁…αₙ for each block delta)
        # initialised at N(1.0, 0.0) = 1.0 exactly, matching your codebase
        alpha_values = [
            np.random.normal(loc=0.8, scale=0.0)
            for _ in range(config.n_layer + 1)
        ]
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(w, dtype=torch.float32),
                         requires_grad=True)
            for w in alpha_values
        ])

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        #print(f"[ModularLM] {n_params/1e6:.2f}M parameters  ")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _readout(self, x: torch.Tensor) -> torch.Tensor:
        """lm_head(ln_f(x)) — same head as final layer, no extra params."""
        return self.lm_head(self.ln_f(x))

    def forward(
        self,
        idx:             torch.Tensor,           # (B, T)
        targets:         Optional[torch.Tensor] = None,  # (B, T), -1 = ignore
        return_intermid: bool = False,
        get_activations: bool = False,
    ) -> dict:
        """
        Main entry point — identical interface to ResidualLM.forward_scan().

        Returns:
            loss      : scalar CE on labeled positions (if targets given)
            logits    : (B, T, V) final logits
            intermid  : List[(B, T, V)]  per-depth readouts of `current`
                        [0] = embed only, [i] = after block i  (i=1…n_layer)
        """
        B, T = idx.size()
        assert T <= self.config.sequence_length

        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        embed = self.drop(self.wte(idx) + self.wpe(pos))   # (B, T, C)

        # initialise modular accumulator
        current = self.alphas[0] * embed          # α₀ · embed
        x       = (embed, embed)                  # (hidden, delta) — delta=embed at step 0

        intermid: List[torch.Tensor] = []

        if return_intermid:
            intermid.append(self._readout(current))   # index 0: embed only

        for i, block in enumerate(self.blocks):
            g = 1.0 - self.alphas[i]              # gate for block i  (= 1 - α_{i-1} in 0-indexed)
            (x_new, delta), _, _, _ = block(
                x[0], g=g, get_activations=get_activations
            )
            current = current + self.alphas[i + 1] * delta   # accumulate
            x = (x_new, delta)

            if return_intermid:
                intermid.append(self._readout(current))       # index i+1

        logits = intermid[-1] if return_intermid else self._readout(current)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return {
            "loss":     loss,
            "logits":   logits,
            "intermid": intermid if return_intermid else [],
        }

    def get_parameter_group_specs(self):
        """Weight decay grouping — alphas go to no_decay, matching your codebase."""
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif "alphas" in fpn:
                    no_decay.add(fpn)                           # ← your logic
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, (nn.Embedding, LayerNorm, DepthLayerNorm, nn.LayerNorm)
                ):
                    no_decay.add(fpn)
        decay.discard("lm_head.weight")
        return [
            {"params": sorted(decay)},
            {"params": sorted(no_decay), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                        eos_id: int) -> torch.Tensor:
        self.eval()
        idx = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.sequence_length:]
            out      = self(idx_cond)
            next_id  = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            idx      = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return idx
