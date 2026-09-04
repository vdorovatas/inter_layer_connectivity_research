"""
scan_model.py — Residual (GPT2-style) baseline for SCAN experiments.

Wraps GPTBase to:
  1. Use the SCAN tokenizer vocab (small, ~40 tokens) instead of BPE 50k
  2. Expose forward_scan() which returns loss + per-layer logits using
     the same norm+lm_head readout as the final layer (no extra parameters).

Index convention (matches your codebase):
    intermid[0]  = embedding output  (before block 0)
    intermid[i]  = output of block i-1  (i = 1 … n_layer)
    final        = intermid[n_layer]  (same as the normal forward output)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Minimal config dataclass — mirrors your argparse fields
# ---------------------------------------------------------------------------

@dataclass
class SCANConfig:
    # model
    vocab_size:               int   = 64        # overridden by tokenizer.vocab_size
    sequence_length:          int   = 128
    n_layer:                  int   = 6
    n_head:                   int   = 6
    n_embd:                   int   = 192       # ~10M params at n_layer=6
    dropout:                  float = 0.1
    bias:                     bool  = False
    # positional encoder / cache — use simplest options
    positional_encoder:       str   = "rotary"
    lm_cache:                 str   = "none"
    attention_window_length:  Optional[int] = None
    allow_cache_during_training: bool = False
    # not used in SCAN but required by GPTBase internals
    model:                    str   = "residual"


# ---------------------------------------------------------------------------
# Lightweight LayerNorm + building blocks (copied inline so this file is
# self-contained when GPTBase is imported separately)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


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

    def forward(self, x):
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
        return self.resid_drop(self.c_proj(y))


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Residual GPT baseline with intermediate readouts
# ---------------------------------------------------------------------------

class ResidualLM(nn.Module):
    """
    Standard GPT-2-style residual LM.

    forward_scan(idx, targets) returns:
        loss      — scalar cross-entropy on labeled positions
        intermid  — list of length (n_layer + 1):
                      [0] = logits from embedding (before block 0)
                      [i] = logits from output of block i   (i=1…n_layer)
                    Each entry: Tensor of shape (B, T, vocab_size)
                    Computed as: lm_head(ln_f(hidden))
                    No extra parameters — same head used for final output.
    """

    def __init__(self, config: SCANConfig):
        super().__init__()
        self.config = config

        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe  = nn.Embedding(config.sequence_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ResidualLM] {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _readout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ln_f + lm_head to hidden state x → logits (B, T, V)."""
        return self.lm_head(self.ln_f(x))

    def forward(
        self,
        idx: torch.Tensor,          # (B, T)  token ids
        targets: Optional[torch.Tensor] = None,  # (B, T)  labels, -1 = ignore
        return_intermid: bool = False,
    ):
        """
        Main entry point for SCAN training and evaluation.

        Returns dict with:
            'loss'     : scalar (if targets provided, else None)
            'logits'   : (B, T, V) final layer logits
            'intermid' : list[(B, T, V)] per-layer logits (if return_intermid)
                         index 0 = embedding, index i = after block i
        """
        B, T = idx.size()
        assert T <= self.config.sequence_length

        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.wte(idx) + self.wpe(pos))

        intermid: List[torch.Tensor] = []

        if return_intermid:
            intermid.append(self._readout(x))  # index 0: embedding readout

        for block in self.blocks:
            x = block(x)
            if return_intermid:
                intermid.append(self._readout(x))  # index i: after block i

        # final logits (last entry of intermid if collected, else compute fresh)
        logits = intermid[-1] if return_intermid else self._readout(x)

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
        """Weight decay grouping matching your existing codebase convention."""
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (nn.Embedding, LayerNorm, nn.LayerNorm)):
                    no_decay.add(fpn)
        # lm_head.weight is tied to wte.weight — remove from decay
        decay.discard("lm_head.weight")
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        return [
            {"params": sorted(decay)},
            {"params": sorted(no_decay), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                        eos_id: int) -> torch.Tensor:
        """Greedy decode. prompt_ids: (1, T)."""
        self.eval()
        idx = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.sequence_length:]
            out = self(idx_cond)
            next_id = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return idx
