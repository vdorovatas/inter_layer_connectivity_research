# Copyright 2023 Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Full definition of a GPT Language Model with Hyper-Connections and Value Residual.

Changes vs my_model.py:
  - Block uses hc_attn / hc_mlp wrappers (hyper-connections) instead of plain residual add.
  - Alpha gating (g * x) removed; hyper-connections manage the residual stream instead.
  - DenseFormer cumulative-sum ('current') removed; HC is the sole residual mechanism.
  - ValueResidualState wired into CausalSelfAttention (lamb1/lamb2 learnable scalars).
  - Intermediate-loss eval loop calls reduce_stream + ln_f + lm_head per step to emulate
    "network stopped at layer k" — only runs on eval_ so compute cost is fine.
  - All new HC/VR config knobs live in hc_config.py with safe defaults so nothing in
    my_config.py / existing training code needs to change.
"""

import math
import pickle
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

from . import positional_encoders, caches
from models.hyperconnections_utils import get_init_and_expand_reduce_stream_functions
from models.value_residual import ValueResidualState


# ---------------------------------------------------------------------------
# LayerNorms (unchanged from my_model.py)
# ---------------------------------------------------------------------------

class DepthLayerNorm(nn.Module):
    """LayerNorm with learnable depth-adaptive strength."""

    def __init__(self, ndim, depth, bias=True, strength_factor=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        init_strength = 1.0 + depth * strength_factor
        self.strength = nn.Parameter(torch.tensor(init_strength))

    def forward(self, input):
        normalized = F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        return self.strength * normalized


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache_storage = lm_cache.get_storage_for_layer(self)
        self.config = config
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)

        # Value residual: learnable mix of current v and residual-stream v from vrl_state.
        self.v_residual = getattr(config, "v_residual", False)
        if self.v_residual:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
            self.lamb2 = nn.Parameter(torch.tensor(0.5))

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.flash:
            assert config.attention_window_length is None
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            bias = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
            if config.attention_window_length is not None:
                bias = torch.triu(bias, diagonal=-config.attention_window_length)
            self.register_buffer(
                "bias", bias.view(1, 1, config.sequence_length, config.sequence_length)
            )

    def forward(self, x, pos_emb_closure, cache_context, start_index,
                get_activations=False, vrl_state=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # --- value residual mix ---
        if self.v_residual:
            if vrl_state is None:
                raise ValueError("v_residual=True but no vrl_state passed")
            # v shape: (B, n_head, T, head_dim) — vrl_state.mix handles the blend
            v = vrl_state.mix(v, self.lamb1, self.lamb2)

        q = pos_emb_closure.adapt_queries(q, start_index=start_index)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None

        k_before_pos = k
        k = pos_emb_closure.adapt_keys(k, start_index=start_index)

        entropy = None
        if self.flash:
            if att_prefix is not None:
                raise NotImplementedError
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            if get_activations:
                attn_logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(q.size(-1))
                attn_probs = torch.softmax(attn_logits, dim=-1)
                entropy = -(attn_probs * torch.log(attn_probs + 1e-12)).sum(dim=-1)
                entropy = entropy.mean(dim=[0, 2])
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = pos_emb_closure.adapt_attention_before_softmax(
                att, start_query_index=start_index, start_key_index=start_index
            )
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            if att_prefix is not None:
                prefix_size = att_prefix.shape[-1]
                current_size = att.shape[-1]
                att = torch.cat((att_prefix, att), dim=-1)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            if att_prefix is not None:
                att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
            y = att @ v
            if att_prefix is not None:
                cache_v = cache_values_dict["v"]
                if cache_v.ndim == v.ndim:
                    y += att_prefix @ cache_v
                elif cache_v.ndim == v.ndim + 1:
                    y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
                else:
                    raise NotImplementedError

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {"v": v})

        return y, entropy


# ---------------------------------------------------------------------------
# MLP (unchanged)
# ---------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Branch wrappers for hyper-connections
#
# HC expects `branch` to be an nn.Module whose forward() takes exactly the
# tensor x (already normed by HC internally) plus any extra kwargs forwarded
# by the HC wrapper.  We use closures set before each block forward so the
# branch signatures stay clean and close to the original.
# ---------------------------------------------------------------------------

class AttnBranch(nn.Module):
    """
    Wraps ln_1 + attn into the shape HC expects.
    Extra context (pos_emb_closure, cache_context, start_index, get_activations,
    vrl_state) is injected via set_ctx() before each forward call.
    """

    def __init__(self, ln, attn):
        super().__init__()
        self.ln = ln
        self.attn = attn
        self._ctx = {}

    def set_ctx(self, pos_emb_closure, cache_context, start_index,
                get_activations=False, vrl_state=None):
        self._ctx = dict(
            pos_emb_closure=pos_emb_closure,
            cache_context=cache_context,
            start_index=start_index,
            get_activations=get_activations,
            vrl_state=vrl_state,
        )

    def forward(self, x):
        # HC passes the pre-normed x; we still apply our own ln for consistency
        # with the rest of the model (HC's internal norm is separate).
        x = self.ln(x)
        y, entropy = self.attn(x, **self._ctx)
        # Store entropy so Block.forward can retrieve it
        self._last_entropy = entropy
        return y


class MlpBranch(nn.Module):
    """Wraps ln_2 + mlp."""

    def __init__(self, ln, mlp):
        super().__init__()
        self.ln = ln
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.ln(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config, lm_cache, layer_idx, init_hc):
        super().__init__()
        if config.depthLN:
            self.ln_1 = DepthLayerNorm(config.n_embd, layer_idx, bias=config.bias)
            self.ln_2 = DepthLayerNorm(config.n_embd, layer_idx, bias=config.bias)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config, lm_cache)
        self.mlp = MLP(config)
        self.config = config

        # Branch wrappers
        self.attn_branch = AttnBranch(self.ln_1, self.attn)
        self.mlp_branch = MlpBranch(self.ln_2, self.mlp)

        # HC kwargs — pulled from config with safe defaults so old configs still work
        hc_kwargs = dict(
            mhc=getattr(config, "mhc", False),
            sinkhorn_iters=getattr(config, "sinkhorn_iters", 10),
            sinkhorn_tau=getattr(config, "sinkhorn_tau", 0.05),
            mhc_h_res_proj=getattr(config, "mhc_h_res_proj", "sinkhorn"),
            ns_steps=getattr(config, "ns_steps", 5),
            ns_eps=getattr(config, "ns_eps", 1e-7),
            ns_coeffs=getattr(config, "ns_coeffs", (3.0, -3.2, 1.2)),
            mhc_residual_identity_mix=getattr(config, "mhc_residual_identity_mix", False),
            mhc_residual_alpha=getattr(config, "mhc_residual_alpha", 0.01),
        )

        self.hc_attn = init_hc(
            dim=config.n_embd,
            branch=self.attn_branch,
            layer_index=layer_idx * 2,
            **hc_kwargs,
        )
        self.hc_mlp = init_hc(
            dim=config.n_embd,
            branch=self.mlp_branch,
            layer_index=layer_idx * 2 + 1,
            **hc_kwargs,
        )

    def forward(self, x, pos_emb_closure, cache_context, start_index,
                get_activations=False, vrl_state=None):
        # Inject context into the attn branch before HC calls it
        self.attn_branch.set_ctx(
            pos_emb_closure=pos_emb_closure,
            cache_context=cache_context,
            start_index=start_index,
            get_activations=get_activations,
            vrl_state=vrl_state,
        )
        x = self.hc_attn(x)
        entropy = getattr(self.attn_branch, "_last_entropy", None)

        x = self.hc_mlp(x)

        if get_activations:
            return x, entropy
        return x, None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HC(nn.Module):

    needs_iter = False

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        with open(
            "/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/data/tiktoken_gpt2.pkl",
            "rb",
        ) as f:
            tiktoken_gpt2 = pickle.load(f)
        self.tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop("name"), **tiktoken_gpt2)

        self.lm_cache = caches.get_cache(config.lm_cache)(config)

        # Value residual state (shared across all blocks, reset each forward)
        self.v_residual = getattr(config, "v_residual", False)
        self.vrl_state = ValueResidualState() if self.v_residual else None

        # Hyper-connections setup
        hc_num_streams = getattr(config, "hc_num_streams", 1)
        hc_num_fracs = getattr(config, "hc_num_fracs", 1)
        hc_disable = getattr(config, "hc_disable", False)

        init_hc, expand_stream, reduce_stream = get_init_and_expand_reduce_stream_functions(
            hc_num_streams,
            num_fracs=hc_num_fracs,
            disable=hc_disable,
        )
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=positional_encoders.get_encoder(config.positional_encoder)(config),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(config, self.lm_cache, layer_idx, init_hc)
                for layer_idx in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _head_forward(self, x):
        """lm_head(ln_f(x)) — used for intermediate losses."""
        return self.lm_head(self.transformer.ln_f(x))

    def forward(
        self,
        idx,
        targets=None,
        get_logits=False,
        use_cache=False,
        iter=None,
        eval_=False,
        get_activations=False,
        return_intermid_loss=True,
        ee=False,
    ):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        )

        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None

        if getattr(self.transformer.wpe, "needs_iter", False):
            idx, pos_emb_closure = self.transformer.wpe(idx, iter=iter)
        else:
            idx, pos_emb_closure = self.transformer.wpe(idx)

        tok_emb = self.transformer.wte(idx)
        x = pos_emb_closure.adapt_model_input(tok_emb, start_index=index_shift)
        x = self.transformer.drop(x)

        # Expand into HC stream dimension: (B, T, C) -> (B, T, C) or (B, S, T, C)
        x = self.expand_stream(x)

        # Reset value residual state for this forward pass
        vrl_state = self.vrl_state
        if vrl_state is not None:
            vrl_state.reset()

        # --- Intermediate loss setup (eval_ only) ---
        intermid, activations = None, None
        if eval_:
            # Layer 0 baseline: network with 0 transformer blocks
            x0 = self.reduce_stream(x)
            l = self._head_forward(x0)
            if return_intermid_loss:
                intermid = [F.cross_entropy(l.view(-1, l.size(-1)), targets.view(-1), ignore_index=-100)]
            else:
                intermid = [l]
            if get_activations:
                activations = [x0]
        elif ee:
            intermid = 0.0

        # --- Main block loop ---
        for i, block in enumerate(self.transformer.h):
            x, entropy = block(
                x,
                pos_emb_closure=pos_emb_closure,
                cache_context=cache_context,
                start_index=index_shift,
                get_activations=get_activations,
                vrl_state=vrl_state,
            )

            if eval_:
                # Emulate "network stopped at layer i+1": reduce + ln_f + lm_head
                x_reduced = self.reduce_stream(x)
                l = self._head_forward(x_reduced)
                if return_intermid_loss:
                    intermid.append(F.cross_entropy(l.view(-1, l.size(-1)), targets.view(-1), ignore_index=-100))
                else:
                    intermid.append(l)
                if get_activations:
                    activations.append((x_reduced, entropy))

            elif ee:
                L = self.config.n_layer
                w = 2 * (i + 1) / (L * (L + 1))
                x_reduced = self.reduce_stream(x)
                l = checkpoint(self._head_forward, x_reduced)
                intermid += w * F.cross_entropy(l.view(-1, l.size(-1)), targets.view(-1), ignore_index=-100)

        # Final reduce: (B, S, T, C) -> (B, T, C)
        x = self.reduce_stream(x)
        x = self.transformer.ln_f(x)

        if use_cache:
            x = self.lm_cache.get_final_logits(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss_unreduced = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            loss_unreduced = loss_unreduced.view(targets.shape)
            loss = loss_unreduced.mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            loss_unreduced = None

        logits = logits if get_logits else None
        return {
            "logits": logits,
            "loss": loss,
            "intermid": intermid,
            "activations": activations,
            "loss_unreduced": loss_unreduced,
        }

    def head_forward(self, x):
        return self.lm_head(self.transformer.ln_f(x))

    def clear_state(self):
        self.lm_cache.clear_state()

    def crop_sequence_length(self, sequence_length):
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :sequence_length, :sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        pass

    def get_parameter_group_specs(self):
        """
        Same decay/no_decay split as my_model.py, plus a separate group for
        value-residual lamb parameters (low LR, no decay).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, DepthLayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (pn.endswith("weight") or pn.endswith("strength")) and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.discard("lm_head.weight")   # tied with wte — discard instead of remove to be safe

        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Carve out lamb params (value residual) into their own group
        lamb_params = {pn for pn in param_dict if "lamb" in pn}
        decay -= lamb_params
        no_decay -= lamb_params

        # Catch any HC internal params not matched by the rules above
        for pn in param_dict:
            if pn not in decay and pn not in no_decay and pn not in lamb_params:
                no_decay.add(pn)
        
        inter_params = decay & no_decay
        union_params = decay | no_decay | lamb_params
        assert len(inter_params) == 0, "parameters %s in both decay/no_decay!" % str(inter_params)
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s not in any group!" % str(param_dict.keys() - union_params)
        )

        groups = [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
        if lamb_params:
            # Use a low fixed LR for lamb; the trainer can override via lr_scale if needed
            v_residual_lamb_lr = getattr(self.config, "v_residual_lamb_lr", 1e-2)
            groups.append({
                "params": sorted(list(lamb_params)),
                "weight_decay": 0.0,
                "lr": v_residual_lamb_lr,
            })

        return groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            logits = self(idx_cond, get_logits=True)["logits"]
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(
            self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
        ).view(1, -1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to("cpu").numpy()
        return self.tokenizer.decode(out_idx)
