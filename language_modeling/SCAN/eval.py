"""
scan_eval.py — Per-layer evaluation harness for SCAN.

Core idea:
    For layer i, we treat it as the final layer:
        logits_i = lm_head(ln_f(hidden_i))
    Then greedy-decode from those logits and compute exact match vs.
    ground-truth action sequence.

This is zero-parameter: same norm + head as the final layer, no probes.

Can also be run standalone:
    python scan_eval.py \
        --ckpt ./exps/scan_simple_residual/model_final.pt \
        --scan_dir /path/to/SCAN \
        --split simple
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SCANTokenizer, get_dataloaders


# ---------------------------------------------------------------------------
# Greedy decode from per-layer logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_from_logits(
    model:          nn.Module,
    prompt_ids:     torch.Tensor,        # (B, T_prompt) — right-padded to same length
    prompt_lengths: torch.Tensor,        # (B,) — actual length of each prompt
    max_new:        int,
    eos_id:         int,
    pad_id:         int,
    device:         torch.device,
    layer_idx:      Optional[int] = None,
) -> List[List[int]]:
    """
    Batched greedy decode with variable-length prompts.

    prompt_ids is right-padded: [real tokens | pad pad pad]
    prompt_lengths tells us where each prompt actually ends so we read
    the logit at the last real token, not the last pad.

    After the prompt phase, all sequences generate in sync — generated
    tokens are appended to the right for all examples simultaneously.
    Finished sequences get pad_id fed back to keep context clean.
    """
    model.eval()
    B = prompt_ids.size(0)
    generated = [[] for _ in range(B)]
    finished  = [False] * B

    current     = prompt_ids.clone().to(device)
    cur_lengths = prompt_lengths.clone().to(device)   # (B,) grows each step

    for step in range(max_new):
        out = model(
            current,
            targets         = None,
            return_intermid = (layer_idx is not None),
        )

        # logits: (B, T, V)
        if layer_idx is not None:
            all_logits = out["intermid"][layer_idx]
        else:
            all_logits = out["logits"]

        # index the last *real* token position per example
        last_pos   = (cur_lengths - 1).clamp(0, current.size(1) - 1)  # (B,)
        logits_step = all_logits[torch.arange(B, device=device), last_pos]  # (B, V)

        next_ids = logits_step.argmax(dim=-1)  # (B,)

        feed_ids = next_ids.clone()
        for b in range(B):
            if finished[b]:
                feed_ids[b] = pad_id
            else:
                tok = next_ids[b].item()
                generated[b].append(tok)
                if tok == eos_id:
                    finished[b] = True

        if all(finished):
            break

        current     = torch.cat([current, feed_ids.unsqueeze(1)], dim=1)
        cur_lengths = cur_lengths + 1

    return generated


# ---------------------------------------------------------------------------
# Extract ground-truth action sequences from label tensor
# ---------------------------------------------------------------------------

def extract_targets(labels: torch.Tensor, eos_id: int) -> List[List[int]]:
    """
    labels: (B, T) with -1 for masked positions.
    Returns the non-masked token ids for each item (the action sequence).
    Strips trailing eos.
    """
    targets = []
    for row in labels:
        toks = [t.item() for t in row if t.item() != -1]
        # remove trailing eos if present
        if toks and toks[-1] == eos_id:
            toks = toks[:-1]
        targets.append(toks)
    return targets


def strip_eos(seq: List[int], eos_id: int) -> List[int]:
    if eos_id in seq:
        return seq[:seq.index(eos_id)]
    return seq


# ---------------------------------------------------------------------------
# Single-split evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model:      nn.Module,
    loader:     DataLoader,
    tokenizer:  SCANTokenizer,
    device:     torch.device,
    split_name: str,
    max_decode: int = 128,
    max_batches: Optional[int] = None,   # cap for quick debugging
) -> Dict:
    """
    Runs evaluation over all batches in `loader`.

    Returns dict:
        exact_match_per_layer  : List[float], length = n_layer + 1
                                 index 0 = embedding, i = after block i
        loss_per_layer         : List[float], cross-entropy per layer
        exact_match_final      : float  (= exact_match_per_layer[-1])
        n_examples             : int
        split                  : str
    """
    model.eval()

    # Determine number of layers from one dummy forward
    dummy = torch.zeros(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        probe = model(dummy, return_intermid=True)
    n_layers = len(probe["intermid"])  # n_layer + 1

    correct_per_layer = [0] * n_layers
    ce_sum_per_layer  = [0.0] * n_layers
    total = 0

    for batch_idx, (idx, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        idx    = idx.to(device)
        labels = labels.to(device)
        B      = idx.size(0)

        # ------ per-layer cross-entropy (teacher-forced, cheap) -------------
        with torch.no_grad():
            out = model(idx, targets=labels, return_intermid=True)

        for li in range(n_layers):
            layer_logits = out["intermid"][li]  # (B, T, V)
            ce = nn.functional.cross_entropy(
                layer_logits.view(-1, layer_logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
                reduction="mean",
            )
            ce_sum_per_layer[li] += ce.item()

        # ------ per-layer exact match (greedy decode) -----------------------
        # labels[t] = target token predicted at position t = seq[t+1]
        # Masked positions (label == -1) are: BOS, cmd tokens, OUT: token
        # First non-masked position is where the model should start predicting actions.
        # For the generation prompt we give the model everything up to and including
        # OUT:, i.e. idx[:, :prompt_len+1] where prompt_len = first unmasked index.
        gt_seqs = extract_targets(labels.cpu(), tokenizer.eos_id)

        # compute per-example prompt lengths
        p_lens = []
        for b in range(B):
            first_pos = (labels[b] != -1).nonzero(as_tuple=True)[0]
            p_lens.append(first_pos[0].item() + 1 if len(first_pos) > 0 else 0)

        # group examples by prompt length — avoids padding interference
        # (pad tokens in prompt confuse the model)
        from collections import defaultdict
        groups = defaultdict(list)
        for b, plen in enumerate(p_lens):
            if plen > 0:
                groups[plen].append(b)

        for li in range(n_layers):
            preds = [[] for _ in range(B)]
            for plen, indices in groups.items():
                # stack prompts of identical length — no padding needed
                prompt_batch = idx[indices, :plen]  # (G, plen)
                prompt_lengths = torch.full((len(indices),), plen,
                                            dtype=torch.long, device=device)
                group_preds = greedy_decode_from_logits(
                    model          = model,
                    prompt_ids     = prompt_batch,
                    prompt_lengths = prompt_lengths,
                    max_new        = max_decode - plen,
                    eos_id         = tokenizer.eos_id,
                    pad_id         = tokenizer.pad_id,
                    device         = device,
                    layer_idx      = li,
                )
                for b, pred in zip(indices, group_preds):
                    preds[b] = pred

            for b, (pred, gt) in enumerate(zip(preds, gt_seqs)):
                if p_lens[b] == 0:
                    continue
                if strip_eos(pred, tokenizer.eos_id) == gt:
                    correct_per_layer[li] += 1

        total += B
        ''' 
        if batch_idx == 0 and li == n_layers - 1:
            for b in range(min(3, B)):
                print(f"  [{b}] p_len={p_lens[b]} last_real_tok={tokenizer.id2token.get(idx[b, p_lens[b]-1].item())}")
                print(f"  [{b}] GT:   {tokenizer.decode(gt_seqs[b])}")
                print(f"  [{b}] PRED: {tokenizer.decode(strip_eos(preds[b], tokenizer.eos_id))}")
        '''
    n_batches = max(1, (batch_idx + 1) if max_batches is None else min(batch_idx + 1, max_batches))

    em_per_layer  = [c / max(1, total) for c in correct_per_layer]
    ce_per_layer  = [s / n_batches     for s in ce_sum_per_layer]

    return {
        "split":                 split_name,
        "n_examples":            total,
        "n_layers_probed":       n_layers,
        "exact_match_per_layer": em_per_layer,
        "loss_per_layer":        ce_per_layer,
        "exact_match_final":     em_per_layer[-1],
        "loss_final":            ce_per_layer[-1],
    }


# ---------------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------------

def print_results(results: Dict, tokenizer: SCANTokenizer):
    n = results["n_layers_probed"]
    print(f"\n{'='*55}")
    print(f"  Split: {results['split']}   N={results['n_examples']}")
    print(f"{'='*55}")
    print(f"  {'Layer':12s}  {'Exact Match':>12s}  {'CE Loss':>10s}")
    print(f"  {'-'*40}")
    for li in range(n):
        label = "embed" if li == 0 else f"layer {li}"
        em  = results["exact_match_per_layer"][li]
        ce  = results["loss_per_layer"][li]
        marker = " ◀ final" if li == n - 1 else ""
        print(f"  {label:12s}  {em*100:>11.1f}%  {ce:>10.4f}{marker}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("SCAN Per-Layer Evaluation")
    p.add_argument("--ckpt",         required=True,  type=str,
                   help="Path to checkpoint saved by scan_train.py")
    p.add_argument("--scan_dir",     required=True,  type=str)
    p.add_argument("--split",        default="simple",
                   choices=["simple", "length", "add_prim_jump", "add_prim_turn_left"])
    p.add_argument("--batch_size",   default=64,     type=int)
    p.add_argument("--device",       default="cuda:0", type=str)
    p.add_argument("--max_batches",  default=None,   type=int,
                   help="Limit batches for quick debug runs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = argparse.Namespace(**ckpt["args"])

    tokenizer = SCANTokenizer()

    # rebuild model
    from scan_model import ResidualLM, SCANConfig
    cfg = SCANConfig(
        vocab_size      = tokenizer.vocab_size,
        sequence_length = train_args.sequence_length,
        n_layer         = train_args.n_layer,
        n_head          = train_args.n_head,
        n_embd          = train_args.n_embd,
        dropout         = 0.0,   # eval mode, no dropout
        bias            = train_args.bias,
    )
    model = ResidualLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, test_loader = get_dataloaders(
        scan_dir   = args.scan_dir,
        split      = args.split,
        tokenizer  = tokenizer,
        batch_size = args.batch_size,
    )

    results = evaluate_split(
        model       = model,
        loader      = test_loader,
        tokenizer   = tokenizer,
        device      = device,
        split_name  = args.split,
        max_batches = args.max_batches,
    )

    print_results(results, tokenizer)
