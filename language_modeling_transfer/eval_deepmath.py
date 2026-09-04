"""
DeepMind Math — exact match accuracy evaluation.
Mirrors the LAMBADA eval style: load raw .txt files, feed question as context,
single forward pass over ctx + answer, compare predictions to ground truth.

Metrics:
  - acc_exact    : all answer tokens predicted correctly (full exact match)
  - acc_last_tok : last token of the answer predicted correctly

Also reports per-layer (intermediate) accuracy.

Usage:
    python eval_deepmind_math.py --config_format base <your_usual_args> \
        --use_pretrained <ckpt_path> \
        --max_examples 2000             # optional cap per split
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import glob
import pickle
import argparse

import torch
import tiktoken
from tqdm import tqdm

import config
import models

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT    = "/leonardo_scratch/large/userexternal/edorovat/mathematics_dataset-v1.0"
ARITHMETIC_GLOB = "arithmetic__*.txt"
EVAL_SPLITS     = ["extrapolate"]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
with open('/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/data/tiktoken_gpt2.pkl', 'rb') as f:
    tiktoken_gpt2 = pickle.load(f)
enc = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)
EOT = enc.eot_token


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(split, max_examples=None):
    """Load (question, answer) pairs from all arithmetic__*.txt files in split."""
    examples = []
    pattern  = os.path.join(DATASET_ROOT, split, ARITHMETIC_GLOB)
    files    = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    for fpath in files:
        with open(fpath, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
        for i in range(0, len(lines) - 1, 2):
            q = lines[i].strip()
            a = lines[i + 1].strip()
            if q and a:
                examples.append((q, a))
                if max_examples and len(examples) >= max_examples:
                    return examples
    return examples


# ---------------------------------------------------------------------------
# Render one example
# ---------------------------------------------------------------------------

def render_example(q, a, max_seq_len=1024):
    """
    Encode question as context, answer as target.
    Template: "Q: {question} A: " + answer

    Returns (ctx_tokens, answer_tokens) or (None, None) if too long.
    """
    ctx_tokens    = enc.encode_ordinary(f"Q: {q} A: ")
    answer_tokens = enc.encode_ordinary(a)

    if len(ctx_tokens) + len(answer_tokens) > max_seq_len:
        return None, None

    return ctx_tokens, answer_tokens


# ---------------------------------------------------------------------------
# Scoring — single forward pass (mirrors LAMBADA eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_single_forward(model, ctx_tokens, answer_tokens, device):
    """
    Single forward pass over ctx + answer tokens.

    answer tokens occupy positions [seq_len-n_ans .. seq_len-1]
    they are predicted by logits at [seq_len-n_ans-1 .. seq_len-2]

    Returns:
        (final_last, final_all)  : int, int
        intermid_scores          : list of (last, all) per layer
    """
    full_tokens = ctx_tokens + answer_tokens
    tokens      = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)

    outputs  = model(tokens, get_logits=True, targets=tokens, eval_=True, return_intermid_loss=False)
    logits   = outputs['logits']    # (1, seq_len, vocab)
    intermid = outputs['intermid']  # list of (1, seq_len, vocab)

    def score(lgts):
        seq_len        = lgts.size(1)
        n_ans          = len(answer_tokens)
        pred_positions = list(range(seq_len - n_ans - 1, seq_len - 1))
        correct_flags  = [
            int(lgts[0, pos, :].argmax(dim=-1).item() == tgt)
            for pos, tgt in zip(pred_positions, answer_tokens)
        ]
        return correct_flags[-1], int(all(correct_flags))

    final_last, final_all = score(logits)
    intermid_scores       = [score(il) for il in intermid]

    return (final_last, final_all), intermid_scores


# ---------------------------------------------------------------------------
# Per-split eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model, split, device, max_examples=None):
    print(f"\n{'='*60}")
    print(f"EVALUATING ON: {split.upper()}")
    print(f"{'='*60}")

    examples = load_examples(split, max_examples=max_examples)
    print(f"Loaded {len(examples):,} examples from '{split}'.")

    num_layers = model.config.n_layer + 1

    # final layer counters
    num_correct_last = 0
    num_correct_all  = 0
    num_total        = 0

    # per-layer counters
    intermid_correct_last = [0.0] * num_layers
    intermid_correct_all  = [0.0] * num_layers

    intermid_scores = []   # keep reference for summary

    for q, a in tqdm(examples, desc=f"[{split}]"):
        ctx_tokens, answer_tokens = render_example(q, a)
        if ctx_tokens is None:
            continue

        (last_c, all_c), intermid_scores = score_single_forward(
            model, ctx_tokens, answer_tokens, device
        )

        num_correct_last += last_c
        num_correct_all  += all_c
        num_total        += 1

        for i, (lc, ac) in enumerate(intermid_scores):
            intermid_correct_last[i] += lc
            intermid_correct_all[i]  += ac

        # live print
        print(
            f"{num_total} acc_last_tok: {num_correct_last}/{num_total}="
            f"{num_correct_last/num_total:.4f} | "
            f"acc_exact: {num_correct_all}/{num_total}="
            f"{num_correct_all/num_total:.4f}"
        )
        for idx in range(len(intermid_scores)):
            print(
                f"LAYER: {idx} || {num_total} "
                f"acc_last_tok: {intermid_correct_last[idx]/num_total:.4f} | "
                f"acc_exact: {intermid_correct_all[idx]/num_total:.4f}"
            )

    # per-split summary
    print(f"\n===== {split.upper()} FINAL RESULTS =====")
    print(f"N={num_total}")
    print(
        f"Final layer  |  acc_last_tok: {num_correct_last/num_total:.4f}  |  "
        f"acc_exact: {num_correct_all/num_total:.4f}"
    )
    print("\nPer-layer (last-tok acc  |  exact-match acc):")
    for idx in range(len(intermid_scores)):
        print(
            f"  L{idx:02d}: {intermid_correct_last[idx]/num_total:.4f}  |  "
            f"{intermid_correct_all[idx]/num_total:.4f}"
        )

    return {
        'split':          split,
        'n':              num_total,
        'acc_last_tok':   num_correct_last / num_total,
        'acc_exact':      num_correct_all  / num_total,
        'intermid_last':  [v / num_total for v in intermid_correct_last],
        'intermid_exact': [v / num_total for v in intermid_correct_all],
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    device     = args.device
    model      = models.make_model_from_args(args).to(device)
    state_dict = torch.load(args.use_pretrained)
    total      = sum(v.sum().item() for v in state_dict['model'].values())
    print(f"Checkpoint checksum: {total}")
    model.load_state_dict(state_dict['model'])
    model.eval()
    print(f"ckpt from itr: {state_dict['itr']}")
    return model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base',
                        choices=config.registered_formats())
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Cap number of examples per split (useful for quick checks)')
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=parser,
        args=rem_args,
        namespace=args
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # load model once, eval on both splits
    model   = load_model(args)
    results = []

    for split in EVAL_SPLITS:
        r = evaluate_split(
            model,
            split,
            device=args.device,
            max_examples=args.max_examples,
        )
        results.append(r)

    # combined summary across both splits
    print(f"\n{'='*60}")
    print("COMBINED SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(
            f"{r['split']:>12s}  |  "
            f"acc_last_tok: {r['acc_last_tok']:.4f}  |  "
            f"acc_exact: {r['acc_exact']:.4f}  |  "
            f"N={r['n']}"
        )
