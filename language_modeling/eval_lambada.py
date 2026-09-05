import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
import models
import argparse
import config
import pickle
import json

LAMBADA_PATH = '/leonardo_work/EUHPC_A04_051/megatron-lm/data/lambada/lambada_test.jsonl'

with open('/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/data/tiktoken_gpt2.pkl', 'rb') as f:
    tiktoken_gpt2 = pickle.load(f)
enc = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)


def render_example(example):
    """
    For LAMBADA, the task is to predict the final word of the passage.
    We treat the passage minus the last word as context, and the last word as the target.
    Returns:
        tokens      : (1, seq_len) full sequence (context + last word)
        mask        : (1, seq_len) 1 only on the last-word tokens
        target_ids  : list of token ids that make up the last word (ground truth)
    """
    text = example["text"].strip()

    # split off the last whitespace-delimited word
    if ' ' not in text:
        return None, None, None

    split_idx = text.rfind(' ')
    context   = text[:split_idx]
    last_word = text[split_idx:]          # keeps the leading space → GPT-2 tokenisation

    ctx_tokens  = enc.encode_ordinary(context)
    last_tokens = enc.encode_ordinary(last_word)

    if len(last_tokens) == 0:
        return None, None, None

    full_tokens = ctx_tokens + last_tokens
    seq_len     = len(full_tokens)

    # skip sequences that are too long
    if seq_len > 1024:
        return None, None, None

    tokens = torch.zeros((1, seq_len), dtype=torch.long)
    mask   = torch.zeros((1, seq_len), dtype=torch.long)
    tokens[0, :seq_len] = torch.tensor(full_tokens)
    # mask covers only the last-word positions
    mask[0, len(ctx_tokens):] = 1

    return tokens, mask, last_tokens


@torch.no_grad()
def evaluate(args):
    device = args.device

    model = models.make_model_from_args(args).to(device)
    state_dict = torch.load(args.use_pretrained)
    total = sum(v.sum().item() for v in state_dict['model'].values())
    print(f"Checkpoint checksum: {total}")
    model.load_state_dict(state_dict['model'])
    model.eval()
    print('ckpt from itr: ', state_dict['itr'])

    examples = []
    with open(LAMBADA_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Data loaded, length: {len(examples)}")

    num_layers = 25  # L0..L24  (same as your other eval)

    # final-layer counters
    num_correct_last  = 0   # correct last token only
    num_correct_all   = 0   # all last-word tokens correct (full word match)
    num_total         = 0

    # per-layer counters  (same two metrics)
    intermid_correct_last = [0.0] * num_layers
    intermid_correct_all  = [0.0] * num_layers

    for example in tqdm(examples, desc="LAMBADA"):
        tokens, mask, last_tokens = render_example(example)
        if tokens is None:
            continue

        tokens = tokens.to(device)
        mask   = mask.to(device)

        outputs = model(
            tokens,
            get_logits=True,
            targets=tokens,
            eval_=True,
            return_intermid_loss=False
        )
        logits  = outputs["logits"]          # (1, seq_len, vocab)
        intermid = outputs["intermid"]        # list of (1, seq_len, vocab)

        # ------------------------------------------------------------------ #
        # helper: given a logit tensor (1, seq_len, vocab), check predictions
        # against last_tokens over the masked positions.
        #
        # "last token correct"  → the very last token of the sequence is
        #                         predicted correctly at position seq_len-2
        #                         (we predict token t from position t-1)
        # "all tokens correct"  → every token in last_word is predicted
        #                         correctly (strict full-word match)
        # ------------------------------------------------------------------ #
        def score(lgts):
            # lgts : (1, seq_len, vocab)
            seq_len   = lgts.size(1)
            n_last    = len(last_tokens)

            # positions in the INPUT that predict the last-word tokens
            # token at position p predicts position p+1
            # last-word tokens occupy indices [seq_len-n_last .. seq_len-1]
            # so they are predicted by logits at [seq_len-n_last-1 .. seq_len-2]
            pred_positions = list(range(seq_len - n_last - 1, seq_len - 1))

            correct_flags = []
            for pos, tgt in zip(pred_positions, last_tokens):
                predicted = lgts[0, pos, :].argmax(dim=-1).item()
                correct_flags.append(int(predicted == tgt))

            last_correct = correct_flags[-1]          # only the final token
            all_correct  = int(all(correct_flags))    # full word match
            return last_correct, all_correct

        # final layer
        last_c, all_c = score(logits)
        num_correct_last += last_c
        num_correct_all  += all_c
        num_total        += 1

        # intermediate layers
        for i, ilgts in enumerate(intermid):
            lc, ac = score(ilgts)
            intermid_correct_last[i] += lc
            intermid_correct_all[i]  += ac

        # live print (matches style of your existing eval)
        print(
            f"{num_total} acc_last_tok: {num_correct_last}/{num_total}="
            f"{num_correct_last/num_total:.4f} | "
            f"acc_full_word: {num_correct_all}/{num_total}="
            f"{num_correct_all/num_total:.4f}"
        )
        for idx in range(len(intermid)):
            print(
                f"LAYER: {idx} || {num_total} "
                f"acc_last_tok: {intermid_correct_last[idx]/num_total:.4f} | "
                f"acc_full_word: {intermid_correct_all[idx]/num_total:.4f}"
            )

    # ------------------------------------------------------------------ #
    # final summary
    # ------------------------------------------------------------------ #
    print("\n===== LAMBADA FINAL RESULTS =====")
    print(
        f"Final layer  |  acc_last_tok: {num_correct_last/num_total:.4f}  |  "
        f"acc_full_word: {num_correct_all/num_total:.4f}  |  N={num_total}"
    )
    print("\nPer-layer (last-tok acc  |  full-word acc):")
    for idx in range(len(intermid)):
        print(
            f"  L{idx:02d}: {intermid_correct_last[idx]/num_total:.4f}  |  "
            f"{intermid_correct_all[idx]/num_total:.4f}"
        )


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base',
                        choices=config.registered_formats())
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=parser,
        args=rem_args,
        namespace=args
    )


if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    evaluate(args)
