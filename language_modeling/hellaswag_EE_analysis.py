# Not super interesting to rewrite myself so
# Copied this file directly from 

"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import numpy as np
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
import argparse
import config
import pickle
#from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

#enc = tiktoken.get_encoding("gpt2")

with open('/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/data/tiktoken_gpt2.pkl', 'rb') as f:
    tiktoken_gpt2 = pickle.load(f)
enc = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)
'''
from transformers import GPT2Tokenizer
enc = GPT2Tokenizer.from_pretrained("local_gpt2")
'''
def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """

    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode_ordinary(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode_ordinary(" " + end) # note: prepending " " because GPT-2 tokenizer
        #end_tokens = enc.encode_ordinary(end) # do not add " " to match pretraining
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)
    
    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    #download(split)
    examples = []
    with open("hellaswag/hellaswag_val.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples

from transformers import GPT2LMHeadModel, GPT2Tokenizer
@torch.no_grad()
def evaluate(args):
    device = args.device
    #torch.set_float32_matmul_precision('high') # use tf32 
    model = models.make_model_from_args(args).to(args.device)
    state_dict = torch.load(args.use_pretrained) 
    #state_dict = torch.load('exps/owt2/learnable_depthLN_hybrid/init_025_005_240k_iters/ckpt.pt')
    #state_dict = torch.load('exps/owt2/mixln_learnable_depthLN_acn/240k_iters/ckpt.pt')
    model.load_state_dict(state_dict['model'])
    print('ckpt from itr: ', state_dict['itr'])
    # model = torch.compile(model) # optionally torch compile the model
    #model = GPT2LMHeadModel.from_pretrained('local_gpt2').to(device)
    examples = iterate_examples("val")
    print('data loaded, length: ', len(examples))

    # -------------------------------
    # SETTINGS
    # -------------------------------
    patience = 2  # number of consecutive stable layers before exit
    device = 'cuda'  # or cpu

    L = None
    V = None

    # accumulators
    agreement_sum = None
    kl_sum = None
    agreement_count = 0
    kl_count = 0

    # Early-exit accuracy counters
    num_total = 0
    num_correct_patience = 0
    exit_distribution = None

    global_total_correct = np.zeros(24, dtype=np.int64)
    global_first_correct_hist = np.zeros(24, dtype=np.int64)
    global_total_correct_and_final = np.zeros(24, dtype=np.int64)

    # -------------------------------
    # LOOP OVER EXAMPLES
    # -------------------------------
    for example in examples:

        # render example
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # forward pass
        outputs = model(
            tokens,
            get_logits=True,
            targets=tokens,
            eval_=True,
            return_intermid_loss=False
        )
        logits = outputs["logits"]             # [B, T, V] final layer
        intermid = outputs["intermid"][1:]     # skip embeddings

        # shift for autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_intermid_logits = [l[..., :-1, :].contiguous() for l in intermid]
        shift_labels = tokens[..., 1:].contiguous()   # [B, T]
        shift_mask = mask[..., 1:].contiguous()      # [B, T]
        mask_bool = shift_mask.bool()

        B, T = shift_mask.shape
        if L is None:
            L = len(shift_intermid_logits)
            V = shift_logits.size(-1)
            agreement_sum = torch.zeros(L, device=device)
            kl_sum = torch.zeros(L, device=device)
            exit_distribution = torch.zeros(L, device=device)

        # -------------------------------
        # 1️⃣ ORACLE METRICS
        # -------------------------------
        final_logits = shift_logits
        final_pred = final_logits.argmax(dim=-1)
        final_probs = F.softmax(final_logits, dim=-1)

        for i, logits_i in enumerate(shift_intermid_logits):
            pred_i = logits_i.argmax(dim=-1)

            # prediction agreement
            agree = (pred_i == final_pred) & mask_bool
            agreement_sum[i] += agree.sum()

            # KL divergence
            log_probs_i = F.log_softmax(logits_i, dim=-1)
            kl = F.kl_div(
                log_probs_i,
                final_probs,
                reduction='none'
            ).sum(dim=-1)
            kl_sum[i] += kl[mask_bool].sum()

        agreement_count += mask_bool.sum()
        kl_count += mask_bool.sum()

        '''
        # -------------------------------
        # 2️⃣ ZERO-SHOT EARLY EXIT (PATIENCE BASED)
        # -------------------------------
        exit_layer = torch.full((B, T), L - 1, dtype=torch.long, device=device)
        stable_count = torch.zeros((B, T), device=device)
        exited = torch.zeros((B, T), dtype=torch.bool, device=device)
        prev_pred = None

        for i, logits_i in enumerate(shift_intermid_logits):
            pred_i = logits_i.argmax(dim=-1)

            if prev_pred is not None:
                same = (pred_i == prev_pred) & mask_bool & (~exited)
                stable_count[same] += 1
                stable_count[~same] = 0
                should_exit = (stable_count >= patience) & (~exited)

                exit_layer[should_exit] = i
                exited[should_exit] = True

            prev_pred = pred_i

        # -------------------------------
        # 3️⃣ CHOOSE COMPLETION WITH HIGHEST LOG-LIKELIHOOD
        # -------------------------------
        # compute per-completion total NLL (sum over tokens)
        shift_losses = torch.zeros((B, T), device=device)
        for l in range(L):
            layer_mask = (exit_layer == l)
            if layer_mask.any():
                logits_l = shift_intermid_logits[l]
                flat_logits = logits_l.view(-1, V)
                flat_targets = shift_labels.view(-1)
                loss_l = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, T)
                shift_losses[layer_mask] = loss_l[layer_mask]

        # sum per completion
        total_nll_per_completion = (shift_losses * shift_mask).sum(dim=1)
        # choose completion with highest log-likelihood (lowest NLL)
        pred = total_nll_per_completion.argmin().item()

        # accuracy
        if pred == label:
            num_correct_patience += 1

        # record exit distribution for the chosen completion
        for l in range(L):
            exit_distribution[l] += (exit_layer[pred] == l).sum().item()

        '''        
        # -------------------------------
        # 4️⃣ PER-LAYER CORRECTNESS / HISTOGRAMS
        # -------------------------------
        # initialize
        total_correct_per_layer = torch.zeros(L, dtype=torch.long, device=device)
        first_correct_layer = torch.full((B, T), L-1, dtype=torch.long, device=device)
        already_correct = torch.zeros((B, T), dtype=torch.bool, device=device)
        total_correct_and_final = torch.zeros(L, dtype=torch.long, device=device)

        # final-layer predictions for EE-relevant filter
        final_pred = shift_intermid_logits[-1].argmax(dim=-1)  # [B, T]
        final_correct_mask = (final_pred == shift_labels)     # [B, T]

        for l, logits_l in enumerate(shift_intermid_logits):
            pred_tokens = logits_l.argmax(dim=-1)  # [B, T]

            # 1️⃣ Total correct at this layer
            correct = (pred_tokens == shift_labels)
            total_correct_per_layer[l] += correct.sum().item()

            # 2️⃣ Histogram: first correct layer
            correct_here = correct & (~already_correct)
            first_correct_layer[correct_here] = l
            already_correct |= correct_here

            # 3️⃣ Correct and final-layer match
            correct_and_final = correct & final_correct_mask
            total_correct_and_final[l] += correct_and_final.sum().item()

        # histogram of first correct layer
        hist_first_correct = torch.bincount(first_correct_layer.view(-1), minlength=L)

        # accumulate across batches
        global_total_correct += total_correct_per_layer.cpu().numpy()
        global_first_correct_hist += hist_first_correct.cpu().numpy()
        global_total_correct_and_final += total_correct_and_final.cpu().numpy()

        num_total += 1
    # -------------------------------
    # 4️⃣ REPORT RESULTS
    # -------------------------------

    print("========== ORACLE AGREEMENT ==========")
    for i in range(L):
        agreement = agreement_sum[i] / agreement_count
        print(f"Layer {i:02d}: {agreement.item():.4f}")

    print("\n========== KL TO FINAL ==========")
    for i in range(L):
        mean_kl = kl_sum[i] / kl_count
        print(f"Layer {i:02d}: {mean_kl.item():.4f}")

    '''
    print("\n========== ZERO-SHOT PATIENCE-BASED ACCURACY ==========")
    print(f"Accuracy: {num_correct_patience / num_total:.4f}")
    print("Exit distribution per layer:", exit_distribution.tolist())
    '''
    print("========== TOTAL TOKENS CORRECT PER LAYER ==========")
    for l in range(L):
        print(f"Layer {l:02d}: {global_total_correct[l]}")

    print("\n========== HISTOGRAM OF FIRST CORRECT LAYER ==========")
    for l in range(L):
        print(f"Layer {l:02d}: {global_first_correct_hist[l]}")

    print("\n========== TOKENS CORRECT AND MATCH FINAL LAYER ==========")
    for l in range(L):
        print(f"Layer {l:02d}: {global_total_correct_and_final[l]}")

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

def entropy_safe(logits, eps=1e-12):
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.exp(logp)

    # clamp to avoid -inf
    logp = torch.clamp(logp, min=-1e9)
    ent = -(p * logp).sum(dim=-1)
    return ent

def top12_diff(logits):
    logp = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(logp)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return margin

if __name__ == "__main__":
    args = get_args()
    evaluate(args)
