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
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    examples = iterate_examples("val")
    print('data loaded, length: ', len(examples))
    exit_distribution = [0 for _ in range(24)]
    for example in examples:
        data, tokens, mask, label = render_example(example)
        #print('tokens shape: ', tokens.shape)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        outputs = model(tokens, get_logits=True, targets=tokens, eval_=True, return_intermid_loss=False) #["logits"]
        logits = outputs["logits"]
        intermid = outputs["intermid"]
        intermid = intermid[1:]
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous() #.view(-1, logits.size(-1))
        shift_intermid_logits = [(l[..., :-1, :]).contiguous() for l in intermid]
        flat_shift_tokens = (tokens[..., 1:]).contiguous().view(-1)
        ## EE logic
        
        shift_mask = (mask[..., 1:]).contiguous()
        list_ent = []
        found = False
        for i, inter in enumerate(shift_intermid_logits):
            entropy = entropy_safe(inter)
            entropy_completion = (entropy * shift_mask).sum() / shift_mask.sum().clamp_min(1)
            #entropy = top12_diff(inter)   # [B, T]
            #entropy_completion = (entropy * shift_mask).sum() / shift_mask.sum().clamp_min(1)
            list_ent.append(entropy_completion.item())
            if entropy_completion < args.threshold:
                shift_logits = inter
                exit_distribution[i] += 1
                found = True
                break
                
        print('entropies: ', list_ent)
        if not found: exit_distribution[-1] += 1
        ##
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none').view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label) 
        print(f"{num_total} out of 10042 acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}") 
        print('Exit distribution: ', exit_distribution)
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
