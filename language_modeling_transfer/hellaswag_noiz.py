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

import random

def add_typo_noise(text, prob=0.1):
    """
    Add simple character-level noise to simulate typos.
    Each character has `prob` chance of being noised.
    Noise operations: deletion, insertion, swap.
    """
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() < prob:
            op = random.choice(["delete", "insert", "swap"])
            if op == "delete" and len(chars) > 1:
                del chars[i]
                continue
            elif op == "insert":
                chars.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
            elif op == "swap" and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
                i += 1  # skip ahead after swap
        i += 1
    return "".join(chars)


def render_example_with_noise(example, noise_prob=0.03):
    """
    Same as render_example, but applies character-level noise to ctx and endings.
    """
    ctx = add_typo_noise(example["ctx"], prob=noise_prob)
    label = example["label"]
    endings = [add_typo_noise(end, prob=noise_prob) for end in example["endings"]]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode_ordinary(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode_ordinary(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

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
    args.model = 'learnable_depthLN_hybrid' 
    model = models.make_model_from_args(args).to(args.device)
    #state_dict = torch.load("exps/owt2/residual/240k_iters/ckpt.pt") 
    state_dict = torch.load('exps/owt2/learnable_depthLN_hybrid/init_025_005_240k_iters/ckpt.pt')
    model.load_state_dict(state_dict['model'])
    print('ckpt from itr: ', state_dict['itr'])
    # model = torch.compile(model) # optionally torch compile the model
    #model = GPT2LMHeadModel.from_pretrained('local_gpt2').to(device)
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    examples = iterate_examples("val")
    print('data loaded, length: ', len(examples))
    for example in examples:
        data, tokens, mask, label = render_example_with_noise(example)
        #print('tokens shape: ', tokens.shape)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens, get_logits=True, targets=tokens)["logits"]
        #logits = model(tokens).logits

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
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
        #tqdm.write(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        
        # debug: pretty print a few examples, and the losses in each case
        if False: #num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)
