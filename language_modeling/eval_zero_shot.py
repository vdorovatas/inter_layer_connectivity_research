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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
from datasets import load_from_disk
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

def render_example(dataset, example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """

    if dataset == 'hellaswag':
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]
    elif dataset == 'piqa':
        ctx = example["goal"]
        label = example["label"]
        endings = [example["sol1"], example["sol2"]] 
    elif dataset == 'boolq':
        ctx = example["passage"] + "\nQuestion:" + example["question"] + "?\nAnswer:"
        label = 0 if example["answer"] == True else 1
        endings = ['yes', 'no'] 
    elif dataset == 'arc_easy':
        ctx = "Question: " + example["question"].strip() + "\nAnswer:"
        # Extract answer choices
        endings = example["choices"]["text"]   # e.g. ["Mercury", "Venus", "Earth", "Mars"]
        labels  = example["choices"]["label"]  # e.g. ["A", "B", "C", "D"]
        # Convert correct answer (like "C") to index (int)
        answer_key = example["answerKey"]
        label = labels.index(answer_key)    
        #label = endings.index(label)  
    elif dataset == 'medqa':
        ctx = example["question"] + \
              "\nA) " + example["options"]["A"] + \
              "\nB) " + example["options"]["B"] + \
              "\nC) " + example["options"]["C"] + \
              "\nD) " + example["options"]["D"] + \
              "\nAnswer:"
        endings = [example["options"]["A"], example["options"]["B"],
                   example["options"]["C"], example["options"]["D"]]
        label   = example["answer"] 
        label = endings.index(label) 
    elif dataset in ['biology', 'chemistry']:
        opts           = example['choices']['text']
        labs           = example['choices']['label']
        correct_letter = example['answerKey']
        ctx            = example['question'] + \
                         "\nA) " + opts[0] + \
                         "\nB) " + opts[1] + \
                         "\nC) " + opts[2] + \
                         "\nD) " + opts[3] + \
                        "\nAnswer:"
        endings = opts
        label   = labs.index(correct_letter)
    
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
   
    # Skip too-long cases
    if max(len(row) for row in tok_rows) > 256:
        return None, None, None, None

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    num_choices = len(endings)
    tokens = torch.zeros((num_choices, max_len), dtype=torch.long)
    mask = torch.zeros((num_choices, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(dataset, split):
    #download(split)
    examples = []
    if dataset == 'hellaswag':
        with open("hellaswag/hellaswag_val.jsonl", "r") as f:
            for line in f:
                example = json.loads(line)
                examples.append(example)
    elif dataset == 'piqa':
        examples = load_from_disk('data/datasets/piqa')['validation']
    elif dataset == 'boolq':    
        examples = load_from_disk('data/datasets/boolq')['validation']
    elif dataset == 'arc_easy':
        examples = load_from_disk('data/datasets/ARC-Easy')['validation']
    elif dataset == 'medqa':
        for ex in load_from_disk('/leonardo_scratch/large/userexternal/edorovat/MedQA-USMLE-4-options')['test']:
            examples.append(ex)
    elif dataset in ['biology', 'chemistry']:
        domain = dataset.capitalize()
        all_ex = [ex for ex in load_from_disk('/leonardo_scratch/large/userexternal/edorovat/sciknoweval_v2')['test']
                  if ex['domain'] == domain
                and ex['details']['level'] in ['L1', 'L2', 'L3'] and ex['type'] == 'mcq-4-choices']
        # same 90/10 split as tokenization — take the val 10%
        import random
        random.seed(42)
        random.shuffle(all_ex)
        split_idx = int(0.9 * len(all_ex))
        examples = all_ex[split_idx:]  # val portion only

    return examples

from transformers import GPT2LMHeadModel, GPT2Tokenizer
@torch.no_grad()
def evaluate(args):
    device = args.device
    model = models.make_model_from_args(args).to(args.device)
    state_dict = torch.load(args.use_pretrained)
    total = sum(v.sum().item() for v in state_dict['model'].values())
    print(f"Checkpoint checksum: {total}")
    model.load_state_dict(state_dict['model'])
    model.eval()
    print('ckpt from itr: ', state_dict['itr'])
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    examples = iterate_examples(args.benchmark, "val")
    print('data loaded, length: ', len(examples))
    intermid_num_correct_norm, intermid_num_correct = 25*[0.0], 25*[0.0]
    for example in examples:
        data, tokens, mask, label = render_example(args.benchmark, example)
        if data is None: continue
        #print('tokens shape: ', tokens.shape)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            # get the logits
            outputs = model(tokens, get_logits=True, targets=tokens, eval_=True, return_intermid_loss=False) #["logits"]
        logits = outputs["logits"]
        intermid = outputs["intermid"]
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_intermid_logits = [(l[..., :-1, :]).contiguous() for l in intermid]
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_intermid_logits = [s.view(-1, s.size(-1)) for s in shift_intermid_logits]
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        shift_intermid_losses = [F.cross_entropy(f, flat_shift_tokens, reduction='none').view(tokens.size(0), -1) for f in flat_shift_intermid_logits]
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        masked_shift_intermid_losses = [(s * shift_mask) for s in shift_intermid_losses]
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        sum_intermid_losses = [s.sum(dim=1) for s in masked_shift_intermid_losses]
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        avg_intermid_losses = [s / shift_mask.sum(dim=1) for s in sum_intermid_losses]
        # now we have a loss for each of the k completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        intermid_pred = [s.argmin().item() for s in sum_intermid_losses]
        intermid_pred_norm = [s.argmin().item() for s in avg_intermid_losses]
        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        for i in range(len(intermid_num_correct)):
            if label == intermid_pred[i]: intermid_num_correct[i] += 1 
        num_correct_norm += int(pred_norm == label)
        for i in range(len(intermid_num_correct_norm)):
            if label == intermid_pred_norm[i]: intermid_num_correct_norm[i] += 1
        print(f"{num_total} out of {len(examples)} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        idx=0
        for pred, norm in zip(intermid_num_correct, intermid_num_correct_norm):
            print(f"LAYER: {idx} || {num_total} out of {len(examples)} acc_norm: {norm}/{num_total}={norm/num_total:.4f}")
            idx+=1
        
def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    parser.add_argument('--benchmark', default='hellaswag', choices=['hellaswag', 'piqa', 'arc_easy', 'boolq', 'medqa', 'biology', 'chemistry'])
    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    evaluate(args)
