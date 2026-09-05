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

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack

'''
def get_batch(data, seq_length, batch_size, device='cpu'):
    ix = torch.randint(len(data) - seq_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    if device != 'cpu':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y
'''
def get_batch(data, seq_length, batch_size, device, mask_data=None):
    if mask_data is None:

        ix = torch.randint(len(data) - seq_length - 1, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y, None
    else:
        ix = torch.randint(len(data), (batch_size,))  # index examples, not token positions
    
        x = torch.stack([torch.from_numpy(data[i, :-1].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i, 1:].astype(np.int64))  for i in ix])
    
        mask = None
        if mask_data is not None:
            # mask for y (shifted by 1, predicting next token)
            mask = torch.stack([torch.from_numpy(mask_data[i, 1:].astype(np.float32)) for i in ix])
            mask = mask.to(device)
    
        x, y = x.to(device), y.to(device)
        return x, y, mask

def get_batch_math(data, seq_length, batch_size, device='cpu'):
    if isinstance(data, dict):
        tokens = data['tokens']
        labels = data['labels']
        use_labels = True
    else:
        tokens = data
        labels = None
        use_labels = False

    # -1 for the +1 shift, same in both modes
    max_start = len(tokens) - seq_length - 1
    ix = torch.randint(max_start, (batch_size,))

    x = torch.stack([
        torch.from_numpy(tokens[i : i + seq_length].astype(np.int64))
        for i in ix
    ])

    if use_labels:
        # +1 shift on labels: x[t] predicts labels[t+1]
        # -100 positions are still correctly ignored by cross_entropy
        y = torch.stack([
            torch.from_numpy(labels[i + 1 : i + 1 + seq_length].astype(np.int64))
            for i in ix
        ])
    else:
        y = torch.stack([
            torch.from_numpy(tokens[i + 1 : i + 1 + seq_length].astype(np.int64))
            for i in ix
        ])

    if device != 'cpu':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    return x, y

@torch.no_grad()
def eval(model, data_tensor, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext(), extra_args=None, data_mask=None):
    assert model.training == False

    loss_list_val, acc_list = [], []
    intermid_list_val = [[] for _ in range(model.module.config.n_layer+1)]
    for _ in range(max_num_batches):
        if extra_args.dataset == 'deepmind_math':
            x, y = get_batch_math(data_tensor, sequence_length, batch_size, device=device)    
        else:
            x, y, mask = get_batch(data_tensor, sequence_length, batch_size, device=device, mask_data=data_mask)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, eval_=True)

        # apply mask if available
        if mask is not None:
            val_loss = (outputs['loss_unreduced'] * mask).sum() / mask.sum().clamp(min=1)
        else:
            val_loss = outputs['loss']
        
        loss_list_val.append(val_loss)
        ####
        for i, inter in enumerate(outputs['intermid']):
            intermid_list_val[i].append(inter)
        ####
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    ### extra forward to get activations
    #with ctx: activations = model(x, targets=y, get_logits=True, eval_=True, get_activations=True)['activations']
    activations = None
    ###
    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    ####
    intermid_val_loss = [torch.stack(l).mean().item() for l in intermid_list_val]
    if extra_args.dataset == 'deepmind_math': intermid_val_ppl = intermid_val_loss
    else: intermid_val_ppl = [(2.71828 ** l) for l in intermid_val_loss]
    del intermid_val_loss
    ####

    return val_acc, val_loss, val_perplexity, intermid_val_ppl, activations

def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)


