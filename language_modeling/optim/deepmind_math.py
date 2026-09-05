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

from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb
import time 
import copy
import traceback
from tqdm import tqdm
from .utils import eval, get_batch, get_batch_math, save_checkpoint
import os

def train_base(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']['tokens']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()

    progress_bar = tqdm(total=iterations, desc="Train iteration", initial=itr, disable=not distributed_backend.is_master_process())
    #################### INIT weights ################################
    alphas_list = ['hybrid', 'long_short', 'gated_acn']
    if extra_args.model in alphas_list:
        print('~ Init Alphas ~')
        print([f"{p.item():.3f}" for p in model.module.alphas])
        print('####')
        if extra_args.depthLN:
            print('Attn DepthLN:', [f"{b.ln_1.strength.item():.3f}" for b in model.module.transformer.h])
            print('MLP  DepthLN:', [f"{b.ln_2.strength.item():.3f}" for b in model.module.transformer.h])
            print('####')
    if extra_args.model == 'denseformer' or extra_args.model == 'grn_v2':
        biases = [layer.qkv_grn.bias for layer in model.module.transformer.h]
        log_layer_biases(biases, itr, ckpt_path)
    if extra_args.model == 'resW_scaled_x':
        mats = collect_residual_mats(model.module)
        np.save(f"/leonardo_scratch/large/userexternal/edorovat/resW_weights/residuals_step_0.npy", mats)
    ##################################################################
    while itr < iterations:
                
        for microstep_idx in range(acc_steps):  # gradient accumulation
            if extra_args.dataset == 'deepmind_math': 
                x, y = get_batch_math(data['train'], sequence_length, batch_size, device=extra_args.device)
            else:
                x, y = get_batch(data['train'], sequence_length, batch_size, device=extra_args.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    if getattr(distributed_backend.get_raw_model(model), "needs_iter", False):
                        outputs = model(x, targets=y, iter=itr, ee=extra_args.ee_training)
                    else:
                        outputs = model(x, targets=y, ee=extra_args.ee_training)

            if extra_args.ee_training: loss = outputs["intermid"]
            else: loss = outputs['loss']
            #################### L1 Loss on alphas ########################
            if extra_args.l1_regularizer_lamda > 0:
                alpha_usage = 0.0
                for alpha in model.module.alphas:
                    alpha_usage = alpha_usage + (1.0 - alpha).abs().sum()
                loss = loss + extra_args.l1_regularizer_lamda * alpha_usage
            
            ## Binarization loss -- push alphas towards 1 (pure long) or 0 (pure short)
            if extra_args.binary_regularizer_lamda > 0:
                bin_loss = 0.0
                for alpha in model.module.alphas:
                    bin_loss += (alpha * (1.0 - alpha)).sum()

                loss += extra_args.binary_regularizer_lamda * bin_loss
            ###############################################################
            loss.backward()
            ###
            if (extra_args.wandb and distributed_backend.is_master_process()) and (itr % eval_freq == 0 or itr == iterations): log_gradients(model, itr)
            ###
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        
        ######
        if extra_args.alphas_clamp and extra_args.model in ['hybrid', 'long_short']: # For Hybrid & LSN
            with torch.no_grad():
                for alpha in model.module.alphas:
                    alpha.clamp_(0.0, 1.0)
        elif extra_args.model == 'denseformer' or extra_args.model == 'denseformer_orig':
            with torch.no_grad():
                for layer in model.module.transformer.h:
                    layer.qkv_grn.bias.data.clamp_(-1.0, 1.0)
        elif extra_args.model == 'res_scaled_x' or extra_args.model == 'res_scaled_f':
            with torch.no_grad():
                for layer in model.module.transformer.h:
                    layer.gain_attn.clamp_(0.0, 1.0)
        ######
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        progress_bar.update(1)  
        loss_value = loss.mean().item()
        ppl = 2.71828 ** loss_value
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
        progress_bar.set_postfix(global_step=itr, loss=loss_value, ppl=ppl,  lr=current_lr) 
        
        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_loss, val_perplexity, intermid_ppl, activations_list = eval(model, data['val'], sequence_length, batch_size,
                                                         extra_args.device, max_num_batches=24, ctx=type_ctx, extra_args=extra_args)
                val_hard_acc, val_hard_loss, val_hard_perplexity, hard_intermid_ppl, activations_list = eval(model, data['val_hard'], sequence_length, batch_size, extra_args.device, max_num_batches=24, ctx=type_ctx, extra_args=extra_args)
                print_string = f"{epoch}/{itr}: [train] loss={train_loss:.3f} | [val] loss={val_loss:.3f} | [val] ppl={val_perplexity:.2f} | [val] acc={val_acc:3f} | [val_hard] loss={val_hard_loss:.3f} | [val_hard] ppl={val_hard_perplexity:.2f} | [val_hard] acc={val_hard_acc:3f}"
                print_string += f" | [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" | [lr] {current_lr:.5f}"
                print(print_string)
                print(f"## Intermid [val] Loss ##\n{[f'{p:.2f}' for p in intermid_ppl]}\n####")
                print(f"## Intermid [val_hard] Loss ##\n{[f'{p:.2f}' for p in hard_intermid_ppl]}\n####")
                alphas_list = ['hybrid', 'long_short', 'gated_acn']
                if extra_args.model in alphas_list:
                    print('~ Alphas ~')
                    print([f"{p.item():.3f}" for p in model.module.alphas])
                    print('####')
                    if extra_args.depthLN:
                        print('Attn DepthLN:', [f"{b.ln_1.strength.item():.3f}" for b in model.module.transformer.h])
                        print('MLP  DepthLN:', [f"{b.ln_2.strength.item():.3f}" for b in model.module.transformer.h])
                        print('####')
                elif extra_args.model == 'denseformer' or extra_args.model == 'grn_v2' or extra_args.model == 'denseformer_orig':
                    biases = [layer.qkv_grn.bias for layer in model.module.transformer.h]
                    log_layer_biases(biases, itr, ckpt_path)
                elif extra_args.model == 'resW_scaled_x':
                    mats = collect_residual_mats(model.module)
                    np.save(f"/leonardo_scratch/large/userexternal/edorovat/resW_weights/residuals_step_{itr}.npy", mats)
                elif extra_args.model == 'res_scaled_x' or extra_args.model == 'res_scaled_f':
                    w_attn = [layer.gain_attn.item() for layer in model.module.transformer.h]
                    #w_mlp = [layer.gain_mlp for layer in model.module.transformer.h]
                    print('~ Weights attn~')
                    print(w_attn)
                    print('---------')
                    print('####')
                '''
                elif extra_args.model == 'denseformer_orig':
                    all_weights = [weight_layer.weight.data for weight_layer in model.module.weights]
                    log_layer_biases(all_weights, itr, ckpt_path)
                '''

                if extra_args.wandb:
                    #print('wandb logging..')
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    }, commit=True)
                    log_activation_stats(activations_list, itr)
                    log_rank_and_condition(activations_list, itr)
                    log_attn_entropy(activations_list, itr)
                    log_anisotropy(activations_list, itr)
                    del activations_list 

                model.train()
                t0 = time.time()
        if distributed_backend.is_master_process():
            if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
                print(f"saving checkpoint to {ckpt_path}/ckpt_{itr}.pt")
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                ckpt_path=f"{ckpt_path}/ckpt_{itr}.pt")

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=f"{ckpt_path}/ckpt.pt")

    return stats

def log_activation_stats(activations_list, itr):
    norm_logs = {}
    mean_logs = {}
    var_logs = {}

    for i, act in enumerate(activations_list):
        layer = f"layer{i}"

        # Stats
        norm_val = act[0].norm().item()
        mean_val = act[0].mean().item()
        var_val = act[0].var().item()

        # Prefix-based keys → W&B auto-groups into one plot each
        norm_logs[f"activations/norm/{layer}"] = norm_val
        mean_logs[f"activations/mean/{layer}"] = mean_val
        var_logs[f"activations/var/{layer}"] = var_val

    # Log everything at once
    wandb.log({
        **norm_logs,
        **mean_logs,
        **var_logs,
        "train/iteration": itr,
    })
  
def log_gradients(model, itr):
    fc_logs = {}
    proj_logs = {}
    attn_logs = {}
    attn_proj_logs = {}

    # Blocks
    for i, block in enumerate(model.module.transformer.h):
        layer = f"layer{i}"

        # Compute norms
        fc_norm = block.mlp.c_fc.weight.grad.norm().item()
        proj_norm = block.mlp.c_proj.weight.grad.norm().item()
        attn_norm = block.attn.c_attn.weight.grad.norm().item()
        attn_proj_norm = block.attn.c_proj.weight.grad.norm().item()

        # Fill dicts using consistent prefixes
        fc_logs[f"grads/fc/{layer}"] = fc_norm
        proj_logs[f"grads/proj/{layer}"] = proj_norm
        attn_logs[f"grads/attn/{layer}"] = attn_norm
        attn_proj_logs[f"grads/attn_proj/{layer}"] = attn_proj_norm

    # Log everything in ONE wandb.log call
    wandb.log({
        **fc_logs,
        **proj_logs,
        **attn_logs,
        **attn_proj_logs,
        "train/iteration": itr,
    })
    
def log_attn_entropy(activations_list, itr):
    mean_logs = {}
    var_logs = {}

    for i, act in enumerate(activations_list):
        layer = f"layer{i}"

        mean_entropy = act[3].mean()
        std_entropy  = act[3].std()

        mean_logs[f"attention_entropy/mean/{layer}"] = mean_entropy.item()
        var_logs[f"attention_entropy/std/{layer}"]  = std_entropy.item()

    # Log everything at once
    wandb.log({
        **mean_logs,
        **var_logs,
        "train/iteration": itr,
    })

def log_anisotropy(activations_list, itr):
    full_logs = {}
    attn_logs = {}
    mlp_logs  = {}

    # Skip first block if you want to match your other metrics
    for i, (full_out, attn_out, mlp_out, attn_entropy) in enumerate(activations_list[1:]):
        layer = f"layer{i}"

        # Compute anisotropy
        full_aniso = anisotropy(full_out)
        attn_aniso = anisotropy(attn_out)
        mlp_aniso  = anisotropy(mlp_out)

        # Log
        full_logs[f"anisotropy/full/{layer}"] = full_aniso.item()
        attn_logs[f"anisotropy/attn/{layer}"] = attn_aniso.item()
        mlp_logs[f"anisotropy/mlp/{layer}"]  = mlp_aniso.item()

    wandb.log({
        **full_logs,
        **attn_logs,
        **mlp_logs,
        "train/iteration": itr,
    })
    
def log_rank_and_condition(activations_list, itr):
    full_logs = {}
    attn_logs = {}
    mlp_logs  = {}

    # Skip the first block's activations to match original behavior
    for i, (full_out, attn_out, mlp_out, _) in enumerate(activations_list[1:]):
        layer = f"layer{i}"

        # Flatten [batch, seq, hidden] → [batch*seq, hidden]
        full_flat = full_out.reshape(-1, full_out.shape[-1])
        attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
        mlp_flat  = mlp_out.reshape(-1, mlp_out.shape[-1])

        # Compute metrics
        full_rank, full_cond, full_smax, full_smin = rank_and_condition_number(full_flat)
        attn_rank, attn_cond, attn_smax, attn_smin = rank_and_condition_number(attn_flat)
        mlp_rank,  mlp_cond,  mlp_smax,  mlp_smin  = rank_and_condition_number(mlp_flat)

        # Full block
        full_logs[f"effective_rank/full/{layer}"] = full_rank
        full_logs[f"cond_num/full/{layer}"] = full_cond
        full_logs[f"s_max/full/{layer}"] = full_smax
        full_logs[f"s_min/full/{layer}"] = full_smin

        # Attention
        attn_logs[f"effective_rank/attn/{layer}"] = attn_rank
        attn_logs[f"cond_num/attn/{layer}"] = attn_cond
        attn_logs[f"s_max/attn/{layer}"] = attn_smax
        attn_logs[f"s_min/attn/{layer}"] = attn_smin

        # MLP
        mlp_logs[f"effective_rank/mlp/{layer}"] = mlp_rank
        mlp_logs[f"cond_num/mlp/{layer}"] = mlp_cond
        mlp_logs[f"s_max/mlp/{layer}"] = mlp_smax
        mlp_logs[f"s_min/mlp/{layer}"] = mlp_smin

    # Log
    wandb.log({
        **full_logs,
        **attn_logs,
        **mlp_logs,
        "train/iteration": itr,
    })


def rank_and_condition_number(x, fast=True, eps=1e-12):
    """
    Compute both Shannon effective rank and spectral condition number
    for a 2D tensor x, using a single SVD.
    Returns: (effective_rank, condition_number)
    """
    x = x.to(torch.float32)
    
    if not fast: _, s, _ = torch.linalg.svd(x.detach().cpu(), full_matrices=False)
    else:
        X_cpu = x.detach().cpu()
        XtX = X_cpu.T @ X_cpu
        eigvals = torch.linalg.eigvalsh(XtX)
        s = torch.sqrt(torch.clamp(eigvals, min=0)) + eps  # singular values

    # ---- Effective rank ----
    s_safe = s + eps                  # avoid log(0)
    p = s_safe / s_safe.sum()
    entropy = -(p * p.log()).sum()
    eff_rank = entropy.exp().item()

    # ---- Condition number ----
    cond = (s.max() / (s.min() + eps)).item()

    return eff_rank, cond, s.max(), s.min()

def anisotropy(x):
    # x: [B, S, D] or [B*S, D] — handle both
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])

    mean_norm_sq = (x.pow(2).sum(dim=-1).mean())
    mean_vec_norm_sq = (x.mean(dim=0).pow(2).sum())
    return torch.log(mean_norm_sq / (mean_vec_norm_sq + 1e-12))


def log_layer_biases(bias_tensors, step, save_dir):
    """
    Appends layer-wise bias weights to a log file.
    Each row in the log corresponds to the weights used by a particular layer
    to combine outputs from previous layers (like residuals or fusions).
    Args:
        bias_tensors (List[torch.Tensor]): list of tensors with shape (1, num_layers)
        step (int): current training step or iteration
        save_path (str): path to the log file
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "layer_weights_evolution.txt")
    mode = 'w' if step == 0 else 'a'
    with open(save_path, mode) as f:
        f.write(f"\n### Step {step} ###\n")
        for bias in bias_tensors:
            if isinstance(bias, torch.Tensor):
                if bias.dim() == 2: bias_list = bias.detach().cpu().squeeze(0).tolist()
                else: 
                    bias_list = torch.norm(bias.squeeze(0), dim=-1).detach().cpu().tolist()
                line = " ".join(f"{val:.6f}" for val in bias_list)
                f.write(line + "\n")
            else:
                raise ValueError("All inputs must be torch.Tensors")

import numpy as np

def collect_residual_mats(model):
    mats = []
    for block in model.transformer.h:
        W = block.gain_attn.weight.detach().cpu().numpy()
        mats.append(W)
    return np.stack(mats)   # (num_layers, d, d)
