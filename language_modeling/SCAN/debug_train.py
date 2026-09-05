"""
scan_train.py — Training loop for SCAN compositional generalization experiments.

Usage:
    python scan_train.py \
        --scan_dir /path/to/SCAN \
        --split simple \
        --model residual \
        --out_dir ./exps/scan_simple_residual

For your custom model, pass --model <your_model_name> and the script will
import it from models/ exactly as main.py does, but route through
forward_scan() / forward() transparently.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import SCANTokenizer, get_dataloaders
from res import ResidualLM, SCANConfig
from eval import evaluate_split
from lsn import ModularLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float,
           scheduler: str) -> float:
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    if scheduler == "none":
        return lr_max
    # cosine decay to 10% of peak
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if scheduler == "cos":
        return lr_max * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    if scheduler == "linear":
        return lr_max * (1.0 - 0.9 * progress)
    return lr_max


def build_model(args, tokenizer: SCANTokenizer) -> nn.Module:
    cfg = SCANConfig(
        vocab_size      = tokenizer.vocab_size,
        sequence_length = args.sequence_length,
        n_layer         = args.n_layer,
        n_head          = args.n_head,
        n_embd          = args.n_embd,
        dropout         = args.dropout,
        bias            = args.bias,
    )
    if args.model == "residual":
        model = ResidualLM(cfg)
    elif args.model == "lsn":
        model = ModularLM(cfg)
    else:
        raise NotImplementedError(
            f"Model '{args.model}' not registered. "
            "Add an elif branch here pointing to your model class."
        )
    return model


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    groups = model.get_parameter_group_specs()
    # resolve param names → actual tensors
    param_dict = {pn: p for pn, p in model.named_parameters()}
    resolved = []
    for g in groups:
        resolved.append({
            **{k: v for k, v in g.items() if k != "params"},
            "params": [param_dict[n] for n in g["params"] if n in param_dict],
        })
    if args.opt == "adamw":
        return torch.optim.AdamW(
            resolved, lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(resolved, lr=args.lr, momentum=0.9,
                           weight_decay=args.weight_decay)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------------
    tokenizer = SCANTokenizer()
    train_loader, test_loader = get_dataloaders(
        scan_dir   = args.scan_dir,
        split      = args.split,
        tokenizer  = tokenizer,
        batch_size = args.batch_size,
        max_len    = args.sequence_length,
    )
    print('Data Ready.')
    # ---- model --------------------------------------------------------------
    model = build_model(args, tokenizer).to(device)
    if not args.no_compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile() failed ({e}), running eager.")
    print('Model Ready.')
    # ---- optimizer ----------------------------------------------------------
    optimizer = build_optimizer(model, args)
    warmup_steps = int(args.warmup_percent * args.iterations)

    # ---- dtype --------------------------------------------------------------
    dtype = getattr(torch, args.dtype.replace("torch.", ""))
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)

    # ---- logging ------------------------------------------------------------
    log = {
        "args": vars(args),
        "train_loss": [],   # (step, loss)
        "eval": [],         # (step, split, layer_idx, exact_match, loss)
    }

    # ---- infinite train iterator --------------------------------------------
    
    def cycle(loader):
        while True:
            yield from loader

    data_iter = cycle(train_loader)
    '''
    # hack: replace data iterator
    batch = next(iter(train_loader))

    def cycle_one():
        while True:
            yield batch

    data_iter = cycle_one()
    '''
    print('Starting...')
    print(f"\n{'='*60}")
    print(f"  SCAN | split={args.split} | model={args.model}")
    print(f"  iters={args.iterations} | lr={args.lr} | n_layer={args.n_layer}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for step in range(1, args.iterations + 1):

        # lr schedule
        lr = get_lr(step, warmup_steps, args.iterations, args.lr, args.scheduler)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- gradient accumulation ------------------------------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        
        for micro_step in range(args.acc_steps):
            idx, targets = next(data_iter)
            idx, targets = idx.to(device), targets.to(device)

            with autocast_ctx:
                out  = model(idx, targets, return_intermid=False)
                loss = out["loss"] / args.acc_steps

            loss.backward()
            #print('Loss: ', loss.item())
            '''
            with torch.no_grad():
                logits = out["logits"]
                print("logits mean/std:",logits.mean().item(),logits.std().item())
                preds = logits.argmax(-1)
                print("pred tokens:", preds[0, -10:].tolist())
                g = []
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        g.append(p.grad.norm().item())
                print("grad norm (mean/max):",sum(g)/len(g), max(g))
            '''
            ###############
            total_loss += loss.item()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        
        if args.model == "lsn":
            with torch.no_grad():
                for alpha in model.alphas:
                    alpha.clamp_(0.0, 1.0)
        # ---- logging --------------------------------------------------------
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d}/{args.iterations} | loss {total_loss:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.1f}s")
            log["train_loss"].append((step, total_loss))
            t0 = time.time()

        # ---- eval -----------------------------------------------------------
        if step % args.eval_freq == 0 or step == args.iterations:
            print(f"\n--- Eval at step {step} ---")
            if args.model == "lsn":
                print('Alphas')
                print([f"{p.item():.3f}" for p in model.alphas])
            
            results = evaluate_split(
                model       = model,
                loader      = test_loader,
                tokenizer   = tokenizer,
                device      = device,
                split_name  = args.split,
                max_decode  = args.sequence_length,
            )
            for layer_idx, (em, ce) in enumerate(
                zip(results["exact_match_per_layer"], results["loss_per_layer"])
            ):
                label = "embed" if layer_idx == 0 else f"layer{layer_idx}"
                print(f"  {label:10s}  EM={em*100:.1f}%  CE={ce:.4f}")
                log["eval"].append((step, args.split, layer_idx, em, ce))
            print(f"  {'FINAL':10s}  EM={results['exact_match_final']*100:.1f}%")
            print()

    # ---- save ---------------------------------------------------------------
    if False:
        ckpt_path = out_dir / "model_final.pt"
        torch.save({
            "model_state": model.state_dict(),
            "args": vars(args),
            "step": args.iterations,
        }, ckpt_path)

    log_path = out_dir / "log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    #print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Saved log        → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("SCAN Training — Residual Baseline")

    # paths
    p.add_argument("--scan_dir",  required=True,  type=str,
                   help="Path to cloned brendenlake/SCAN repo")
    p.add_argument("--out_dir",   default="./exps/scan", type=str)
    p.add_argument(
        "--split",
        default="simple",
        choices=[
            "simple",
            "length",
            "add_prim_jump",
            "add_prim_turn_left",
            "add_prim_jump_num2_rep1",
            "add_prim_jump_num2_rep2",
            "add_prim_jump_num4_rep1",
            "add_prim_jump_num4_rep2",
            "add_prim_jump_num8_rep1",
            "add_prim_jump_num8_rep2",
            "add_prim_jump_num16_rep1",
            "add_prim_jump_num16_rep2",
            "add_prim_jump_num32_rep1",
            "add_prim_jump_num32_rep2",
            "simple_p1",
            "simple_p2",
            "simple_p4",
            "simple_p8",
            "simple_p16",
            "simple_p32",
        ],
    )
    
    # training
    p.add_argument("--iterations",       default=20000, type=int)
    p.add_argument("--batch_size",       default=128,   type=int)
    p.add_argument("--acc_steps",        default=1,     type=int)
    p.add_argument("--lr",               default=3e-4,  type=float)
    p.add_argument("--warmup_percent",   default=0.05,  type=float)
    p.add_argument("--weight_decay",     default=1e-2,  type=float)
    p.add_argument("--beta1",            default=0.9,   type=float)
    p.add_argument("--beta2",            default=0.95,  type=float)
    p.add_argument("--scheduler",        default="cos",
                   choices=["cos", "linear", "none"])
    p.add_argument("--opt",              default="adamw",
                   choices=["adamw", "sgd"])
    p.add_argument("--grad_clip",        default=1.0,   type=float)
    p.add_argument("--seed",             default=2,     type=int)
    p.add_argument("--eval_freq",        default=500,   type=int)

    # model
    p.add_argument("--model",            default="residual", type=str)
    p.add_argument("--n_layer",          default=6,     type=int)
    p.add_argument("--n_head",           default=6,     type=int)
    p.add_argument("--n_embd",           default=192,   type=int)
    p.add_argument("--sequence_length",  default=128,   type=int)
    p.add_argument("--dropout",          default=0.1,   type=float)
    p.add_argument("--bias",             default=False, type=bool)
    p.add_argument("--dtype",            default="torch.float32", type=str)
    p.add_argument("--no_compile",       action="store_true")
    p.add_argument('--device', default='cuda:0', type=str)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('In')
    train(args)
