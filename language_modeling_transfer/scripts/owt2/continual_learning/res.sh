#!/bin/bash

#SBATCH --job-name=res_CL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00 #48
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/continual_learning/C1/res.out
#SBATCH --account=EUHPC_D33_268 
#SBATCH --qos=normal #boost_qos_dbg #normal #lprod #normal


source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

LR=0.00005
BS=32
ACC_STEPS=1
# TASK 1
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "CL11_boolq_res_6k_iters" --dataset 'boolq' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 1400 --n_layer 24 --eval_freq 500 --resume 'exps/owt2/hybrid/res_240k_iters/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS

# TASK 2
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "CL12_hellaswag_res_6k_iters" --dataset 'hellaswag' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 6250 --n_layer 24 --eval_freq 500 --resume 'exps/boolq/hybrid/CL11_boolq_res_6k_iters/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS

# TASK 3
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "CL13_piqa_res_2k_iters" --dataset 'piqa' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 2500 --n_layer 24 --eval_freq 500 --resume 'exps/hellaswag/hybrid/CL12_hellaswag_res_6k_iters/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS

# TASK 4
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "CL14_arc_easy_res_3k_iters" --dataset 'arc_easy' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 400 --n_layer 24 --eval_freq 500 --resume 'exps/piqa/hybrid/CL13_piqa_res_2k_iters/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS
