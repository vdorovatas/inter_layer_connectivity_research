#!/bin/bash

#SBATCH --job-name=eval_0shot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/eval/deepmind_math_res.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_D33_268 #EUHPC_A04_051
#SBATCH --qos=boost_qos_dbg

source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

python -u eval_deepmath.py --model 'hybrid' --alphas_mean 0.0  --use_pretrained 'exps/deepmind_math/hybrid/res_240k_iters/ckpt.pt'  
