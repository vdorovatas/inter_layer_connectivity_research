#!/bin/bash

#SBATCH --job-name=LSN_eval_0shot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/eval/lambada_RUN3_LSN_skipattn_240k_iters_1.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_D33_268 #EUHPC_A04_051
#SBATCH --qos=boost_qos_dbg

source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

python -u eval_lambada.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --use_pretrained 'exps/owt2/long_short/RUN3_long_short_skipattn_240k_iters_1/ckpt.pt' #'exps/owt2/long_short/RUN2_long_short_skipattn_240k_iters_1/ckpt.pt' #--n_embd 1280 --n_head 20 --batch_size 96 
