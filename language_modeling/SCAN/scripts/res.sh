#!/bin/bash

#SBATCH --job-name=SCAN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/SCAN/outputs/res_add_prim_turn_left.log
#SBATCH --account=EUHPC_D33_268 #EUHPC_A04_051
#SBATCH --qos=normal #boost_qos_dbg

source /leonardo_work/EUHPC_A04_051//vdoro/language_modeling/DenseFormer/experiments/env/bin/activate

SCAN_DIR="data/"          # ← point to cloned brendenlake/SCAN
OUT_ROOT="./logs/"
MODEL="residual"

# Small model matched to SCAN scale (~4M params)
N_LAYER=6
N_HEAD=6
N_EMBD=192

COMMON="
  --scan_dir   $SCAN_DIR
  --model      $MODEL
  --n_layer    $N_LAYER
  --n_head     $N_HEAD
  --n_embd     $N_EMBD
  --sequence_length 128
  --batch_size 128
  --acc_steps  1
  --lr         3e-4
  --scheduler  cos
  --warmup_percent 0.05
  --weight_decay 1e-2
  --grad_clip  1.0
  --eval_freq  500
  --seed       2
  --dtype      torch.float32
"

echo "=========================================="
echo " Split 3/3: add primitive (turn_left)"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_turn_left \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/addprim --no_compile
