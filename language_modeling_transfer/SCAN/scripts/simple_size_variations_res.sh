#!/bin/bash

#SBATCH --job-name=SCAN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/SCAN/outputs/res_simple_size_variations.log
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
echo " Split: simple p1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p1 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p1 --no_compile

echo "=========================================="
echo " Split: simple p2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p2 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p2 --no_compile

echo "=========================================="
echo " Split: simple p4"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p4 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p4 --no_compile

echo "=========================================="
echo " Split: simple p8"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p8 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p8 --no_compile

echo "=========================================="
echo " Split: simple p16"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p16 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p16 --no_compile

echo "=========================================="
echo " Split: simple p32"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple_p32 \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/p32 --no_compile

echo "=========================================="
echo " Split: simple full"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    simple \
    --iterations 10000 \
    --out_dir  $OUT_ROOT/$MODEL/simple/full --no_compile
