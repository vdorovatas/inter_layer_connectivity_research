#!/bin/bash

#SBATCH --job-name=SCAN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/SCAN/outputs/lsn_08_add_prim_jump_with_additional_examples.log
#SBATCH --account=EUHPC_D33_268 #EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod

source /leonardo_work/EUHPC_A04_051//vdoro/language_modeling/DenseFormer/experiments/env/bin/activate

SCAN_DIR="data/"          # ← point to cloned brendenlake/SCAN
OUT_ROOT="./logs/"
MODEL="lsn"

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
echo " Split: jump_num2_rep1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num2_rep1 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num2/rep1 --no_compile


echo "=========================================="
echo " Split: jump_num2_rep2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num2_rep2 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num2/rep2 --no_compile

echo "=========================================="
echo " Split: jump_num4_rep1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num4_rep1 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num4/rep1 --no_compile


echo "=========================================="
echo " Split: jump_num4_rep2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num4_rep2 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num4/rep2 --no_compile

echo "=========================================="
echo " Split: jump_num8_rep1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num8_rep1 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num8/rep1 --no_compile


echo "=========================================="
echo " Split: jump_num8_rep2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num8_rep2 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num8/rep2 --no_compile

echo "=========================================="
echo " Split: jump_num16_rep1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num16_rep1 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num16/rep1 --no_compile


echo "=========================================="
echo " Split: jump_num16_rep2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num16_rep2 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num16/rep2 --no_compile

echo "=========================================="
echo " Split: jump_num32_rep1"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num32_rep1 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num32/rep1 --no_compile


echo "=========================================="
echo " Split: jump_num32_rep2"
echo "=========================================="
python -u debug_train.py $COMMON \
    --split    add_prim_jump_num32_rep2 \
    --iterations 20000 \
    --out_dir  $OUT_ROOT/$MODEL/08_add_prim_jump_with_additional_examples/num32/rep2 --no_compile
