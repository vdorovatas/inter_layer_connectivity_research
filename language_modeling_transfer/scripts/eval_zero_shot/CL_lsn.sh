#!/bin/bash

#SBATCH --job-name=eval_cl_all_lsn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/eval/CL/eval_cl_all_lsn.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=normal #boost_qos_dbg

BASE_DIR="/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments"
OUTPUT_DIR="${BASE_DIR}/outputs/eval/CL"

source ${BASE_DIR}/env/bin/activate
cd ${BASE_DIR}

mkdir -p "${OUTPUT_DIR}"

BENCHMARKS=("boolq" "arc_easy" "piqa" "hellaswag")

CHECKPOINTS=(
    # CL1
    "boolq|CL11_boolq_lsn_6k_iters"
    "hellaswag|CL12_hellaswag_lsn_6k_iters"
    "piqa|CL13_piqa_lsn_2k_iters"
    "arc_easy|CL14_arc_easy_lsn_3k_iters"
    # CL2
    "hellaswag|CL21_hellaswag_lsn_6k_iters"
    "arc_easy|CL22_arc_easy_lsn_3k_iters"
    "boolq|CL23_boolq_lsn_6k_iters"
    "piqa|CL24_piqa_lsn_2k_iters"
    # CL3
    "piqa|CL31_piqa_lsn_2k_iters"
    "boolq|CL32_boolq_lsn_6k_iters"
    "arc_easy|CL33_arc_easy_lsn_3k_iters"
    "hellaswag|CL34_hellaswag_lsn_6k_iters"
)

for CKPT_ENTRY in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r DATASET EXP_NAME <<< "${CKPT_ENTRY}"
    CKPT_PATH="exps/${DATASET}/long_short/${EXP_NAME}/ckpt.pt"

    for BENCHMARK in "${BENCHMARKS[@]}"; do
        OUT_FILE="${OUTPUT_DIR}/${BENCHMARK}_${EXP_NAME}.out"

        if [ -f "${OUT_FILE}" ]; then
            echo "=== SKIP (exists): ${OUT_FILE} ==="
            continue
        fi

        echo "=== ${BENCHMARK} x ${EXP_NAME} ==="
        python -u eval_zero_shot.py \
            --benchmark "${BENCHMARK}" \
            --model 'long_short' \
            --alphas_mean 1.0 --alphas_train --skipattn \
            --use_pretrained "${CKPT_PATH}" \
            > "${OUT_FILE}" 2>&1

    done
done

echo "All evaluations done."
