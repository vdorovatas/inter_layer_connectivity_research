#!/bin/bash
#SBATCH --job-name=eval_cl_all_lsn_ood
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/eval/OOD_CL/eval_cl_all_lsn_ood.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=normal

BASE_DIR="/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments"
CKPT_BASE="/leonardo_scratch/large/userexternal/edorovat/gpt2/exps"
OUTPUT_DIR="/leonardo_scratch/large/userexternal/edorovat/gpt2/outputs/eval/OOD_CL"

source ${BASE_DIR}/env/bin/activate
cd ${BASE_DIR}
mkdir -p "${OUTPUT_DIR}"

BENCHMARKS=("medqa" "biology" "chemistry")

CHECKPOINTS=(
    # OOD CL1
    "medqa|OOD_CL11_medqa_lsn"
    "biology|OOD_CL12_biology_lsn"
    "chemistry|OOD_CL13_chemistry_lsn"
    # OOD CL2
    "biology|OOD_CL21_biology_lsn"
    "chemistry|OOD_CL22_chemistry_lsn"
    "medqa|OOD_CL23_medqa_lsn"
    # OOD CL3
    "chemistry|OOD_CL31_chemistry_lsn"
    "medqa|OOD_CL32_medqa_lsn"
    "biology|OOD_CL33_biology_lsn"
)

for CKPT_ENTRY in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r DATASET EXP_NAME <<< "${CKPT_ENTRY}"
    CKPT_PATH="${CKPT_BASE}/${DATASET}/long_short/${EXP_NAME}/ckpt.pt"

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
