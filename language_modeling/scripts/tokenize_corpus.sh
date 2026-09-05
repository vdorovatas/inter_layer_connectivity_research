#!/bin/bash

#SBATCH --job-name=tok
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/tokenize_corpus.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=boost_qos_dbg


source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate

cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/data

python -u tokenize_scienceqa.py
python -u tokenize_medqa.py
#python -u tokenize_arc_easy.py
#python tokenize_corpus_optimized.py --dataset_path ./../../../data/pg19/validation --save_path ./datasets/pg19/ --output_filename vval.bin --streaming
