#!/bin/bash

#SBATCH --job-name=hybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=96:00:00               # time limit: 1 hour
#SBATCH --output=outputs/hybrid_400ep_resume.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 400 --batch-size 1024 --torchvision-inception-crop  --name hybrid_400ep --arch hybrid --lr 0.001 --warmup 30000 --resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/hybrid_400ep/checkpoints/checkpoint.pth.tar


