#!/bin/bash

#SBATCH --job-name=l_dLN_hybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=96:00:00               # time limit: 1 hour
#SBATCH --output=outputs/learnable_depthLN_hybrid_05_005_linear-head-300ep_lr1e-3_warmup30k.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 300 --batch-size 1024 --torchvision-inception-crop  --name learnable_depthLN_hybrid_1_linear-head-300ep_lr1e-3_warmup30k --arch learnable_depthLN_hybrid --lr 0.001 --warmup 30000 


