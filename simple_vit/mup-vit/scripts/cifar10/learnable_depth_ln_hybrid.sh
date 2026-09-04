#!/bin/bash

#SBATCH --job-name=l_dLN_hybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=12:00:00               # time limit: 1 hour
#SBATCH --output=outputs/cifar10/learnable_depthLN_hybrid_025_005_linear-head-300ep_lr1e-3_warmup30k.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal #boost_qos_dbg #lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun cifar10_main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 15 --batch-size 128 --torchvision-inception-crop  --name learnable_depthLN_hybrid_1_linear-head-300ep_lr1e-3_warmup30k --arch learnable_depthLN_hybrid --lr 0.0005 --warmup 800 --resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/learnable_depthLN_hybrid_linear-head-300ep_lr1e-3_warmup30k/checkpoints/model_best.pth.tar 


