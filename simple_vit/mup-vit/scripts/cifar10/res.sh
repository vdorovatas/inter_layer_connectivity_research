#!/bin/bash

#SBATCH --job-name=res
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=96:00:00               # time limit: 1 hour
#SBATCH --output=outputs/cifar10/res_linear-head-300ep_lr1e-3_warmup30k.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun cifar10_main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 10 --batch-size 128 --torchvision-inception-crop  --name res_linear-head-300ep_lr1e-3_warmup30k --arch res --lr 0.0005 --warmup 800 --resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/res_linear-head-300ep_lr1e-3_warmup30k/checkpoints/model_best.pth.tar


