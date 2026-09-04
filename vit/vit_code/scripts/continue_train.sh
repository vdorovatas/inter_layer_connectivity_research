#!/bin/bash

#SBATCH --job-name=vit_from_ckpt
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
##SBATCH --time=4-00:00:00           
#SBATCH --time=24:00:00             
#SBATCH --output=outputs/train_continue.out    # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
##SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u main.py --lr=0.00003 --batch_size=512 --weight_decay=0.3 --mode='train' --model_type long --data_path /leonardo/prod/data/ai/imagenet/ilsvrc2012 --epochs 900 --resume True 

