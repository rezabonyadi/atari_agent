#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=p05
#SBATCH -n 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH -o re_out_p05.txt
#SBATCH -e re_error_p05.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/atari

srun -n2 python ./learn_play_constant_settings.py --num_gpus=1
