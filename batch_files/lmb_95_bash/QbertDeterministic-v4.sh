#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=lmb_95_QbertDeterministic-v4 
#SBATCH -o re_out_lmb_95_QbertDeterministic-v4.txt 
#SBATCH -e re_err_lmb_95_QbertDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py QbertDeterministic-v4 ../output/parameters/DDQN/lmb_95/ 0.0 0.95 False True --num_gpus=1 
