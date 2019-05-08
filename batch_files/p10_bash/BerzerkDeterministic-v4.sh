#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=p10_BerzerkDeterministic-v4 
#SBATCH -o re_out_p10_BerzerkDeterministic-v4.txt 
#SBATCH -e re_err_p10_BerzerkDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py BerzerkDeterministic-v4 ../output/DQN/p10/ 10.0 0.0 False False --num_gpus=1 
