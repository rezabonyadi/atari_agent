#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=l65_p1_KangarooDeterministic-v4 
#SBATCH -o re_out_l65_p1_KangarooDeterministic-v4.txt 
#SBATCH -e re_err_l65_p1_KangarooDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py KangarooDeterministic-v4 ../output/DQN/l65_p1/ 1.0 0.65 False False --num_gpus=1 
