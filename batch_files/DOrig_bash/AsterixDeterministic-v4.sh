#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=DOrig_AsterixDeterministic-v4 
#SBATCH -o re_out_DOrig_AsterixDeterministic-v4.txt 
#SBATCH -e re_err_DOrig_AsterixDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py AsterixDeterministic-v4 ../output/DQN/DOrig/ 0.0 -1.0 False False --num_gpus=1 
