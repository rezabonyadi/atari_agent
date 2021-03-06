#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=e50_p1_KungFuMasterDeterministic-v4 
#SBATCH -o re_out_e50_p1_KungFuMasterDeterministic-v4.txt 
#SBATCH -e re_err_e50_p1_KungFuMasterDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py KungFuMasterDeterministic-v4 ../output/DDDQN/e50_p1/ 1.0 50.0 False True --num_gpus=1 
