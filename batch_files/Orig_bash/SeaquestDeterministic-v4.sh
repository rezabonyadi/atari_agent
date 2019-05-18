#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH -n 2 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=Orig_SeaquestDeterministic-v4 
#SBATCH -o re_out_Orig_SeaquestDeterministic-v4.txt 
#SBATCH -e re_err_Orig_SeaquestDeterministic-v4.txt 
module load anaconda/3.6 
source activate /opt/ohpc/pub/apps/atari 
srun -n2 python ../learn_play_runtime.py SeaquestDeterministic-v4 ../output/DDDQN/Orig/ 0.0 -1.0 False True --num_gpus=1 
