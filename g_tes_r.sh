#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=p100_Grv1
#SBATCH -n 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH -o re_out_p100_Grv1.txt
#SBATCH -e re_error_p100_Grv1.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/atari

srun -n2 python ./learn_play_runtime.py GravitarDeterministic-v4 ./output/punish_100_1/ 10.0 -2.0 False True --num_gpus=1

#SBATCH --nodes=1
#SBATCH --job-name=p100_Grv2
#SBATCH -n 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH -o re_out_p100_Grv2.txt
#SBATCH -e re_error_p100_Grv2.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/atari

srun -n2 python ./learn_play_runtime.py GravitarDeterministic-v4 ./output/punish_100_1/ 100.0 -2.0 False True --num_gpus=1
