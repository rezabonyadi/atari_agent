#!/bin/bash
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/atari

#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --job-name=p100_Gr1
#SBATCH -o re_out_p100_Gr1.txt
#SBATCH -e re_error_p100_Gr1.txt

srun -n2 python ./learn_play_runtime.py GravitarDeterministic-v4 ./output/punish_100_1/ 100.0 -2.0 False True --num_gpus=1 &

#SBATCH --job-name=p100_Gr2
#SBATCH -o re_out_p100_Gr2.txt
#SBATCH -e re_error_p100_Gr2.txt

srun -n2 python ./learn_play_runtime.py GravitarDeterministic-v4 ./output/punish_100_1/ 100.0 -2.0 False True --num_gpus=1
