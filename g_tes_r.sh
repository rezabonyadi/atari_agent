#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=p100_Kan
#SBATCH -n 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH -o re_out_p100_Kan.txt
#SBATCH -e re_error_p100_Kan.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/atari

srun -n2 python ./learn_play_runtime.py KangarooDeterministic-v4 ./output/punish_100/ 100.0 -2.0 False True --num_gpus=1
