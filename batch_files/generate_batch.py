import os

directory = 'my_tes/'
os.mkdir(directory)

game_name = ' GravitarDeterministic-v4'
out_directory_game = ' ../../output/punish_100_1/'
params = ' 100.0 -2.0 False True'
f_name = 'name.sh'
job_name = 'p100_Gr1'
out_file = 're_out_p100_Gr1.txt'
error_file = 're_error_p100_Gr1.txt'

def create_batch_file(f_name, directory, job_name, out_file, error_file, game_name, out_directory_game, params):
    f = open(''.join([directory, f_name]), 'w')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --nodes=1 \n')
    f.write('#SBATCH -n 2 \n')
    f.write('#SBATCH --cpus-per-task=2 \n')
    f.write('#SBATCH --mem=32000 \n')
    f.write('#SBATCH --partition=gpu \n')
    f.write('#SBATCH --gres=gpu:1 \n')
    f.write(''.join(['#SBATCH --job-name=', job_name, ' \n']))
    f.write(''.join(['#SBATCH -o ', out_file, ' \n']))
    f.write(''.join(['#SBATCH -e ', error_file, ' \n']))
    f.write('module load anaconda/3.6 \n')
    f.write('source activate /opt/ohpc/pub/apps/atari \n')
    f.write(''.join(['srun -n2 python ../../learn_play_runtime.py', game_name, out_directory_game, params, ' --num_gpus=1 \n']))
    f.close()

