import os

def create_batch_srun(f_name, directory, job_name, out_file, error_file, game_name, out_directory_game, params):
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
    f.write(''.join(['srun -n2 python ../learn_play_runtime.py ', game_name, out_directory_game, params, ' --num_gpus=1 \n']))
    f.close()


analysis_name = 'e50_p1'
params = ' 1.0 50.0 False True'

directory = ''.join([analysis_name, '_bash/'])

games = [
    'BreakoutDeterministic-v4', 'AsterixDeterministic-v4', 'CarnivalDeterministic-v4', 'MsPacmanDeterministic-v4',
    'UpNDownDeterministic-v4', 'AssaultDeterministic-v4', 'BerzerkDeterministic-v4',
    'QbertDeterministic-v4', 'AmidarDeterministic-v4', 'SpaceInvadersDeterministic-v4',
    'FrostbiteDeterministic-v4', 'KangarooDeterministic-v4', 'GravitarDeterministic-v4',
    'RiverraidDeterministic-v4'
         ]

if not os.path.exists(directory):
    os.mkdir(directory)

out_directory_game = ''.join([' ../output/DDQN/', analysis_name, '/'])

f = open(''.join(['Run_', analysis_name, '.sh']), 'w')

for game_name in games:
# game_name = ' GravitarDeterministic-v4'
    f_name = ''.join([game_name, '.sh'])
    f.write(''.join(['sbatch ', directory, '/', f_name , ' \n']))
    job_name = ''.join([analysis_name, '_', game_name])
    out_file = ''.join(['re_out_', job_name, '.txt'])
    error_file = ''.join(['re_err_', job_name, '.txt'])
    create_batch_srun(f_name, directory, job_name, out_file, error_file, game_name, out_directory_game, params)

f.close()

