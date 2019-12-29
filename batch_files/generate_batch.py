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

games = [
    'BreakoutDeterministic-v4', 'AsterixDeterministic-v4', 'CarnivalDeterministic-v4',
    'UpNDownDeterministic-v4', 'AssaultDeterministic-v4', 'BerzerkDeterministic-v4',
    'QbertDeterministic-v4', 'AmidarDeterministic-v4', 'SpaceInvadersDeterministic-v4',
    'FrostbiteDeterministic-v4', 'KangarooDeterministic-v4', 'GravitarDeterministic-v4',
    'RiverraidDeterministic-v4', 'HeroDeterministic-v4', 'JamesbondDeterministic-v4',
    'SeaquestDeterministic-v4', 'BankHeistDeterministic-v4', 'AirRaidDeterministic-v4',
    'SolarisDeterministic-v4', 'PrivateEyeDeterministic-v4', 'WizardOfWorDeterministic-v4',
    'ZaxxonDeterministic-v4', 'FreewayDeterministic-v4', 'VentureDeterministic-v4',
    'RoadRunnerDeterministic-v4', 'TutankhamDeterministic-v4', 'KungFuMasterDeterministic-v4',
    'KrullDeterministic-v4', 'CentipedeDeterministic-v4', 'MsPacmanDeterministic-v4',
    'AlienDeterministic-v4'
]

# games=['RoadRunnerDeterministic-v4', 'TutankhamDeterministic-v4', 'KungFuMasterDeterministic-v4',
#        'KrullDeterministic-v4']

# games = ['BeamRiderDeterministic-v4']
analysis_name = 'l65_p1'
params = ' 1.0 0.65 False True'

# analysis_name = 'original'
# params = ' 0.0 -50.0 False True'
# punish exponent Linear_exploration Double

out_directory_game = ''.join([' ../output/DDQN/lambda/', analysis_name, '/'])


directory = ''.join([analysis_name, '_bash/'])

if not os.path.exists(directory):
    os.mkdir(directory)

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

