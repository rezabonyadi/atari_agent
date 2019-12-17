from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
from utils import HandleResults
import sys


'''
Set the main settings in the default_settings.jsn
* PUNISH controls the positive punishment, to be set to non-zero to work
* REWARD_EXTRAPOLATION_EXPONENT controls the exponent for backfilling. Set to -1.0 to turn this off (i.e., use the 
actual reward values only)

Some settings are in the memory class:
* START_EPISODE: This will be the start episode of the linear increase. 
* END_EPISODE: This will be the end episode of the linear increase.
* START_EXPONENT: The start exponent (1.0)
* END_EXPONENT: Final exponent value (10.0)
* IGNORE_EXPONENT_EPISODE: At what episode ignore using the exponent (just punishment if on)
 '''


def run_episode(max_episode_length, episode, game_env, player, total_frames, evaluation=False):
    terminal_life_lost = game_env.reset()
    episode_reward = 0
    life_seq = 0
    frame_number = 0
    gif_frames = []
    while True:
        # Get state, make action, get next state (rewards, terminal, ...), record the experience, train if necessary
        current_state = game_env.get_current_state()
        action = player.take_action(current_state, total_frames, evaluation)
        processed_new_frame, reward, terminal, terminal_life_lost, original_frame = game_env.step(action)

        if not evaluation:
            if frame_number >= max_episode_length:
                terminal = True
                terminal_life_lost = True

            player.updates(total_frames, episode, action, processed_new_frame, reward, terminal_life_lost, life_seq)

        episode_reward += reward
        life_seq += 1

        if terminal_life_lost:
            life_seq = 0

        # game_env.env.render()
        total_frames += 1
        frame_number += 1

        if terminal:
            break

    return episode_reward, total_frames


def learn_by_game(results_handler, settings_dict):

    # if load_folder is not '':
    #     player, game_env, max_episode_length, max_number_of_episodes, all_settings = \
    #         results_handler.load_settings_folder(load_folder, load_model)
    # else:
    #     player, game_env, max_episode_length, max_number_of_episodes, all_settings = \
    #         results_handler.load_settings_default(GAME_ENV)

    player, game_env, max_episode_length, max_number_of_episodes, all_settings = \
        results_handler.load_settings_dictionary(GAME_ENV, settings_dict)

    for k, v in all_settings.items():
        print(k, ': ', v)

    print('****************************')

    results_handler.save_settings(all_settings)
    res_dict = {}

    highest_reward = 0
    total_frames = 0.0
    prev_frames = 0.0
    all_rewards = []
    time = datetime.datetime.now()
    prev_time = time
    best_evaluation = -100000

    for episode in range(max_number_of_episodes):
        episode_reward, total_frames = run_episode(max_episode_length, episode, game_env, player, total_frames)

        # all_rewards[episode] = episode_reward
        all_rewards.append(episode_reward)

        if episode_reward>highest_reward:
            highest_reward = episode_reward

        if episode % 10 == 0:
            # evaluation_reward = 0
            # for i in range(100):
            #     r, _ = run_episode(max_episode_length, episode, game_env, player, 0, evaluation=True)
            #     evaluation_reward += r
            # evaluation_reward /= 100
            #
            # if evaluation_reward > best_evaluation:
            #     best_evaluation = evaluation_reward
                # print('Best eval: ', str(best_evaluation))

            now = datetime.datetime.now()
            res_dict['time'] = str(now - time)
            res_dict['episode'] = episode
            res_dict['total_frames'] = total_frames
            res_dict['epsilon'] = format(player.epsilon, '.3f')
            res_dict['highest_reward'] = highest_reward
            # res_dict['best_eval'] = best_evaluation
            res_dict['mean_rewards'] = np.mean(all_rewards[-10:])
            res_dict['mean_loss'] = format(np.mean(player.losses[-10:]), '.5f')
            # res_dict['memory_vol'] = player.memory.count
            # res_dict['fps'] = (total_frames - prev_frames) / ((now - prev_time).total_seconds())
            # res_dict['sparsity'] = np.mean(player.memory.sparsity_lengths[-10:])
            # res_dict['estimating_reward'] = player.memory.use_estimated_reward
            # res_dict['reward_exponent'] = player.memory.reward_extrapolation_exponent
            res_dict['terminal_freq'] = format(np.mean(player.memory.terminal_lengths[-100:]), '.3f')
            res_dict['sparsity_freq'] = format(np.mean(player.memory.sparsity_lengths[-100:]), '.3f')
            res_dict['reward_values'] = format(np.mean(player.memory.rewards_values[-100:]), '.3f')
            res_dict['punishment'] = format(player.punishment, '.3f')

            # res_dict['best_evaluation'] = best_evaluation

            results_handler.save_res(res_dict)

            prev_time = now
            prev_frames = total_frames

    results_handler.save_settings(all_settings, player)

# OUT_FOLDER = './output/punish_100/'

# games = [
#     'BreakoutDeterministic-v4', 'AsterixDeterministic-v4', 'CarnivalDeterministic-v4', 'MsPacmanDeterministic-v4',
#     'UpNDownDeterministic-v4', 'AssaultDeterministic-v4', 'BerzerkDeterministic-v4',
#     'QbertDeterministic-v4', 'AmidarDeterministic-v4', 'SpaceInvadersDeterministic-v4'
#          ]

# games = ['FrostbiteDeterministic-v4', 'KangarooDeterministic-v4', 'GravitarDeterministic-v4', 'TutankhamDeterministic-v4',
# 'RiverraidDeterministic-v4']

settings_dict = dict()
MAX_EPISODE_LENGTH= 18000
NO_OP_STEPS= 10
MAX_EPISODES= 2000
AGENT_HISTORY_LENGTH= 4
UPDATE_FREQ= 4
NETW_UPDATE_FREQ= 10000
REPLAY_MEMORY_START_SIZE = 50000
DISCOUNT_FACTOR= 0.99
MEMORY_SIZE = 1000000
BS= 32
LEARNING_RATE= 0.0001
PUNISH= 100.0
INI_EPSILON= 1.0
END_EPSILON= 0.1
MIN_OBSERVE_EPISODE= 200
GAME_ENV= "BreakoutDeterministic-v4"
REWARD_EXTRAPOLATION_EXPONENT = 2.0
frame_height = 84
frame_width = 84
LINEAR_EXPLORATION_EXPONENT = False
USE_DOUBLE_MODEL = True

GAME_ENV = sys.argv[1]
OUT_FOLDER = sys.argv[2]
PUNISH = float(sys.argv[3])
REWARD_EXTRAPOLATION_EXPONENT = float(sys.argv[4])
LINEAR_EXPLORATION_EXPONENT = (sys.argv[5] == 'True')
USE_DOUBLE_MODEL = (sys.argv[6] == 'True')

settings_dict['GAME_ENV'] = GAME_ENV
settings_dict['AGENT_HISTORY_LENGTH'] = AGENT_HISTORY_LENGTH
settings_dict['MEMORY_SIZE'] = MEMORY_SIZE
settings_dict['BS'] = BS
settings_dict['LEARNING_RATE'] = LEARNING_RATE
settings_dict['INI_EPSILON'] = INI_EPSILON
settings_dict['END_EPSILON'] = END_EPSILON
settings_dict['MIN_OBSERVE_EPISODE'] = MIN_OBSERVE_EPISODE
settings_dict['NETW_UPDATE_FREQ'] = NETW_UPDATE_FREQ
settings_dict['UPDATE_FREQ'] = UPDATE_FREQ
settings_dict['DISCOUNT_FACTOR'] = DISCOUNT_FACTOR
settings_dict['REPLAY_MEMORY_START_SIZE'] = REPLAY_MEMORY_START_SIZE
settings_dict['frame_height'] = frame_height
settings_dict['frame_width'] = frame_width
settings_dict['NO_OP_STEPS'] = NO_OP_STEPS
settings_dict['MAX_EPISODE_LENGTH'] = MAX_EPISODE_LENGTH
settings_dict['MAX_EPISODES'] = MAX_EPISODES
settings_dict['PUNISH'] = PUNISH
settings_dict['REWARD_EXTRAPOLATION_EXPONENT'] = REWARD_EXTRAPOLATION_EXPONENT
settings_dict['LINEAR_EXPLORATION_EXPONENT'] = LINEAR_EXPLORATION_EXPONENT
settings_dict['USE_DOUBLE_MODEL'] = USE_DOUBLE_MODEL


handler = HandleResults(GAME_ENV, OUT_FOLDER)
learn_by_game(handler, settings_dict)

# Run like: python ./learn_play_runtime.py BreakoutDeterministic-v4 ./output/punish_1_exp_2/ 1.0 2.0 False True
