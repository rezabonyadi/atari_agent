from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import os
import json

folder_to_use = ''
settings_file_name = 'settings.jsn'

def save_settings(MAX_EPISODE_LENGTH, NO_OP_STEPS, MAX_EPISODES, AGENT_HISTORY_LENGTH, UPDATE_FREQ, NETW_UPDATE_FREQ, \
           REPLAY_MEMORY_START_SIZE, DISCOUNT_FACTOR, MEMORY_SIZE, BS, LEARNING_RATE, PUNISH, INI_EPSILON, END_EPSILON, \
           MIN_OBSERVE_EPISODE, GAME_ENV, player):
    d = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    folder_to_use = ''.join(['./output/', GAME_ENV, '/results_', d, '/'])
    os.makedirs(folder_to_use)
    player.save_player_learner(''.join([folder_to_use, 'main_learner_init.mdl']))

    pass


def load_settings(folder):
    settings_dict = {}
    if folder is not '':
        with open(''.join([folder, settings_file_name]), 'rt') as json_file:
            settings_dict = json.load(json_file)
    else:
        settings_dict['MAX_EPISODE_LENGTH'] = 108000  # Equivalent of 5 minutes of gameplay at 60 frames per second
        settings_dict['NO_OP_STEPS'] = 10  # Number of 'NOOP' or 'FIRE' actions at the beginning of an evaluation episode
        settings_dict['MAX_EPISODES'] = 1000000
        settings_dict['AGENT_HISTORY_LENGTH'] = 4
        settings_dict['UPDATE_FREQ'] = 4  # Every four actions a gradient descend step is performed
        settings_dict['NETW_UPDATE_FREQ'] = 10000  # Number of chosen actions between updating the target network.
        settings_dict['REPLAY_MEMORY_START_SIZE'] = 50000  # Number of completely random actions, before the agent starts learning
        settings_dict['DISCOUNT_FACTOR'] = 0.99  # gamma in the Bellman equation
        settings_dict['MEMORY_SIZE'] = 1000000  # Number of transitions stored in the replay memory
        settings_dict['BS'] = 32  # Batch size
        settings_dict['LEARNING_RATE'] = 0.0001  # Set to 0.00025 in Pong for quicker results.
        settings_dict['PUNISH'] = 1.0
        settings_dict['INI_EPSILON'] = 1.0
        settings_dict['END_EPSILON'] = .1
        settings_dict['MIN_OBSERVE_EPISODE'] = 200

        settings_dict['GAME_ENV'] = 'BreakoutDeterministic-v4'
        settings_dict['frame_height'] = 84
        settings_dict['frame_width'] = 84

        with open('default_settings.jsn', 'wt') as outfile:
            json.dump(settings_dict, outfile)

    game_env = Atari(settings_dict['GAME_ENV'], settings_dict['frame_height'], settings_dict['frame_width'],
                     agent_history_length=settings_dict['AGENT_HISTORY_LENGTH'], no_op_steps=settings_dict['NO_OP_STEPS'])

    player = Player(game_env, settings_dict['AGENT_HISTORY_LENGTH'], settings_dict['MEMORY_SIZE'], settings_dict['BS'],
                    settings_dict['LEARNING_RATE'], settings_dict['INI_EPSILON'], settings_dict['END_EPSILON'],
                    settings_dict['MIN_OBSERVE_EPISODE'], settings_dict['NETW_UPDATE_FREQ'], settings_dict['UPDATE_FREQ'],
                    settings_dict['DISCOUNT_FACTOR'], settings_dict['REPLAY_MEMORY_START_SIZE'], settings_dict['PUNISH'])

    if folder is not '':
        player.load_player_learner(''.join([folder, 'main_learner.mdl']))

    return player, game_env, settings_dict.MAX_EPISODE_LENGTH, settings_dict.MAX_EPISODES

def save_res():
    pass