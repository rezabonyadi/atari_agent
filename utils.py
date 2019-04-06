from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import os
import json

folder_to_use = ''
settings_file_name = 'settings.jsn'


def save_settings(settings, player):
    GAME_ENV = settings['GAME_ENV']
    d = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    folder_to_use = ''.join(['./output/', GAME_ENV, '/results_', d, '/'])
    os.makedirs(folder_to_use)

    settings_dict = settings

    with open(''.join([folder_to_use, 'settings.jsn']), 'wt') as outfile:
        json.dump(settings_dict, outfile)

    player.save_player_learner(''.join([folder_to_use, 'main_learner_init.mdl']))


def load_default(GAME_ENV):
    settings_dict = {}
    file_name = './default_settings.jsn'  # default_settings.jsn i in the root
    with open(file_name, 'rt') as json_file:
        settings_dict = json.load(json_file)
    settings_dict['GAME_ENV'] = GAME_ENV
    game_env = Atari(settings_dict['GAME_ENV'], settings_dict['frame_height'], settings_dict['frame_width'],
                     agent_history_length=settings_dict['AGENT_HISTORY_LENGTH'],
                     no_op_steps=settings_dict['NO_OP_STEPS'])

    player = Player(game_env, settings_dict['AGENT_HISTORY_LENGTH'], settings_dict['MEMORY_SIZE'], settings_dict['BS'],
                    settings_dict['LEARNING_RATE'], settings_dict['INI_EPSILON'], settings_dict['END_EPSILON'],
                    settings_dict['MIN_OBSERVE_EPISODE'], settings_dict['NETW_UPDATE_FREQ'],
                    settings_dict['UPDATE_FREQ'],
                    settings_dict['DISCOUNT_FACTOR'], settings_dict['REPLAY_MEMORY_START_SIZE'],
                    settings_dict['PUNISH'])
    return player, game_env, settings_dict['MAX_EPISODE_LENGTH'], settings_dict['MAX_EPISODES'], settings_dict

def load_settings(folder):
    settings_dict = {}
    with open(''.join([folder, settings_file_name]), 'rt') as json_file:
        settings_dict = json.load(json_file)

    game_env = Atari(settings_dict['GAME_ENV'], settings_dict['frame_height'], settings_dict['frame_width'],
                     agent_history_length=settings_dict['AGENT_HISTORY_LENGTH'], no_op_steps=settings_dict['NO_OP_STEPS'])

    player = Player(game_env, settings_dict['AGENT_HISTORY_LENGTH'], settings_dict['MEMORY_SIZE'], settings_dict['BS'],
                    settings_dict['LEARNING_RATE'], settings_dict['INI_EPSILON'], settings_dict['END_EPSILON'],
                    settings_dict['MIN_OBSERVE_EPISODE'], settings_dict['NETW_UPDATE_FREQ'], settings_dict['UPDATE_FREQ'],
                    settings_dict['DISCOUNT_FACTOR'], settings_dict['REPLAY_MEMORY_START_SIZE'], settings_dict['PUNISH'])

    player.load_player_learner(''.join([folder, 'main_learner.mdl']))

    return player, game_env, settings_dict['MAX_EPISODE_LENGTH'], settings_dict['MAX_EPISODES'], settings_dict

def save_res():
    pass