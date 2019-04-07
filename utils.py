from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import os
import json
import csv

class HandleResults:

    folder_to_use = ''
    settings_file_name = 'settings.jsn'
    time = datetime.datetime.now()

    def __init__(self, GAME_ENV):
        # GAME_ENV = settings['GAME_ENV']
        d = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        self.folder_to_use = ''.join(['./output/', GAME_ENV, '/results_', d, '/'])
        os.makedirs(self.folder_to_use)
        self.results_file_name = ''.join([self.folder_to_use, 'results.csv'])


    def save_settings(self, settings, player):
        settings_dict = settings

        with open(''.join([self.folder_to_use, 'settings.jsn']), 'wt') as outfile:
            json.dump(settings_dict, outfile, indent=4)

        player.save_player_learner(self.folder_to_use)

    def load_default_settings(self, GAME_ENV):

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

    def load_settings(self, folder, load_model):
        settings_dict = {}
        with open(''.join([folder, self.settings_file_name]), 'rt') as json_file:
            settings_dict = json.load(json_file)

        game_env = Atari(settings_dict['GAME_ENV'], settings_dict['frame_height'], settings_dict['frame_width'],
                         agent_history_length=settings_dict['AGENT_HISTORY_LENGTH'], no_op_steps=settings_dict['NO_OP_STEPS'])

        player = Player(game_env, settings_dict['AGENT_HISTORY_LENGTH'], settings_dict['MEMORY_SIZE'], settings_dict['BS'],
                        settings_dict['LEARNING_RATE'], settings_dict['INI_EPSILON'], settings_dict['END_EPSILON'],
                        settings_dict['MIN_OBSERVE_EPISODE'], settings_dict['NETW_UPDATE_FREQ'], settings_dict['UPDATE_FREQ'],
                        settings_dict['DISCOUNT_FACTOR'], settings_dict['REPLAY_MEMORY_START_SIZE'], settings_dict['PUNISH'])

        if load_model:
            player.load_player_learner(folder)

        return player, game_env, settings_dict['MAX_EPISODE_LENGTH'], settings_dict['MAX_EPISODES'], settings_dict

    def save_res(self, res_dict):
        if not os.path.isfile(self.results_file_name):
            with open(self.results_file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                headings = list(res_dict.keys())
                writer.writerow(headings)
            file.close()

        with open(self.results_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            values = list(res_dict.values())
            writer.writerow(values)
        file.close()

        print(res_dict)

