from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
from utils import HandleResults

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v0'
# GAME_ENV = 'PongDeterministic-v4'
# GAME_ENV = 'Alien-v4'
results_handler = HandleResults(GAME_ENV)

def main_loop(load_folder='', load_model=False):

    if load_folder is not '':
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES, all_settings = results_handler.load_settings(load_folder, load_model)
    else:
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES, all_settings = results_handler.load_default_settings(GAME_ENV)

    results_handler.save_settings(all_settings, player)
    res_dict = {}

    highest_reward = 0
    total_frames = 0.0
    all_rewards = np.zeros(MAX_EPISODES)
    time = datetime.datetime.now()
    prev_time = time

    for episode in range(MAX_EPISODES):
        terminal_life_lost = game_env.reset()
        episode_reward = 0
        episode_seq = 0
        for frame_number in range(MAX_EPISODE_LENGTH):
            # Get state, make action, get next state (rewards, terminal, ...), record the experience, train if necessary
            current_state = game_env.get_current_state()
            action = player.take_action(current_state, episode)
            processed_new_frame, reward, terminal, terminal_life_lost, _ = game_env.step(action)

            player.memory.add_experience(action, processed_new_frame, reward, terminal_life_lost, episode_seq)
            episode_reward += reward
            player.updates(total_frames, episode)
            episode_seq += 1

            if terminal_life_lost:
                episode_seq = 0

            # game_env.env.render()
            total_frames += 1

            if terminal:
                terminal = False
                break


        all_rewards[episode] = episode_reward
        # all_rewards.append(episode_reward)

        if episode_reward>highest_reward:
            highest_reward = episode_reward

        if episode % 10==0:
            time_passed = (datetime.datetime.now() - time).total_seconds()

            res_dict['time'] = str(datetime.datetime.now() - time)
            res_dict['highest_reward'] = highest_reward
            res_dict['episode'] = episode
            res_dict['mean_rewards'] = np.mean(all_rewards[episode-10:episode])
            res_dict['total_frames'] = total_frames
            res_dict['epsilon'] = player.epsilon
            res_dict['mean_loss'] = np.mean(player.losses[-100:])
            res_dict['memory_vol'] = player.memory.count
            res_dict['fps'] = total_frames / time_passed

            results_handler.save_res(res_dict)



main_loop()
