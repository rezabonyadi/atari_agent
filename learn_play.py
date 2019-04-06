from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import utils

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v0'
# GAME_ENV = 'PongDeterministic-v4'
# GAME_ENV = 'Alien-v4'


def main_loop(load_folder=''):

    if load_folder is not '':
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES, all_settings = utils.load_settings(load_folder)
    else:
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES, all_settings = utils.load_default(GAME_ENV)

    utils.save_settings(all_settings, player)

    highest_reward = 0
    total_frames = 0.0
    all_rewards = []
    time = datetime.datetime.now()
    prev_time = time

    for episode in range(MAX_EPISODES):
        terminal_life_lost = game_env.reset()
        episode_reward = 0
        for frame_number in range(MAX_EPISODE_LENGTH):
            # Get state, make action, get next state (rewards, terminal, ...), record the experience, train if necessary
            current_state = game_env.get_current_state()
            action = player.take_action(current_state, episode)
            processed_new_frame, reward, terminal, terminal_life_lost, _ = game_env.step(action)

            player.memory.add_experience(action, processed_new_frame, reward, terminal_life_lost)
            episode_reward += reward
            player.updates(total_frames, episode)

            # game_env.env.render()

            if terminal:
                terminal = False
                break

            total_frames += 1

        all_rewards.append(episode_reward)

        if episode_reward>highest_reward:
            highest_reward = episode_reward

        if episode % 10==0:
            time_passed = (datetime.datetime.now() - time).total_seconds()
            print(datetime.datetime.now() - time, ': episod is ', str(episode),
                  ', highest reward is ', str(highest_reward),
                  ', average reward is ', str(np.mean(all_rewards[-100:])),
                  ', total frames is ', str(total_frames),
                  ', epsilon is ', str(player.epsilon),
                  ' , loss is ', str(np.mean(player.losses[-100:])),
                  ' , fps is ', str(total_frames/time_passed))

main_loop()
