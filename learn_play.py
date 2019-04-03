from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import cv2

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode

MAX_EPISODES = 1000000
AGENT_HISTORY_LENGTH = 4

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v0'
# GAME_ENV = 'PongDeterministic-v4'


def main_loop():
    frame_height = 84
    frame_width = 84

    game_env = Atari(GAME_ENV, frame_height, frame_width, agent_history_length=AGENT_HISTORY_LENGTH,
                     no_op_steps=NO_OP_STEPS)

    player = Player(game_env, AGENT_HISTORY_LENGTH, learning_rate=0.00001, minimum_observe_episode=20)

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
            prev_time = datetime.datetime.now()


main_loop()