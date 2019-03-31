from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import cv2

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
MAX_FRAMES = 30000000            # Total number of frames the agent sees
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

MAX_EPISODES = 1000000
AGENT_HISTORY_LENGTH = 4

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v0'
# GAME_ENV = 'PongDeterministic-v4'


def main_loop():
    frame_height = 84
    frame_width = 84

    game_env = Atari(GAME_ENV, frame_height, frame_width, agent_history_length=AGENT_HISTORY_LENGTH,
                     no_op_steps=10)

    player = Player(game_env, AGENT_HISTORY_LENGTH, batch_size=BS,
                    mem_size=MEMORY_SIZE, hidden=1024, learning_rate=0.00001, minimum_observe_episode=20)

    highest_reward = 0
    total_frames = 0
    all_rewards = []
    time = datetime.datetime.now()

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
            print(datetime.datetime.now() - time, ': ',
                str(episode), ', ', str(highest_reward), ', ', np.mean(all_rewards[-100:]), ', ', str(total_frames),
                  ', ', str(player.epsilon), ', ', np.mean(player.losses[-100:]))


main_loop()