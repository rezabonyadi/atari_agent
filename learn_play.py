from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import utils

MAX_EPISODE_LENGTH = 108000       # Equivalent of 5 minutes of gameplay at 60 frames per second
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode

MAX_EPISODES = 1000000
AGENT_HISTORY_LENGTH = 4
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
BS = 32                          # Batch size
LEARNING_RATE = 0.0001          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
PUNISH = 1.0
INI_EPSILON = 1.0
END_EPSILON = .1
MIN_OBSERVE_EPISODE = 200

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v0'
# GAME_ENV = 'PongDeterministic-v4'
# GAME_ENV = 'Alien-v4'

def main_loop(load_folder=''):

    if load_folder is not '':
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES = utils.load_settings(load_folder)
    else:
        player, game_env, MAX_EPISODE_LENGTH, MAX_EPISODES = utils.load_settings('')

        # game_env = Atari(GAME_ENV, frame_height, frame_width, agent_history_length=AGENT_HISTORY_LENGTH,
        #     no_op_steps=NO_OP_STEPS)
        # player = Player(game_env, AGENT_HISTORY_LENGTH, MEMORY_SIZE, BS,
        #     LEARNING_RATE, INI_EPSILON, END_EPSILON, MIN_OBSERVE_EPISODE,
        #     NETW_UPDATE_FREQ, UPDATE_FREQ, DISCOUNT_FACTOR, REPLAY_MEMORY_START_SIZE, PUNISH)

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

main_loop(load_folder='./')
