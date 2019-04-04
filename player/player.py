import numpy as np
import os
import datetime
from player.player_components.memory import ReplayMemory
from player.player_components.learner import QLearner

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


class Player:
    def __init__(self, game_env, agent_history_length=4, max_mem_size=MEMORY_SIZE, batch_size=BS,
                 learning_rate=LEARNING_RATE, init_epsilon=1.0, minimum_observe_episode=200,
                 update_target_frequency=NETW_UPDATE_FREQ, train_frequency=UPDATE_FREQ,
                 gamma=DISCOUNT_FACTOR, exploratory_memory_size=REPLAY_MEMORY_START_SIZE):
        self.n_actions = game_env.action_space_size
        self.epsilon = init_epsilon
        self.minimum_observe_episodes = minimum_observe_episode
        self.update_target_frequency = update_target_frequency
        self.gamma = gamma
        self.game_env = game_env
        self.train_frequency = train_frequency
        self.exploratory_memory_size = exploratory_memory_size

        self.memory = ReplayMemory(game_env.frame_height, game_env.frame_width,
                                   agent_history_length, max_mem_size, batch_size, game_env.is_graphical)
        self.learner = QLearner(self.n_actions, learning_rate,
                               game_env.frame_height, game_env.frame_width, agent_history_length, gamma=self.gamma)
        self.losses = []

        # self.actuator = ???

    def take_action(self, current_state, episode):
        if np.random.rand() <= self.epsilon or episode < self.minimum_observe_episodes:
            return np.random.randint(0, self.n_actions)
        else:
            current_state = np.expand_dims(current_state, axis=0)
            q_value = self.learner.predict(current_state)

            return np.argmax(q_value[0,:])
        # current_state = np.expand_dims(current_state, axis=0)
        # q_value = self.learner.predict(current_state)  # separate old model to predict
        #
        # return np.argmax(q_value[0, :])

        # current_state = np.expand_dims(current_state, axis=0)
        # q_value = self.learner.predict(current_state)[0]  # separate old model to predict
        # v = q_value - q_value.min()
        # v /= v.sum()
        # v = np.cumsum(v)
        #
        # r = np.random.rand()
        # indx = np.argwhere(v >= r)
        # return indx[0][0]
        #

    def learn(self, no_passed_frames):
        if no_passed_frames % self.train_frequency == 0:
            current_state_batch, actions, rewards, next_state_batch, terminal_flags = self.memory.get_minibatch()
            loss = self.learner.train(current_state_batch, actions, rewards, next_state_batch, terminal_flags)
            self.losses.append(loss)

        if no_passed_frames % self.update_target_frequency == 0:
            self.learner.update_target_network()

    def updates(self, no_passed_frames, episode):
        if no_passed_frames > self.exploratory_memory_size:
            self.update_epsilon(episode)
            self.learn(no_passed_frames)

    def update_epsilon(self, episode):
        self.epsilon -= 0.00001
        self.epsilon = max(self.epsilon, 0.1)
        # print('Epsilon: ', str(self.epsilon))

    def save_player(self, info_to_save):
        d = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        address = ''.join(['./output/results_', d, '/'])
        os.makedirs(address)

        pass

    def load_player(self):
        pass