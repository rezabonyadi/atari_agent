import numpy as np
import os
import datetime
from player.player_components.memory import ReplayMemory
from player.player_components.learner import QLearner
from keras.models import model_from_json


class Player:
    def __init__(self, game_env, agent_history_length, max_mem_size, batch_size,
                 learning_rate, init_epsilon, end_epsilon, minimum_observe_episode,
                 update_target_frequency, train_frequency,
                 gamma, exploratory_memory_size, punishment):
        self.n_actions = game_env.action_space_size
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.minimum_observe_episodes = minimum_observe_episode
        self.update_target_frequency = update_target_frequency
        self.gamma = gamma
        self.game_env = game_env
        self.train_frequency = train_frequency
        self.exploratory_memory_size = exploratory_memory_size
        self.total_memory_size = max_mem_size
        self.batch_size = batch_size
        self.agent_history_length = agent_history_length
        self.learning_rate = learning_rate
        self.punishment = punishment

        self.memory = ReplayMemory(self.game_env.frame_height, self.game_env.frame_width,
                                   self.agent_history_length, self.total_memory_size,
                                   self.batch_size, self.game_env.is_graphical,
                                   punishment=self.punishment)
        self.learner = QLearner(self.n_actions, self.learning_rate,
                               self.game_env.frame_height, self.game_env.frame_width, self.agent_history_length,
                                gamma=self.gamma)
        self.losses = []
        self.q_values = []

        # self.actuator = ???

    def take_action(self, current_state, episode):
        if np.random.rand() <= self.epsilon or episode < self.minimum_observe_episodes:
            action = np.random.randint(0, self.n_actions)
        else:
            current_state = np.expand_dims(current_state, axis=0)
            q_values = self.learner.predict(current_state)

            action, q_value = self.learner.action_selection_policy(q_values)
            self.q_values.append(q_value)
        return action

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
        self.epsilon = max(self.epsilon, self.end_epsilon)
        # print('Epsilon: ', str(self.epsilon))

    def save_player_learner(self, folder):
        model_json = self.learner.main_learner.model.to_json(indent=4)
        with open(''.join([folder, 'model_structure.jsn']), "w") as json_file:
            json_file.write(model_json)

        self.learner.main_learner.model.save_weights(''.join([folder, 'model_weights.wts']))

    def load_player_learner(self, folder):
        json_file = open(''.join([folder, 'model_structure.jsn']), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(''.join([folder,'model_weights.wts']))

        self.learner.main_learner.model = loaded_model
