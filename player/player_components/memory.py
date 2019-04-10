import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.initializers import VarianceScaling
from keras import layers, callbacks
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K


class ReplayMemory:

    def __init__(self, frame_height, frame_width, agent_history_length=4, size=1000000, batch_size=32,
                 is_graphical=True, use_spotlight=False, use_estimated_reward=True, punishment=0.0,
                 reward_extrapolation_exponent=10.0):

        self.use_estimated_reward = use_estimated_reward
        self.use_spotlight = use_spotlight
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        self.is_graphical = is_graphical
        self.punishment = punishment
        self.reward_extrapolation_exponent = reward_extrapolation_exponent

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        if is_graphical:
            self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        else:
            self.frames = np.empty((self.size, self.frame_height), dtype=np.float16)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.frame_number_in_epison = np.empty(self.size, dtype=np.int)
        self.sparsity_lengths = []
        self.min_reward = 1000000.0
        self.max_reward = -1000000.0


        if is_graphical:
            self.minibatch_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
            self.minibatch_new_states = np.empty((self.batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.uint8)
        else:
            self.minibatch_states = np.empty((self.batch_size, self.agent_history_length,
                                              self.frame_height), dtype=np.float16)
            self.minibatch_new_states = np.empty((self.batch_size, self.agent_history_length,
                                                  self.frame_height), dtype=np.float16)

        self.minibatch_indices = np.empty(self.batch_size, dtype=np.int32)
        input_shape = (frame_height, frame_width, 1)

        self.spotlight = SpotlightAttention(input_shape)

    def add_experience(self, action, frame, reward, terminal, frame_in_seq):
        self.min_reward = np.min((self.min_reward, reward))
        self.max_reward = np.max((self.max_reward, reward))

        if self.use_spotlight:
            f = np.expand_dims(frame, axis=0)
            f = np.expand_dims(f, axis=3)

            seen_before = self.spotlight.seen_before(f)
            self.spotlight.spotlight_train(f)
        else:
            seen_before = False

        if not seen_before:
            if terminal:
                reward = -self.punishment

            self.actions[self.current] = action
            self.frames[self.current, ...] = frame
            self.rewards[self.current] = reward
            self.terminal_flags[self.current] = terminal
            self.frame_number_in_epison[self.current] = frame_in_seq

            if self.use_estimated_reward:
                self.revise_rewards(reward)

            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.size

    def revise_rewards(self, current_reward):
        if current_reward != 0:
            prev_reward_indx = self.current - 1

            while (self.frame_number_in_epison[prev_reward_indx] > 0) and (self.rewards[prev_reward_indx] == 0.0) \
                    and (prev_reward_indx > 0):
                prev_reward_indx -= 1

            start_indx = prev_reward_indx + 1
            end_indx = self.current
            sparsity_length = end_indx - start_indx
            self.sparsity_lengths.append(sparsity_length)

            for i in range(start_indx, end_indx):
                self.rewards[i] = self.get_estimated_reward(current_reward, sparsity_length, i - start_indx)

    def get_estimated_reward(self, recent_reward, sparsity_length, current_index):

        return recent_reward*np.power(current_index/sparsity_length, self.reward_extrapolation_exponent)

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # if self.terminal_flags[index - self.agent_history_length:index].any():
                #     continue
                if self.frame_number_in_epison[index] - self.frame_number_in_epison[index - self.agent_history_length] != self.agent_history_length:
                    continue
                break
            self.minibatch_indices[i] = index

    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.minibatch_indices):
            self.minibatch_states[i] = self._get_state(idx - 1)
            self.minibatch_new_states[i] = self._get_state(idx)

        return np.transpose(self.minibatch_states, axes=(0, 2, 3, 1)), self.actions[self.minibatch_indices], \
               self.rewards[self.minibatch_indices], np.transpose(self.minibatch_new_states, axes=(0, 2, 3, 1)), \
               self.terminal_flags[self.minibatch_indices]


class SpotlightAttention:

    def __init__(self, input_shape):
        self.embedding_dimension = 10
        self.spotlight_model = self.build_spotlight_model(input_shape, self.embedding_dimension)
        self.threshold = .01

    def build_spotlight_model(self,input_shape, embedding_dimension):
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)
        net = Conv2D(8, 8, strides=(4, 4), activation='relu', use_bias=False)(normalized)
        net = Conv2D(16, 4, strides=(2, 2), activation='relu', use_bias=False)(net)
        net = Conv2D(16, 3, strides=(1, 1), activation='relu', use_bias=False)(net)
        net = Flatten()(net)
        net = Dense(embedding_dimension * 2, activation='relu', use_bias=False)(net)
        net = Dense(embedding_dimension, use_bias=False)(net)
        model = Model(inputs=frames_input, outputs=net)
        optimizer = RMSprop()
        # optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer, loss='mean_squared_error')

        return model

    def spotlight_train(self, image):
        out = np.ones((1,self.embedding_dimension))
        history = self.spotlight_model.fit(image, out, epochs=1, verbose=0)

    def seen_before(self, image):
        res = self.spotlight_model.predict(image)
        dist = np.linalg.norm(res-np.ones(self.embedding_dimension))
        return dist < self.threshold
