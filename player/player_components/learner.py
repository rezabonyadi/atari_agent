import tensorflow as tf
from keras import layers, callbacks
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.initializers import VarianceScaling

import tensorflow as tf
from keras import backend as K


class QLearner:
    def __init__(self, n_actions, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4,
                 batch_size=32, gamma=0.99):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.main_learner = DQN(self.n_actions, learning_rate,
                                self.frame_height, self.frame_width, agent_history_length)

        self.target_learner = DQN(self.n_actions, learning_rate,
                                  self.frame_height, self.frame_width, agent_history_length)

        self.targets = np.zeros((batch_size,))
        self.set_computation_device()

        self.tbCallBack = callbacks.TensorBoard(log_dir='./output/Tensorboards', histogram_freq=0, write_graph=True, write_images=True)

    @staticmethod
    def set_computation_device():
        num_cores = 14
        GPU = True

        if GPU:
            num_GPU = 1
            num_CPU = 1
        else:
            num_CPU = 1
            num_GPU = 0

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores,
                                allow_soft_placement=True,
                                device_count={'CPU': num_CPU,
                                              'GPU': num_GPU}
                                )

        session = tf.Session(config=config)
        K.set_session(session)

    def predict(self, states):
        actions_mask = np.ones((states.shape[0], self.n_actions))
        return self.main_learner.model.predict([states, actions_mask])  # separate old model to predict

    def train(self, current_state_batch, actions, rewards, next_state_batch, terminal_flags):

        self.calculate_target_q_values(next_state_batch, terminal_flags, rewards)

        one_hot_actions = np.eye(self.n_actions)[np.array(actions).reshape(-1)]
        one_hot_targets = one_hot_actions * self.targets[:, None]

        history = self.main_learner.model.fit([current_state_batch, one_hot_actions], one_hot_targets,
                                 epochs=1, batch_size=self.batch_size, verbose=0, callbacks=[self.tbCallBack])

        return history.history['loss'][0]

    def update_target_network(self):
        print('Updating the target network')
        self.target_learner.model.set_weights(self.main_learner.model.get_weights())

    def calculate_target_q_values(self, next_state_batch, terminal_flags, rewards):
        actions_mask = np.ones((self.batch_size, self.n_actions))
        q_next_state = self.main_learner.model.predict([next_state_batch, actions_mask])  # separate old model to predict
        action = np.argmax(q_next_state, axis=1)
        q_target = self.target_learner.model.predict([next_state_batch, actions_mask])  # separate old model to predict

        for i in range(self.batch_size):
            if terminal_flags[i]:
                self.targets[i] = rewards[i] - 1.0
            else:
                self.targets[i] = rewards[i] + self.gamma * q_target[i, action[i]]


class DQN:
    """Implements a Deep Q Network"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_actions, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
        """
        Args:
            n_actions: Integer, number of possible actions
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        input_shape = (frame_height, frame_width, agent_history_length)
        # model = self.legacy_model(input_shape, self.n_actions)
        # model = self.dueling_convnet(input_shape, self.n_actions)
        model = self.my_convnet(input_shape, self.n_actions)
        # model = self.nature_convnet(input_shape, self.n_actions)
        model.summary()

        optimizer = RMSprop(lr=self.learning_rate, rho=0.95)
        # optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer, loss=self.huber_loss)

        self.model = model

    def huber_loss(self, y, q_value):
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        return loss


    @staticmethod
    def legacy_model(input_shape, num_actions):
        frames_input = layers.Input(input_shape, name='inputs')
        actions_input = layers.Input((num_actions,), name='action_mask')

        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)

        conv_1 = layers.convolutional.Conv2D(
            64, (8, 8), strides=(2, 2), activation='relu', kernel_initializer='VarianceScaling')(normalized)
        conv_2 = layers.convolutional.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='VarianceScaling')(conv_1)
        conv_3 = layers.convolutional.Conv2D(
            32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='VarianceScaling')(conv_2)
        conv_4 = layers.convolutional.Conv2D(
            32, (7, 7), strides=(1, 1), activation='relu', kernel_initializer='VarianceScaling')(conv_3)

        conv_flattened = layers.core.Flatten()(conv_4)
        hidden = layers.Dense(256, activation='relu')(conv_flattened)
        output = layers.Dense(num_actions)(hidden)

        filtered_output = layers.Multiply(name='QValue')([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        return model

    @staticmethod
    def linear(input_shape, num_actions):
        model = Sequential()
        model.add(Flatten(
            input_shape=input_shape))
        model.add(Dense(
            num_actions,
            activation=None))
        return model

    @staticmethod
    def convnet(input_shape, num_actions):
        model = Sequential()
        model.add(Conv2D(16, 8, strides=(4, 4), activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, 4, strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_actions, activation=None))
        return model

    @staticmethod
    def convnet_bn(input_shape, num_actions):
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)
        net = Conv2D(16, 8, strides=(4, 4), activation='relu')(normalized)
        net = Conv2D(32, 4, strides=(2, 2), activation='relu')(net)
        net = Flatten()(net)
        net = Dense(32, activation='relu')(net)
        net = BatchNormalization()(net)
        net = Dense(num_actions, activation=None)(net)
        model = DQN.add_action_mask_layer(net, frames_input, num_actions)

        return model

    @staticmethod
    def simpler_convnet(input_shape, num_actions):
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)
        net = Conv2D(16, 8, strides=(4, 4), activation='relu')(normalized)
        net = Conv2D(32, 4, strides=(2, 2), activation='relu')(net)
        net = Flatten()(net)
        net = Dense(32, activation='relu')(net)
        net = Dense(num_actions, activation=None)(net)
        model = DQN.add_action_mask_layer(net, frames_input, num_actions)

        return model

    @staticmethod
    def nature_convnet(input_shape, num_actions):
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)
        net = Conv2D(32, 8, strides=(4, 4), activation='relu')(normalized)
        net = Conv2D(64, 4, strides=(2, 2), activation='relu')(net)
        net = Conv2D(64, 3, strides=(1, 1), activation='relu')(net)
        net = Flatten()(net)
        net = Dense(512, activation='relu')(net)
        net = Dense(num_actions, activation=None)(net)
        model = DQN.add_action_mask_layer(net, frames_input, num_actions)

        return model

    @staticmethod
    def dueling_convnet(input_shape, num_actions):
        initializer = VarianceScaling(scale=2.0)
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)

        net = Conv2D(32, (8, 8), strides=(4, 4),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(normalized)
        net = Conv2D(64, (4, 4), strides=(2, 2),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)
        net = Conv2D(64, (3, 3), strides=(1, 1),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)
        net = Conv2D(1024, (7, 7), strides=(1, 1),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)

        net = Flatten()(net)
        advt = Dense(256, kernel_initializer=initializer)(net)
        # advt = Dense(50, kernel_initializer=initializer)(net)

        advt = Dense(num_actions)(advt)
        value = Dense(256, kernel_initializer=initializer)(net)
        # value = Dense(50, kernel_initializer=initializer)(net)

        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, num_actions]))(value)
        final = Add()([value, advt])

        model = DQN.add_action_mask_layer(final, frames_input, num_actions)

        # model = Model(inputs=inputs, outputs=final)
        return model

    @staticmethod
    def my_convnet(input_shape, num_actions):
        initializer = VarianceScaling(scale=2.0)
        frames_input = Input(shape=input_shape)
        normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)

        net = Conv2D(8, (8, 8), strides=(4, 4),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(normalized)
        net = Conv2D(16, (4, 4), strides=(2, 2),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)
        net = Conv2D(16, (4, 4), strides=(1, 1),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)
        net = Conv2D(16, (4, 4), strides=(1, 1),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)
        net = Conv2D(32, (3, 3), strides=(1, 1),
                     activation='relu', kernel_initializer=initializer,
                     padding='valid', use_bias=False)(net)

        net = Flatten()(net)
        advt = Dense(16, kernel_initializer=initializer)(net)
        # advt = Dense(50, kernel_initializer=initializer)(net)

        advt = Dense(num_actions)(advt)
        value = Dense(16, kernel_initializer=initializer)(net)
        # value = Dense(50, kernel_initializer=initializer)(net)

        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, num_actions]))(value)
        final = Add()([value, advt])

        model = DQN.add_action_mask_layer(final, frames_input, num_actions)

        # model = Model(inputs=inputs, outputs=final)
        return model

    @staticmethod
    def add_action_mask_layer(final, frames_input, num_actions):
        actions_input = layers.Input((num_actions,), name='action_mask')
        filtered_output = layers.Multiply(name='QValue')([final, actions_input])
        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        return model
