import numpy as np


class ReplayMemory:

    def __init__(self, frame_height, frame_width, agent_history_length=4, size=1000000, batch_size=32, is_graphical=True):

        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        self.is_graphical = is_graphical

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        if is_graphical:
            self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        else:
            self.frames = np.empty((self.size, self.frame_height), dtype=np.float16)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

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

    def add_experience(self, action, frame, reward, terminal):

        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

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
                if self.terminal_flags[index - self.agent_history_length:index].any():
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

