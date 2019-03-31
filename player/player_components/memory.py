import numpy as np


class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, frame_height, frame_width, agent_history_length=4, size=1000000, batch_size=32, is_graphical=True):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        self.is_graphical = is_graphical

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        if is_graphical:
            self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        else:
            self.frames = np.empty((self.size, self.frame_height), dtype=np.float16)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
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
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        # if frame.shape != (self.frame_height, self.frame_width):
        #     raise ValueError('Dimension of frame is wrong!')
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
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.minibatch_indices):
            self.minibatch_states[i] = self._get_state(idx - 1)
            # io.imshow(np.squeeze(self.minibatch_states[i,0,:,:]))
            self.minibatch_new_states[i] = self._get_state(idx)
            # io.imshow(np.squeeze(self.minibatch_new_states[i, 0, :, :]))

        return np.transpose(self.minibatch_states, axes=(0, 2, 3, 1)), self.actions[self.minibatch_indices], \
               self.rewards[self.minibatch_indices], np.transpose(self.minibatch_new_states, axes=(0, 2, 3, 1)), \
               self.terminal_flags[self.minibatch_indices]

