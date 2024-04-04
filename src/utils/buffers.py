import random
import numpy as np
from abc import abstractmethod, ABC
from collections import deque, namedtuple
from dataclasses import dataclass

class ExperienceBuffer(ABC):
    """
    Base abstract class for the experience replay buffer.
    """
    @abstractmethod
    def store_experience(self, state, action, reward, done, next_state):
        """
        Stores a new experience tuple (observation, action, reward, done,
        next_observation) to the buffer.
        """
        pass

    @abstractmethod
    def random_sample(self, batch_size: int):
        """
        Randomly samples a batch of experiences from the buffer to be used
        during the training process.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the length of the buffer.
        """
        return len(self.buffer)

    def is_ready(self, batch_size) -> bool:
        """
        Returns a boolean value that indicates if the buffer is ready to be
        sampled.
        """
        return len(self.buffer) >= batch_size


Buffer = namedtuple(
    'Buffer',
    field_names=[
        'observation',
        'action',
        'reward',
        'done',
        'next_observation'
    ]
)


@dataclass
class ReplayBuffer(ExperienceBuffer):
    """
    Class that implements the experience replay buffer used to stabilize the
    learning process and improve the efficiency of the reinforcement learning
    samples.

    References:
    -----------
    - Richard S. Sutton, Andrew G. Barto, Reinforcement Learning. 
                                    An Introduction (2nd Ed.), MIT Press (2018).
    - Shangtong Zhang, Richard S. Sutton, 
                    A Deeper Look at Experience Replay, arXiv:1712.01275 (2017).
    """
    max_capacity: int

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.max_capacity)

    def store_experience(self,
        observation,
        action,
        reward,
        done,
        next_observation
    ) -> None:
        # If the buffer is full, remove the oldest experience
        if len(self.buffer) >= self.max_capacity:
            self.buffer.popleft()

        # Add the new experience to the buffer
        buffer = Buffer(
            observation,
            action,
            reward,
            done,
            next_observation
        )
        self.buffer.append(buffer)

    def random_sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return tuple(map(np.array, zip(*batch)))


@dataclass
class PrioritizedReplayBuffer(ExperienceBuffer):
    """
    Class that implements the prioritized experience replay buffer. It builds on
    the elements from the ReplayBuffer class to prioritize experience that have
    a higher contribution to the learning process.

    References:
    -----------
    - Richard S. Sutton, Andrew G. Barto, Reinforcement Learning. 
                                    An Introduction (2nd Ed.), MIT Press (2018).
    - Shangtong Zhang, Richard S. Sutton, 
                    A Deeper Look at Experience Replay, arXiv:1712.01275 (2017).
    - Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, 
                        Prioritized Experience Replay, arXiv:1511.05952 (2015).
    """
    max_capacity: int
    alpha: float
    beta: float
    beta_step: int

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.max_capacity)
        self.priorities = np.zeros(self.max_capacity, dtype=np.float32)

    def store_experience(self,
        observation,
        action,
        reward,
        done,
        next_observation
    ) -> None:
        max_priority = self._get_max_priority()

        # If the buffer is full, remove the oldest experience
        if len(self.buffer) >= self.max_capacity:
            self.buffer.popleft()

        buffer = Buffer(
            observation,
            action,
            reward,
            done,
            next_observation
        )
        self.buffer.append(buffer)

        self.priorities[len(self.buffer) - 1] = max_priority

    def random_sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        sampling_probs = self._calculate_sampling_probs()
        samples = random.choices(
            population=range(len(self.buffer)),
            weights=sampling_probs,
            k=batch_size
        )
        sampled_probs = sampling_probs[samples]

        weights = self._calculate_weights(sampled_probs)

        self._update_beta()

        batch = [self.buffer[i] for i in samples]
        return tuple(map(np.array, zip(*batch))), weights, samples

    def _get_max_priority(self) -> np.ndarray:
        """
        Returns the maximum priority of the experiences. If the replay buffer
        is empty returns 1, else returns the maximum value of the priorities.
        """
        return np.max(self.priorities) if self.buffer else 1

    def _calculate_sampling_probs(self) -> np.ndarray:
        """
        Calculates the sampling probabilities of the experiences using the
        formula described in https://arxiv.org/abs/1511.05952.
        """
        priorities = self.priorities[:len(self.buffer)]
        sampling_probs = priorities ** self.alpha
        return sampling_probs / np.sum(sampling_probs)

    def _calculate_weights(self, sampled_probs: np.ndarray) -> np.ndarray:
        """
        Calculates the weights of a sample using the formula described in
        https://arxiv.org/abs/1511.05952.
        """
        weights = (len(self.buffer) * sampled_probs) ** (-self.beta)
        weights /= max(weights)
        return weights

    def _update_beta(self) -> None:
        self.beta = min(1.0, self.beta + self.beta_step)

    def update_priorities(self, samples: np.ndarray, priorities: np.ndarray) -> None:
        """
        Updates the priorities of the experiences at the given indices provided
        by the variable samples.
        """
        self.priorities[samples] = priorities.squeeze()


@dataclass
class PPOReplayBuffer:
    """
    Class that implements a replay experience buffer.
    """
    capacity: int
    def __post_init__(self):
        self.buffer = deque(maxlen=self.capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer, batch_size, replace=True)
        return np.array(batch)

    def __len__(self):
        return len(self.buffer)