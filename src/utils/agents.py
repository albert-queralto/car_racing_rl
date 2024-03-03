import torch
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym
from copy import deepcopy
from dataclasses import dataclass
from src.utils.buffers import ReplayBuffer


@dataclass
class BaseAgent(ABC):
    """Abstract class for RL agents."""
    model: torch.nn.Module

    def __post_init__(self):
        # Device where the neural network is executed depending on if CUDA is
        # available or not
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.main_network = self.model.to(device=self.device)
        self.target_network = deepcopy(self.main_network)
        
        self.update_loss = []

    def select_action(self,
            environment: gym.Env,
            network: torch.nn.Module,
            observation: torch.Tensor,
            epsilon: float,
        ) -> torch.Tensor:
        """
        Selects the action to be performed in the environment based on the value
        of epsilon. If epsilon is larger than a random number between 0 and 1,
        performs a random action from the action space. Otherwise, performs an
        action based on the Q-values (maximum Q-value).
        """
        if np.random.random() < epsilon:
            return environment.action_space.sample()
            # return np.random.randint(0, environment.action_space.n)
        q_values = self._calculate_qvalues(network, observation)
        return q_values.argmax().item()

    @abstractmethod
    def calculate_loss(self, *args, **kwargs):
        """Abstract method for calculating loss."""

    def _calculate_qvalues(self,
            network: torch.nn.Module,
            observation_tensor: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculates the Q-values for an observation based on the current network
        weights. The weights are being updated regularly during training.
        """        
        return network.forward(observation_tensor)

    def update_sync_networks(self, episode_steps: int, 
            update_frequency: int,
            sync_frequency: int,
            gamma: float,
            buffer: ReplayBuffer,
            batch_size: int
        ) -> None:
        if episode_steps % update_frequency == 0:
            self.main_network.optimizer.zero_grad()
            batch = buffer.random_sample(batch_size)
            loss = self.calculate_loss(batch, gamma)
            loss.backward()
            self.main_network.optimizer.step()

            if self.device.type == 'cuda':
                self.update_loss.append(loss.cpu().detach().numpy())
            else:
                self.update_loss.append(loss.detach().numpy())
                    
        if episode_steps % sync_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())


@dataclass
class DQNAgent(BaseAgent):
    """Implements the Agent to solve an environment using a Deep Q-Learning
    algorithm."""

    def calculate_loss(self, batch: list, discount_factor: float) -> torch.Tensor:
        """Calculates the loss based on the predicted Q-values from the main
        network and the expected Q-values calculated using the target network.
        The Q-values are calculated using the get_values method, whereas the
        expected Q-values are calculated from the target Q-values and applying
        the Bellman equation, i.e. using the rewards and a discount factor.
        The loss is calculated using the Mean Squared Error.

        Parameters:
        -----------
        batch: list
            List containing the batch of experiences.
        discount_factor: float
            Discount factor used to calculate the loss.

        Returns:
        --------
        loss: torch.Tensor
            Tensor containing the loss values.
        """
        if batch is not None:
            observations, actions, rewards, dones, next_observations = list(batch)
            rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
            actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device)
            dones_t = torch.ByteTensor(dones).to(device=self.device)

            q_values = self._calculate_qvalues(self.main_network, observations)
            q_values = torch.gather(q_values, 1, actions_vals.unsqueeze(1))

            with torch.no_grad():
                next_q_values = self._calculate_qvalues(
                    self.target_network, next_observations
                )
                max_next_q_values = next_q_values.max(dim=-1)[0].detach()

                # Bellman equation
                target_q_values = (
                    rewards_vals + discount_factor * (1 - dones_t) * max_next_q_values
                )

            return torch.nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
    
    
@dataclass
class DuelingDQNAgent(BaseAgent):
    """Implements the Agent to solve an environment using a Dueling Deep
    Q-Learning algorithm."""

    def calculate_loss(self, batch: list, discount_factor: int) -> torch.Tensor:
        """Calculates the loss based on the predicted Q-values from the main
        network and the expected Q-values calculated using the target network.
        The Q-values are calculated using the get_values method, whereas the
        expected Q-values are calculated from the target Q-values and applying
        the Bellman equation, i.e. using the rewards and a discount factor.
        The loss is calculated using the Mean Squared Error.

        Parameters:
        -----------
        batch: list
            List containing the batch of experiences.
        discount_factor: float
            Discount factor used to calculate the loss.

        Returns:
        --------
        loss: torch.Tensor
            Tensor containing the loss values.
        """
        if batch is not None:
            observations, actions, rewards, dones, next_observations = list(batch)
            rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
            actions_vals = torch.LongTensor(actions).to(device=self.device).unsqueeze(1)
            dones_t = torch.ByteTensor(dones).to(device=self.device)

            q_values = self._calculate_qvalues(self.main_network, observations)
            q_values = torch.gather(q_values, 1, actions_vals).squeeze()

            next_actions = torch.max(
                self._calculate_qvalues(self.main_network, next_observations),
                dim=-1
            )[1]

            if self.device == 'cuda':
                next_action_vals = next_actions.reshape(-1,1).to(device=self.device)
            else:
                next_action_vals = torch.LongTensor(next_actions.cpu()).reshape(-1,1).to(
                    device=self.device)

            with torch.no_grad():
                target_qvalues = self._calculate_qvalues(
                    self.target_network, next_observations
                )
                next_q_values = torch.gather(
                    target_qvalues, 1, next_action_vals
                ).detach()

                max_next_q_values = next_q_values.max(dim=-1)[0]

                # Bellman equation
                expected_q_values = (
                    rewards_vals + 
                    discount_factor * (1 - dones_t) * max_next_q_values
                )

            return torch.nn.MSELoss()(q_values, expected_q_values)


@dataclass
class PriorityExperienceReplayMixin(BaseAgent):
    """Mixin class to add the priority experience replay functionality to the
    DQNAgent and DuelingDQNAgent classes."""

    def calculate_loss(self,
            batch: list,
            discount_factor: int,
            weights: np.ndarray
        ) -> tuple[float, np.ndarray]:
        observations, actions, rewards, dones, next_observations = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device).reshape(-1,1)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)
        weights = torch.FloatTensor(np.array(weights, copy=False)).to(self.device)
        
        q_values = torch.gather(
            self.get_qvalues(self.main_network, observations), 1, actions_vals)
        
        next_actions = torch.max(self.get_qvalues(self.main_network, 
                                                next_observations), dim=-1)[1]

        if self.device == 'cuda':
            next_action_vals = next_actions.reshape(-1,1).to(device=self.device)
        else:
            next_action_vals = torch.LongTensor(next_actions).reshape(-1,1).to(
                device=self.device)
        
        target_qvalues = self.get_qvalues(self.target_network, next_observations)
        next_q_values = torch.gather(target_qvalues, 1, next_action_vals).detach()
        max_next_q_values = next_q_values.max(dim=-1)[0]
        
        # Calculates the expected Q values using the 1-step Bellman equation
        expected_q_values = (rewards_vals + 
                    discount_factor * (1 - dones_t) * max_next_q_values)
        loss = torch.nn.MSELoss()(q_values, expected_q_values.reshape(-1, 1))
        weighted_loss = torch.mul(weights, loss).reshape(-1, 1)
        sample_priorities = (weighted_loss + 1e-6).data.cpu().numpy()

        return weighted_loss.mean(), sample_priorities


# @dataclass
# class Nstep