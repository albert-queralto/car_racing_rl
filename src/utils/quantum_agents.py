import numpy as np
import gymnasium as gym

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from src.utils.buffers import ReplayBuffer, PPOReplayBuffer

import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data import DataLoader, RandomSampler

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
        q_values = self._calculate_qvalues(network, observation)
        q_values = q_values.mean(dim=0, keepdim=True)
        action = q_values.argmax().item()
        return int(action)

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

    def update_sync_networks(self,
            episode_steps: int, 
            update_frequency: int,
            sync_frequency: int,
            gamma: float,
            buffer: ReplayBuffer,
            batch_size: int
        ) -> None:
        """
        Updates the main_network if the number of episode_steps is divisible by
        the update_frequency, and syncs the target_network with the
        main_network if the number of episode_steps is divisible by the
        sync_frequency.
        """
        if episode_steps % update_frequency == 0:
            self.main_network.optimizer.zero_grad()
            # Gets the batch as well as the weights and batch indices
            batch, weights, samples = buffer.random_sample(batch_size)
            # Calculates the loss and priorities based on the batch, weights and gamma
            loss, sample_priorities = self.calculate_loss(batch, gamma, weights)
            buffer.update_priorities(samples, sample_priorities)
            loss.backward()
            self.main_network.optimizer.step()

            if self.device.type == 'cuda':
                self.update_loss.append(loss.cpu().detach().numpy())
            else:
                self.update_loss.append(loss.detach().numpy())
                    
        if episode_steps % sync_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def calculate_loss(self,
            batch: list,
            discount_factor: float,
            weights: np.ndarray
        ) -> tuple[float, np.ndarray]:
        observations, actions, rewards, dones, next_observations = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device).reshape(-1,1)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)
        weights = torch.FloatTensor(np.array(weights, copy=False)).to(self.device)
        
        q_values = torch.gather(
            self._calculate_qvalues(self.main_network, observations), 1, actions_vals)
        
        next_actions = torch.max(self._calculate_qvalues(self.main_network, 
                                                next_observations), dim=-1)[1]

        if self.device.type == 'cuda':
            next_action_vals = next_actions.reshape(-1,1).to(device=self.device)
        else:
            next_action_vals = torch.LongTensor(next_actions.cpu()).reshape(-1,1).to(
                device=self.device)
        
        target_qvalues = self._calculate_qvalues(self.target_network, next_observations)
        next_q_values = torch.gather(target_qvalues, 1, next_action_vals).detach()
        max_next_q_values = next_q_values.max(dim=-1)[0]
        
        # Calculates the expected Q values using the 1-step Bellman equation    
        expected_q_values = (rewards_vals + 
                    discount_factor * (1 - dones_t) * max_next_q_values)
        
        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        weighted_loss = torch.mul(weights, loss).reshape(-1, 1)
        sample_priorities = (weighted_loss + 1e-6).data.cpu().numpy()

        return weighted_loss.mean(), sample_priorities


@dataclass
class PPOAgent:
    """
    Class that implements the PPO algorithm.
    Based on the code from: https://github.com/xtma/pytorch_car_caring/
    """
    network: torch.nn.Module
    max_grad_norm: float
    clip_param: float
    ppo_epoch: int
    buffer_capacity: int
    batch_size: int

    def __post_init__(self):
        # Automatic device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initializes the network model with the optimizer
        self.net = self.network.double().to(self.device)

        # Initializes the buffer
        self.buffer = PPOReplayBuffer(self.buffer_capacity)
        self.update_loss = []

    def select_action(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(observation)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def store_transition(self, transition):
        self.buffer.store(transition)

    def calculate_loss_and_update(self, discount_factor: float):
        observation = torch.tensor(self.buffer['observation'], dtype=torch.double).to(self.device)
        action = torch.tensor(self.buffer['action'], dtype=torch.double).to(self.device)
        reward = torch.tensor(self.buffer['reward'], dtype=torch.double).to(self.device).view(-1, 1)
        next_observation = torch.tensor(self.buffer['next_observation'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = reward + discount_factor * self.net(next_observation)[1]
            adv = target_v - self.net(observation)[1]

        dataset = torch.utils.data.TensorDataset(observation, action, reward, next_observation, old_a_logp, target_v, adv)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=RandomSampler(dataset), shuffle=False)

        for _ in range(self.ppo_epoch):
            for batch in dataloader:
                obs_batch, action_batch, _, _, old_a_logp_batch, target_v_batch, adv_batch = batch

                alpha, beta = self.net(obs_batch)[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(action_batch).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp_batch)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_batch
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(obs_batch)[1], target_v_batch)
                loss = action_loss + 2.0 * value_loss

                if self.device.type == 'cuda':
                    self.update_loss.append(loss.cpu().detach().numpy())
                else:
                    self.update_loss.append(loss.detach().numpy())

                self.net.optimizer.zero_grad()
                loss.backward()
                self.net.optimizer.step()