import os
import io
import json
import sys
import time
import numpy as np
from datetime import datetime
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from typing import Any
from dataclasses import dataclass
from enum import Enum

from pathlib import Path
MAIN_PATH = Path().resolve()
sys.path.append(str(MAIN_PATH))

from src.utils.setup_dirs import setup_dirs
from src.utils.custom_wrappers import CustomEnvWrapper
from src.utils.buffers import ExperienceBuffer, ReplayBuffer, PrioritizedReplayBuffer
from src.utils.agents import BaseAgent, DQNAgent, DuelingDQNAgent
from src.utils.network_models import DQN, DuelingDQN

setup_dirs(MAIN_PATH)


class Mode(Enum):
    """
    Enum class to define the mode of the action.
    """
    EXPLORATION = 1
    TRAIN = 2



class SetupStorage:
    """
    Sets up the storages for the training results and the model.
    """

    def initialize_dictionaries(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Initializes the dictionaries with the input parameters and the results.

        Returns:
        --------
        training_results: dict[str, Any]
            Initialized dictionary to store the training results.
        model_store: dict[str, Any]
            Initialized dictionary to store the model parameters.
        """
        buffer_keys = ['alpha', 'beta', 'beta_step', 'max_capacity']
        n_step_key = ['n_steps']

        training_results_keys = [
            'model_date', 'model_name', 'model_type', 'environment_name',
            'environment_type', 'epsilon_evolution', 'episode_count',
            'episode_steps', 'episode_rewards', 'average_rewards', 'loss_evolution'
        ]

        model_store_keys = [
            'model_date', 'model_name', 'model_type', 'environment_name',
            'environment_type', 'environment_parameters', 'hidden_layers',
            'batch_size', 'learning_rate', 'discount_factor', 'epsilon_start',
            'epsilon_decay', 'epsilon_end', 'nblock', 'network_update_frequency',
            'network_sync_frequency', 'max_episodes', 'activation_function',
            'optimizer', 'model_binary'
        ] + buffer_keys + n_step_key

        # Initializes the training_results and model_store dictionaries using
        # the corresponding keys
        training_results = {key: [] for key in training_results_keys}
        model_store = {key: [] for key in model_store_keys}

        return training_results, model_store



@dataclass
class InfoPrinter:
    """Class responsible for printing information on screen."""

    def print_training_info(self, training_params: dict[str, Any]) -> None:
        """
        Prints the training parameters on screen.

        Parameters:
        -----------
        training_params: dict[str, Any]
            Dictionary with the training parameters.
        """
        print("Training the neural network...")
        print("Params:\n-------")
        
        for param_name, param_value in training_params.items():
            print(f"{param_name}: {param_value}")

    def print_episode_info(self,
            buffer: ExperienceBuffer,
            episode: int,
            max_episodes: int,
            time_frame_counter: int,
            episode_reward: float,
            mean_reward: float,
            mean_loss: float,
            epsilon: float,
            maximum_reward: float,
        ) -> None:
        """
        Prints the episode information on screen during training.
        """
        buffer_type = buffer.__class__.__name__

        info = (
            f"Episode: {episode}/{max_episodes} "
            f"- Time Frames: {time_frame_counter} "
            f"- Episode Reward: {episode_reward:.2f} "
            f"- Mean Reward: {mean_reward:.2f} "
            f"- Loss: {mean_loss:.2f} "
            f"- Epsilon: {epsilon:.3f} "
            f"- Max Reward: {maximum_reward:.2f}"
        )

        if buffer_type == 'PrioritizedReplayBuffer':
            info += f"- Beta {buffer.beta:.2f}"
        
        print(info, end="\r", flush=True)

    def print_device_info(self, device: torch.device) -> None:
        """
        Prints the device information on screen.
        """
        print(f"Device: {device}")



@dataclass
class AgentTraining:
    agent: BaseAgent
    env: gym.Env
    buffer: ExperienceBuffer
    training_params: dict[str, Any]
    environment_params: dict[str, Any]
    buffer_params: dict[str, Any]

    def __post_init__(self) -> None:
        # Unpacks the parameters from the dictionaries and creates the variables
        self._unpack_params()

        # Automatic device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initializes the info printer
        self.info_printer = InfoPrinter()

        # Initializes the dictionaries for the training results and the model store
        self.training_results, self.model_store = SetupStorage().initialize_dictionaries()

    def _reset_training_variables(self) -> None:
        """
        Resets the variables used during network training.
        """
        self.model_date = datetime.now()
        self.epsilon = self.epsilon_start
        self.episode = 0
        self.end_training = False

    def _reset_episode_variables(self) -> None:
        """
        Resets the variables used in each episode.
        """
        self.observation, _ = self.env.reset() # Grayscale and stacked image (4, 84, 84)
        self.episode_reward = 0
        self.episode_steps = 0
        self.is_gamedone = False
        self.negative_reward_counter = 0
        self.time_frame_counter = 1

    def _unpack_params(self) -> None:
        """
        Unpacks the parameters from the dictionaries and creates the variables.
        """
        for param_dict in [self.training_params, self.environment_params, self.buffer_params]:
            for attr_name, value in param_dict.items():
                setattr(self, attr_name, value)

    def save_training_results(self):
        """
        Saves the training results in the training_results dictionary.
        """
        self.training_results['model_date'].append(self.model_date)
        self.training_results['model_name'].append(self.model_name)
        self.training_results['model_type'].append(self.model_type)
        self.training_results['environment_name'].append(self.environment_name)
        self.training_results['environment_type'].append(self.environment_type)
        self.training_results['epsilon_evolution'].append(self.epsilon)
        self.training_results['episode_count'].append(self.episode)
        self.training_results['episode_steps'].append(self.episode_steps)
        self.training_results['episode_rewards'].append(self.episode_reward)
        
        self.mean_reward = np.mean(self.training_results['episode_rewards'][-self.nblock:])
        self.mean_loss = np.mean(self.agent.update_loss)
        
        self.training_results['average_rewards'].append(self.mean_reward)
        self.training_results['loss_evolution'].append(self.mean_loss)

    def save_model_params(self):
        """
        Saves the model parameters in the model_store dictionary.
        """
        self.model_store['model_date'].append(self.model_date)
        self.model_store['model_name'].append(self.model_name)
        self.model_store['model_type'].append(self.model_type)
        self.model_store['environment_name'].append(self.environment_name)
        self.model_store['environment_type'].append(self.environment_type)
        self.model_store['environment_parameters'].append(self.env_params)
        self.model_store['hidden_layers'].append(self.hidden_layers)
        self.model_store['batch_size'].append(self.batch_size)
        self.model_store['max_capacity'].append(self.max_capacity)
        self.model_store['learning_rate'].append(self.learning_rate)
        self.model_store['discount_factor'].append(self.discount_factor)
        self.model_store['epsilon_start'].append(self.epsilon_start)
        self.model_store['epsilon_decay'].append(self.epsilon_decay)
        self.model_store['epsilon_end'].append(self.epsilon_end)
        self.model_store['nblock'].append(self.nblock)
        self.model_store['network_update_frequency'].append(self.network_update_frequency)
        self.model_store['network_sync_frequency'].append(self.network_sync_frequency)
        self.model_store['max_episodes'].append(self.max_episodes)
        self.model_store['n_steps'].append(self.n_steps)
        self.model_store['alpha'].append(self.alpha)
        self.model_store['beta'].append(self.beta)
        self.model_store['beta_step'].append(self.beta_step)
        self.model_store['activation_function'].append(self.activation_function)
        self.model_store['optimizer'].append(self.optimizer)

    def save_model_binary(self, model: torch.nn.Module) -> None:
        """Saves the model binary to a file."""
        model_str = f"{self.model_date}_{self.model_name}.pt"
        file_path = os.path.join(MAIN_PATH, 'models', model_str)
        torch.save(model, file_path)

    def save_training_results_to_file(self) -> None:
        """Saves the training results stored a dictionary to a JSON file."""
        results_str = f"{self.model_date}_{self.model_name}_training_results.json"
        file_path = os.path.join(MAIN_PATH, 'models', results_str)

        with open(file_path, 'w') as file:
            json.dump(self.training_results, file, indent=4)

    def save_model_params_to_file(self) -> None:
        """Saves the model parameters stored a dictionary to a JSON file."""
        model_str = f"{self.model_date}_{self.model_name}_model_params.json"
        file_path = os.path.join(MAIN_PATH, 'models', model_str)

        with open(file_path, 'w') as file:
            json.dump(self.model_store, file, indent=4)

    def check_end_training(self,
            max_episodes: int,
            reward_threshold: int
        ) -> bool:
        """
        Checks if the training has ended by either reaching the maximum number
        of episodes or surpassing the reward threshold.
        """
        if self.episode >= max_episodes:
            self.end_training = True
            print("Maximum episodes reached")

        elif self.mean_reward >= reward_threshold:
            self.end_training = True
            print(f"\nReward threshold reached in {self.episode} episodes")

        return self.end_training

    def train(self) -> None:
        
        if self.env.render_mode == 'rgb_array':
            self.env = RecordVideo(self.env, os.path.join(MAIN_PATH, 'videos'), episode_trigger=lambda x: x % 1 == 0)
        
        self._reset_training_variables()
        self._reset_episode_variables()
        self.info_printer.print_device_info(self.agent.device)
        
        print("Filling replay buffer with experiences...")
        self.info_printer.print_training_info(self.training_params)
        
        # Fill the buffer with experiences in order to start the training
        while not self.buffer.is_ready(batch_size=self.batch_size):
            self.perform_action_step(Mode.EXPLORATION)

        while not self.end_training:
            self._reset_episode_variables()

            while not self.is_gamedone:
                self.perform_episode()
                if self.end_training:
                    self.save_model_params()
                    self.env.close()
                    
                    # Save the training results to a file, as well as the model
                    # parameters and the model binary
                    self.save_training_results_to_file()
                    self.save_model_params_to_file()
                    self.save_model_binary(self.agent.main_network)
                    break
        
    def perform_episode(self) -> None:
        """
        Performs an episode in the environment and updates the agent's network.
        """
        self.is_gamedone = self.perform_action_step(mode=Mode.TRAIN)
        
        self.agent.update_sync_networks(
            self.episode_steps,
            self.network_update_frequency,
            self.network_sync_frequency,
            self.discount_factor,
            self.buffer,
            self.batch_size
        )
        
        self.time_frame_counter += 1
        
        if self.is_gamedone:
            self.episode += 1
            self.mean_reward = 0
            self.mean_loss = 0
            self.save_training_results()
            self.agent.update_loss = []
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            self.maximum_reward = max(self.maximum_reward, self.mean_reward)
        
            self.info_printer.print_episode_info(
                buffer=self.buffer,
                episode=self.episode,
                max_episodes=self.max_episodes,
                time_frame_counter=self.time_frame_counter,
                episode_reward=self.episode_reward,
                mean_reward=self.mean_reward,
                mean_loss=self.mean_loss,
                epsilon=self.epsilon,
                maximum_reward=self.maximum_reward
            )
        
            self.end_training = self.check_end_training(
                max_episodes=self.max_episodes,
                reward_threshold=self.reward_threshold
            )

    def perform_action_step(self, mode: Mode) -> bool:
        """
        Performs an action step in the environment and updates the agent's network.
        """     
        # Selects an action based on the mode and the observation
        action = self.select_action(mode, self.observation)

        # Performs a step in the environment
        next_observation, instantaneous_reward, done, _, _ = env.step(action)
        
        # Calculates the value of the negative reward counter and adjusts the
        # episode reward based on the instantaneous reward and the action taken
        self.set_negative_reward_counter(instantaneous_reward)
        self.adjust_episode_reward(instantaneous_reward, action)

        self.handle_nsteps(self.observation, action, instantaneous_reward, done, next_observation)

        self.observation = next_observation.copy()

        if done:
            self.observation, _ = self.env.reset()

        early_stop = self.early_stop_episode()
        if early_stop is True:
            return early_stop

        return done

    def handle_nsteps(self,
            observation_tensor: torch.Tensor,
            action: np.ndarray, 
            reward: float, 
            done: bool, 
            next_observation_tensor: torch.Tensor
        ):
        """
        Handles the n-steps for the training.
        """
        if self.n_steps:
            self.agent.add_step(observation_tensor, action, reward, done, next_observation_tensor)
            if self.agent.completed():
                self.complete_n_step_rollout()

        else:
            self.buffer.store_experience(observation_tensor, action, reward, done, next_observation_tensor)

    def complete_n_step_rollout(self):
        """
        Completes n-step rollout and stores experiences in the buffer.
        """
        self.agent.n_step_roll_out_collapse(gamma=self.discount_factor)
        for observation, action, reward, done, next_observation in self.agent.steps:
            self.buffer.store_experience(observation, action, reward, done, next_observation)
        self.agent.reset()
    
    def early_stop_episode(self) -> bool:
        """
        Stops the episode early if the negative reward counter surpasses a
        certain threshold or the episode reward is negative.
        
        Based on the code from:
        https://github.com/matteoprata/DeepRL-Class/blob/main/src/main_train.py
        """
        if (self.negative_reward_counter >= self.negative_reward_counter_threshold
            ) or (self.episode_reward < self.episode_reward_threshold):
            return True
        return False
    
    def set_negative_reward_counter(self, instantaneous_reward: float) -> None:
        """
        Sets the negative reward counter based on the instantaneous reward.
        
        Based on the code from:
        https://github.com/matteoprata/DeepRL-Class/blob/main/src/main_train.py
        """
        if (self.time_frame_counter > self.time_frame_counter_threshold
            ) and (instantaneous_reward < self.instantaneous_reward_threshold):
            self.negative_reward_counter += 1
        else:
            self.negative_reward_counter = 0

    def adjust_episode_reward(self, instantaneous_reward: float, action: int) -> None:
        """
        Adjusts the episode reward based on the instantaneous reward and the
        action taken.
        
        Based on the code from:
        https://github.com/matteoprata/DeepRL-Class/blob/main/src/main_train.py
        """
        if action == 3:
            instantaneous_reward *= self.gas_weight

        self.episode_reward += instantaneous_reward

    def select_action(self, mode: Mode, observation: np.ndarray) -> int:
        """Selects an action based on the mode and the observation."""
        if mode == Mode.TRAIN:
            return self.agent.select_action(
                environment=self.env,
                network=self.agent.main_network,
                observation=observation,
                epsilon=self.epsilon,
            )
        return self.env.action_space.sample()


if __name__ == '__main__':
    # Start time
    start_time = time.time()

    # Sets the training params
    training_params = {
        'model_name': 'DQN',
        'model_type': 'DQN',
        'input_channels': 3,
        'hidden_layers_dim': 256,
        'epsilon_start': 1,
        'epsilon_decay': 0.995,
        'epsilon_end': 0.01,
        'time_frame_counter_threshold': 100,
        'instantaneous_reward_threshold': 0,
        'negative_reward_counter_threshold': 50,
        'episode_reward_threshold': -10,
        'network_update_frequency': 1,
        'network_sync_frequency': 100,
        'learning_rate': 0.00025,#3.9e-5, 
        'discount_factor': 0.99,
        'conv_params': [
            (4, 32, (8, 8), (4, 4)),
            (32, 64, (4, 4), (2, 2)),
            (64, 256, (2, 2), (1, 1)),
        ],
        'max_episodes': 1000,
        'n_steps': None,
        'activation_function': 'ReLU',
        'optimizer': 'Adam',
        'nblock': 100
    }
    
    # Sets the environment params
    environment_params = {
        'environment_name': "CarRacing-v2", # Name of the environment
        'environment_type': 'Gym', # Type of environment
        'maximum_reward': -10000, # Initial value for the maximum reward
        'width': 84, # Width of the image
        'height': 84, # Height of the image
        'skip_frames': 5, # Number of frames to skip
        'wait_frames': 50, # Number of frames to wait
        'stack_frames': 4, # Number of frames to stack
        'gas_weight': 1.5, # Weight for the gas action
        'env_params': {
            'render_mode': 'human', # rgb_array
            'continuous': False, # Continuous or discrete action space
        }
    }
    
    # Sets the buffer params
    buffer_params = {
        'max_capacity': 100000,
        'batch_size': 64,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_step': 0.001
    }
    
    env = gym.make(
        id=environment_params['environment_name'],
        **environment_params['env_params']
    )
    
    env = CustomEnvWrapper(
        env,
        skip_frames=environment_params['skip_frames'],
        wait_frames=environment_params['wait_frames'],
        stack_frames=environment_params['stack_frames']
    )
    
    environment_params.update({
        'reward_threshold': env.spec.reward_threshold,
        'action_space': env.action_space.n,
        'observation_space_shape': (4, 84, 84),
    })
    
    # Initialize the buffer, the model, the agent and the network updater
    buffer = ReplayBuffer(max_capacity=buffer_params['max_capacity'])

    model = DQN(
        learning_rate=training_params['learning_rate'],
        input_shape=environment_params['observation_space_shape'],
        output_size=environment_params['action_space'],
        hidden_layers_dim=training_params['hidden_layers_dim'],
        conv_params=training_params['conv_params'],
        activation_function=training_params['activation_function'],
    )
    
    agent = DQNAgent(model)
    
    # Initialize the training class
    trainer = AgentTraining(
        agent = agent,
        env=env,
        buffer=buffer,
        training_params=training_params,
        environment_params=environment_params,
        buffer_params=buffer_params
    )

    # Start the training
    trainer.train()