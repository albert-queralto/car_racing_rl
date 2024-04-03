import os
import sys
from dataclasses import dataclass
import numpy as np
import json
# from codecarbon import track_emissions
from memory_profiler import profile
from typing import Any
from datetime import datetime

import gymnasium as gym
import torch

from pathlib import Path
MAIN_PATH = Path().resolve()
sys.path.append(str(MAIN_PATH))

from src.utils.network_models import PPO
from src.utils.agents import PPOAgent
from src.utils.buffers import ExperienceBuffer
from src.utils.custom_wrappers import ContinuousEnvWrapper


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
            'episode', 'time_frame_counter', 'reward_moving_avg',
            'episode_steps', 'episode_reward', 'average_rewards', 'loss_evolution',
        ]
        
        model_store_keys = [
            'model_date', 'model_name', 'model_type', 'environment_name',
            'environment_type', 'environment_params', 'hidden_layers_dim',
            'maximum_reward', 'skip_frames', 'wait_frames', 'stack_frames',
            'batch_size', 'learning_rate', 'discount_factor', 
            'nblock', 'network_update_frequency',
            'network_sync_frequency', 'max_episodes', 'activation_function',
            'optimizer', 'time_frame_counter_threshold', 'instantaneous_reward_threshold',
            'negative_reward_counter_threshold', 'episode_reward_threshold',
            'conv_params', 'gas_weight'
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
            reward_moving_avg: float,
            mean_reward: float,
            mean_loss: float,
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
            f"- Moving Average: {reward_moving_avg:.2f} "
            f"- Mean Reward: {mean_reward:.2f} "
            f"- Loss: {mean_loss:.2f} "
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
    agent: PPOAgent
    env: gym.Env
    buffer: Any
    training_params: dict[str, Any]
    environment_params: dict[str, Any]
    buffer_params: dict[str, Any]
    
    def __post_init__(self) -> None:
        self._unpack_params()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.info_printer = InfoPrinter()
        self.training_results, self.model_store = SetupStorage().initialize_dictionaries()

    def _reset_training_variables(self) -> None:
        """
        Resets the variables used during network training.
        """
        self.model_date = str(datetime.now())
        self.episode = 0
        self.end_training = False
        self.reward_moving_avg = 0

    def _reset_episode_variables(self) -> None:
        """
        Resets the variables used in each episode.
        """
        self.observation, _ = self.env.reset()
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
        attribute_names = [
            'episode', 'time_frame_counter',
            'episode_steps', 'episode_reward', 'reward_moving_avg'
        ]

        for attr_name in attribute_names:
            attr_value = getattr(self, attr_name)
            self.training_results[attr_name].append(attr_value)

        self.mean_reward = np.mean(self.training_results['episode_reward'][-self.nblock:])
        self.mean_loss = np.mean(self.agent.update_loss)
        
        self.training_results['average_rewards'].append(self.mean_reward)
        self.training_results['loss_evolution'].append(self.mean_loss)

    def save_model_params(self):
        """
        Saves the model parameters in the model_store dictionary.
        """
        for attr_name in self.model_store.keys():
            self.model_store[attr_name].append(getattr(self, attr_name))

    def save_model_binary(self, model: torch.nn.Module) -> None:
        """Saves the model binary to a file."""
        model_str = f"{self.model_date}_{self.model_name}.pt"
        file_path = os.path.join(MAIN_PATH, 'models', model_str)
        torch.save(model, file_path)

    def _convert_to_string(self, dictionary):
        dictionary = dictionary.copy()
        for key, value in dictionary.items():
            dictionary[key] = str(value)
        return dictionary

    def save_training_results_to_file(self) -> None:
        """Saves the training results stored a dictionary to a JSON file."""
        results_str = f"{self.model_date}_{self.model_name}_training_results.json"
        file_path = os.path.join(MAIN_PATH, 'models', results_str)
        training_results = self._convert_to_string(self.training_results)

        with open(file_path, 'w') as file:
            json.dump(training_results, file, indent=4)

    def save_model_params_to_file(self) -> None:
        """Saves the model parameters stored a dictionary to a JSON file."""
        model_str = f"{self.model_date}_{self.model_name}_model_params.json"
        file_path = os.path.join(MAIN_PATH, 'models', model_str)
        model_store = self._convert_to_string(self.model_store)

        with open(file_path, 'w') as file:
            json.dump(model_store, file, indent=4)

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

        elif self.reward_moving_avg > reward_threshold:
            print(f"Moving average reward is now {self.reward_moving_avg} and the last episode runs to {self.episode_reward}")
            self.end_training = True

        return self.end_training

    # @track_emissions
    @profile
    def train(self) -> None:
        self._reset_training_variables()
        self._reset_episode_variables()
        self.info_printer.print_device_info(self.agent.device)
        
        # for i_ep in range(100000):
        while not self.end_training:
            self._reset_episode_variables()

            # for t in range(self.max_steps):
            while not self.is_gamedone:
                self.perform_episode()
                if self.end_training:
                    self.save_model_params()
                    self.env.close()
                    
                    # Save the training results to a file, as well as the model
                    # parameters and the model binary
                    self.save_training_results_to_file()
                    self.save_model_params_to_file()
                    self.save_model_binary(self.agent.net)
                    break

    def perform_episode(self) -> None:
        self.is_gamedone = self.perform_action_step()
        self.time_frame_counter += 1    
        self.reward_moving_avg = self.reward_moving_avg * 0.99 + self.episode_reward * 0.01

        if self.is_gamedone:
            self.episode += 1
            self.mean_reward = 0
            self.mean_loss = 0
            self.save_training_results()
            self.agent.update_loss = []
            
            self.maximum_reward = max(self.maximum_reward, self.mean_reward)
            
            self.info_printer.print_episode_info(
                buffer=self.buffer,
                episode=self.episode,
                max_episodes=self.max_episodes,
                time_frame_counter=self.time_frame_counter,
                episode_reward=self.episode_reward,
                reward_moving_avg=self.reward_moving_avg,
                mean_reward=self.mean_reward,
                mean_loss=self.mean_loss,
                maximum_reward=self.maximum_reward
            )
            
            self.save_training_results_to_file()
            self.save_model_binary(self.agent.net)
        
            self.end_training = self.check_end_training(self.max_episodes, self.reward_threshold)

    def perform_action_step(self) -> bool:
        """Perform an action step in the environment."""
        action, a_logp = self.agent.select_action(self.observation)
        next_observation, reward, done, truncated, _ = self.env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

        if action[1] > 0.5:
            reward *= self.gas_weight

        if self.agent.store_transition((self.observation, action, a_logp, reward, next_observation)):
            self.agent.calculate_loss_and_update(self.discount_factor)
        
        self.set_negative_reward_counter(reward)
        self.episode_reward += reward
        self.observation = next_observation.copy()
        if done or truncated:
            self.observation, _ = self.env.reset()
            return True

        early_stop = self.early_stop_episode()
        if early_stop:
            return early_stop

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



if __name__ == "__main__":
    # Sets the training params
    training_params = {
        'model_name': 'PPO',
        'input_shape': 4,
        'cnn_output_size': 256,
        'hidden_layers_dim': 256,
        'num_actions': 3,
        'layer_sizes': [8, 16, 32, 64, 128],
        'kernel_sizes': [4, 3, 3, 3, 3],
        'strides': [2, 2, 2, 2, 1],
        'max_grad_norm': 0.5,
        'clip_param': 0.1,
        'ppo_epoch': 10,
        'time_frame_counter_threshold': 100,
        'instantaneous_reward_threshold': 0,
        'negative_reward_counter_threshold': 50,
        'episode_reward_threshold': -10,
        'learning_rate': 0.00025,#3.9e-5, 
        'discount_factor': 0.99,
        'max_episodes': 100000,
        'max_steps': 1000,
        'n_steps': None,
        'activation_function': 'ReLU',
        'optimizer': 'Adam',
        'nblock': 100
    }
    
    # Sets the environment params
    environment_params = {
        'environment_name': "CarRacing-v2", # Name of the environment
        'maximum_reward': -10000, # Initial value for the maximum reward
        'skip_frames': 8, # Number of frames to skip
        'wait_frames': 50, # Number of frames to wait
        'stack_frames': 4, # Number of frames to stack
        'gas_weight': 4, # Weight for the gas action
        'env_params': {
            'render_mode': 'rgb_array',
            'continuous': True, # Continuous or discrete action space
        }
    }
    
    # Sets the buffer params
    buffer_params = {
        'max_capacity': 100000,
        'batch_size': 128,
    }
    
    env = gym.make(
        id=environment_params['environment_name'],
        **environment_params['env_params']
    )
    
    # env = CustomEnvWrapper(
    #     env,
    #     skip_frames=environment_params['skip_frames'],
    #     wait_frames=environment_params['wait_frames'],
    #     stack_frames=environment_params['stack_frames']
    # )
    env = ContinuousEnvWrapper(env=env, skip_frames=environment_params['skip_frames'])
    
    environment_params.update({
        'reward_threshold': env.reward_threshold,
        # 'action_space': env.action_space.n,
        'observation_space_shape': (4, 84, 84),
    })

    model = PPO(
        input_shape=training_params['input_shape'],
        cnn_output_size=training_params['cnn_output_size'],
        hidden_size=training_params['hidden_layers_dim'],
        num_actions=training_params['num_actions'],
        learning_rate=training_params['learning_rate'],
        layer_sizes=training_params['layer_sizes'],
        kernel_sizes=training_params['kernel_sizes'],
        strides=training_params['strides'],
        activation_function=training_params['activation_function'],
    )
    
    agent = PPOAgent(
        network=model,
        max_grad_norm=training_params['max_grad_norm'],
        clip_param=training_params['clip_param'],
        ppo_epoch=training_params['ppo_epoch'],
        buffer_capacity=buffer_params['max_capacity'],
        batch_size=buffer_params['batch_size']
    )

    trainer = AgentTraining(
        agent=agent,
        env=env,
        buffer=None,
        training_params=training_params,
        environment_params=environment_params,
        buffer_params=buffer_params
    )

    trainer.train()