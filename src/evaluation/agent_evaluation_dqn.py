import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import gymnasium as gym
from dataclasses import dataclass

from pathlib import Path
MAIN_PATH = Path().resolve()
sys.path.append(str(MAIN_PATH))

from src.utils.custom_wrappers import CustomEnvWrapper
from src.utils.agents import DQNAgent, BaseAgent
from src.utils.network_models import DQN

@dataclass
class AgentEvaluation:
    """
    Class that evaluates the agent by executing the number of episodes defined
    with the maximum value of steps and performing the actions given by the
    learned Q-values.
    """
    env: gym.Env
    agent: BaseAgent
    save_folder: str
    file_name: str
    episodes: int

    def __post_init__(self):
        self.max_steps = self.env.spec.max_episode_steps
        self.action_space = self.env.action_space.n
        self.reward_episode_counts = {
                                        'episode_rewards': [], 
                                        'episode_steps': [], 
                                        'game_count': []
                                    }

    def evaluate(self) -> None:
        """
        Method that evaluates the agent by executing the number of episodes
        defined with the maximum value of steps and performing the actions
        given by the learned Q-values.
        """
        self.agent.main_network.eval()

        for episode in range(self.episodes):
            episode_reward = 0
            episode_steps = 0

            state, _ = self.env.reset()
            done = False

            while not done:
                self.env.render()

                action = self.agent.select_action(
                    environment=self.env,
                    network=self.agent.main_network,
                    observation=state,
                    epsilon=0.01
                )
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                print(f"Game: {episode}, Reward: {episode_reward}")
                state = next_state

                if episode_steps >= self.max_steps:
                    done = True

            self.reward_episode_counts['episode_rewards'].append(episode_reward)
            self.reward_episode_counts['episode_steps'].append(episode_steps)
            self.reward_episode_counts['game_count'].append(episode)

if __name__ == "__main__":
    
    # Sets the training params
    test_params = {
        'input_shape': 4,
        'hidden_layers_dim': 256,
        'learning_rate': 0.0001,
        'conv_params': [
            (4, 32, (8, 8), (4, 4)),
            (32, 64, (4, 4), (2, 2)),
            (64, 32, (2, 2), (1, 1)),
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
        'skip_frames': 5, # Number of frames to skip
        'wait_frames': 50, # Number of frames to wait
        'stack_frames': 4, # Number of frames to stack
        'gas_weight': 4, # Weight for the gas action
        'env_params': {
            'render_mode': 'human',
            'continuous': False, # Continuous or discrete action space
        }
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

    save_folder = '/home/docker/car_racing/models/DQN/DQN_1/'
    file_name = '2024-03-29_DQN.pt'
    
    model = torch.load(os.path.join(save_folder, file_name))
    
    agent = DQNAgent(model)
    
    tester = AgentEvaluation(
        env=env,
        agent=agent,
        save_folder=save_folder,
        file_name=file_name,
        episodes=1000
    )
    tester.evaluate()