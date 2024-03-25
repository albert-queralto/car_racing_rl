import sys
import argparse
from dataclasses import dataclass
import numpy as np
import time
# from codecarbon import track_emissions
from memory_profiler import profile
from typing import Any


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from pathlib import Path
MAIN_PATH = Path().resolve()
sys.path.append(str(MAIN_PATH))

from src.utils.network_models import PPO
from src.utils.buffers import ExperienceBuffer

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v2')
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb, _ = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # four frames for decision
        return np.array(self.stack), {}

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _, _ = self.env.step(action)

            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, die, {}

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


@dataclass
class PPOAgent:
    network: nn.Module
    max_grad_norm: float
    clip_param: float
    ppo_epoch: int
    buffer_capacity: int
    batch_size: int
        
    def __post_init__(self):
        self.training_step = 0
        
        self.net = self.network.double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0


    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                print(alpha.shape)
                print(beta.shape)
                print(s[index].shape)
                print(a[index].shape)
                dist = Beta(alpha, beta)
                print(dist)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                print(a_logp)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.net.optimizer.zero_grad()
                loss.backward()
                self.net.optimizer.step()



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
            'environment_type', 'epsilon', 'episode', 'time_frame_counter',
            'episode_steps', 'episode_reward', 'average_rewards', 'loss_evolution'
        ]
        
        model_store_keys = [
            'model_date', 'model_name', 'model_type', 'environment_name',
            'environment_type', 'environment_params', 'hidden_layers_dim',
            'maximum_reward', 'skip_frames', 'wait_frames', 'stack_frames',
            'batch_size', 'learning_rate', 'discount_factor', 'epsilon_start',
            'epsilon_decay', 'epsilon_end', 'nblock', 'network_update_frequency',
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
    agent: PPOAgent
    env: gym.Env
    buffer: Any
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
        self.model_date = str(datetime.now())
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
        attribute_names = [
            'epsilon', 'episode', 'time_frame_counter',
            'episode_steps', 'episode_reward'
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
        for key, value in dictionary.items():
            dictionary[key] = str(value)

    def save_training_results_to_file(self) -> None:
        """Saves the training results stored a dictionary to a JSON file."""
        results_str = f"{self.model_date}_{self.model_name}_training_results.json"
        file_path = os.path.join(MAIN_PATH, 'models', results_str)
        self._convert_to_string(self.training_results)

        with open(file_path, 'w') as file:
            json.dump(self.training_results, file, indent=4)

    def save_model_params_to_file(self) -> None:
        """Saves the model parameters stored a dictionary to a JSON file."""
        model_str = f"{self.model_date}_{self.model_name}_model_params.json"
        file_path = os.path.join(MAIN_PATH, 'models', model_str)
        self._convert_to_string(self.model_store)

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

        elif self.episode_reward >= reward_threshold:
            self.end_training = True

        return self.end_training

    # @track_emissions
    @profile
    def train(self) -> None:
        training_records = []
        running_score = 0
        state, _ = env.reset()
        for i_ep in range(100000):
            score = 0
            state, _ = env.reset()

            for t in range(1000):
                action, a_logp = agent.select_action(state)
                state_, reward, done, die, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                if args.render:
                    env.render()
                if agent.store((state, action, a_logp, reward, state_)):
                    print('updating')
                    agent.update()
                score += reward
                state = state_
                if done or die:
                    break
            running_score = running_score * 0.99 + score * 0.01

            if i_ep % args.log_interval == 0:
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
                agent.save_param()
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break



if __name__ == "__main__":
    # Sets the training params
    training_params = {
        'model_name': 'DuelingDQN',
        'model_type': 'DuelingDQN',
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
        'network_update_frequency': 1,
        'network_sync_frequency': 100,
        'learning_rate': 0.00025,#3.9e-5, 
        'discount_factor': 0.99,
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
            'render_mode': 'rgb_array',
            'continuous': False, # Continuous or discrete action space
        }
    }
    
    # Sets the buffer params
    buffer_params = {
        'max_capacity': 2000,
        'batch_size': 128,
        # 'alpha': 0.6,
        # 'beta': 0.4,
        # 'beta_step': 0.001
    }
    
    # env = gym.make(
    #     id=environment_params['environment_name'],
    #     **environment_params['env_params']
    # )
    
    # env = CustomEnvWrapper(
    #     env,
    #     skip_frames=environment_params['skip_frames'],
    #     wait_frames=environment_params['wait_frames'],
    #     stack_frames=environment_params['stack_frames']
    # )
    
    # environment_params.update({
    #     'reward_threshold': env.spec.reward_threshold,
    #     'action_space': env.action_space.n,
    #     'observation_space_shape': (4, 84, 84),
    # })
    
    env = Env()

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

    # training_records = []
    # running_score = 0
    # state, _ = env.reset()
    # for i_ep in range(100000):
    #     score = 0
    #     state, _ = env.reset()

    #     for t in range(1000):
    #         action, a_logp = agent.select_action(state)
    #         state_, reward, done, die, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    #         if args.render:
    #             env.render()
    #         if agent.store((state, action, a_logp, reward, state_)):
    #             print('updating')
    #             agent.update()
    #         score += reward
    #         state = state_
    #         if done or die:
    #             break
    #     running_score = running_score * 0.99 + score * 0.01

    #     if i_ep % args.log_interval == 0:
    #         print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
    #         agent.save_param()
    #     if running_score > env.reward_threshold:
    #         print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
    #         break