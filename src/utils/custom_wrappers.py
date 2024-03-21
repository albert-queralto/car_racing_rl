"""
This module contains custom wrappers for the Gym environment. The wrappers
are used to preprocess the observations, stack frames, and skip frames.

The code is based on the Stable Baselines 3 Atari wrappers:
- https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py
"""

import numpy as np
import gymnasium as gym
from src.utils.utils import preprocess_observation, stack_frames


class StackFramesWrapper(gym.Wrapper):
    """
    A Gym wrapper that stacks observations.
    """
    def __init__(self, env: gym.Env, stack_frames: int = 4, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.stack_frames = stack_frames
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Resets the environment and sets up the first stacked frame."""
        observation = self.env.reset(**kwargs)[0]

        # Stacks the initial frames to create a sense of motion for the network
        self.stacked_frames = stack_frames(None, observation, True)
        return self.stacked_frames, {}
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Performs one step in the environment and updates the current frame stack."""
        # Accumulates the rewards of the skip frames
        observation, reward, terminated, truncated, _ = self.env.step(action)

        # Places the observation at the end of the stack
        self.stacked_frames = stack_frames(self.stacked_frames, observation, False)

        return self.stacked_frames, reward, terminated, truncated, {}


class WaitFramesWrapper(gym.Wrapper):
    """
    A Gym wrapper that waits some frames after resetting the environment.
    """
    def __init__(self, env: gym.Env, wait_frames: int, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.wait_frames = wait_frames
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Resets the environment and sets up the first stacked frame."""
        observation = self.env.reset(**kwargs)[0]

        # Waits some frames to start the game (until the camera has zoomed in)
        for _ in range(self.wait_frames):
            observation, _, _, _, _ = self.env.step(0)
        
        return observation, {}


class SkipFramesWrapper(gym.Wrapper):
    """A Gym wrapper that skips a certain number of frames."""
    def __init__(self, env, skip_frames, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.skip_frames = skip_frames

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0
        for _ in range(self.skip_frames):
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, {}


class PreprocessObservationWrapper(gym.Wrapper):
    """
    A Gym wrapper that preprocesses the observations.
    """
    def __init__(self, env: gym.Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Resets the environment and sets up the first stacked frame."""
        observation = self.env.reset(**kwargs)[0]
        return preprocess_observation(observation), {}
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Performs one step in the environment and updates the current frame stack."""
        # Accumulates the rewards of the skip frames
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return preprocess_observation(observation), reward, terminated, truncated, {}


class CustomRewardWrapper(gym.Wrapper):
    """
    A Gym wrapper that customizes the rewards.
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Customizes the reward
        if np.mean(observation[:, :, 1]) > 185.0:
            reward -= 0.05
        return observation, reward, terminated, truncated, info


class CustomEnvWrapper(gym.Wrapper):
    """
    A Gym wrapper that combines SkipFramesWrapper, WaitFramesWrapper, and StackFramesWrapper.
    """
    def __init__(self,
        env,
        skip_frames,
        wait_frames,
        stack_frames,
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self.env = CustomRewardWrapper(
            SkipFramesWrapper(
                WaitFramesWrapper(
                    StackFramesWrapper(
                        PreprocessObservationWrapper(env),
                        stack_frames
                    ),
                    wait_frames
                ),
                skip_frames
            )
        )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    
    