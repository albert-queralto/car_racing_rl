import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional
from src.utils.model_setup import cnn_model, linear_model



class DQN(nn.Module):
    """
    Neural network DQN algorithm implementation.
    """
    def __init__(
            self,
            learning_rate: float,
            input_shape: int,
            output_size: int,
            hidden_layers_dim: int,
            conv_params: list[tuple[int, int, tuple[int, int], tuple[int, int]]],
            activation_function: Optional[nn.Module] = nn.ReLU
        ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_layers_dim = hidden_layers_dim
        self.conv_params = conv_params
        self.activation_function = activation_function
        
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._build_network() # Builds the neural network model
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network to calculate the Q-values for the
        DQN case.
        """
        if type(observation) is np.ndarray:
            observation = torch.from_numpy(observation).float().to(device=self.device)
        features = self.cnn_model(observation.to(self.device)).reshape(-1,  self.fc_layer_input_size)

        return self.linear_model(features)

    def _build_network(self) -> None:
        """Builds the neural network model."""
        self.cnn_model = cnn_model(
            conv_params=self.conv_params,
            activation_function=self.activation_function
        )
        self.cnn_model.to(device=self.device)
        
        self.fc_layer_input_size = self.feature_size()
        
        self.linear_model = linear_model(
            input_size=self.fc_layer_input_size,
            output_size=self.output_size,
            hidden_layer_dim=self.hidden_layers_dim, 
            activation_function=self.activation_function
        ).to(device=self.device)

    def feature_size(self) -> int:
        dummy_input = torch.randn((1,) + self.input_shape).to(device=self.device)
        return self.cnn_model(dummy_input).view(1, -1).size(1)

class DuelingDQN(nn.Module):
    """
    Neural network Dueling DQN algorithm.
    """
    def __init__(
            self,
            learning_rate: float,
            input_shape: int,
            output_size: int,
            hidden_layers_dim: int,
            conv_params: list[tuple[int, int, tuple[int, int], tuple[int, int]]],
            activation_function: Optional[nn.Module] = nn.ReLU
        ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_layers_dim = hidden_layers_dim
        self.conv_params = conv_params
        self.activation_function = activation_function
        
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._build_network() # Builds the neural network model
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network to calculate the Q-values for the
        Dueling DQN case. Adds the Advantage and Value functions and returns
        the Q value based on the equation: Q(s,a) = A(s,a) + V(s) - mean(A(s,a))

        Parameters:
        -----------
        observation: torch.Tensor
            Observation from the environment.

        Returns:
        --------
        torch.Tensor
            Q-values for each action in the environment.
        """
        if type(observation) is np.ndarray:
            observation = torch.from_numpy(observation).float().to(device=self.device)
        
        features = self.cnn_model(observation)
        features = features.view(-1, self.fc_layer_input_size)
        
        # Advantage function
        advantage = self.advantage(features)
        advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        
        # Value function
        value = self.value(features)
        value = value.expand(-1, self.output_size)
        return value + advantage

    def _build_network(self) -> nn.Module:
        """
        Builds the neural network model using the linear_nn_model function to
        create a linear neural network with activation functions based on the
        input hidden layer dimensions.
        """
        self.cnn_model = cnn_model(
            conv_params=self.conv_params,
            activation_function=self.activation_function
        )
        self.cnn_model.to(device=self.device)
        self.fc_layer_input_size = self.feature_size()
        
        self.advantage = linear_model(
            input_size=self.fc_layer_input_size,
            output_size=self.output_size,
            hidden_layer_dim=self.hidden_layers_dim, 
            activation_function=self.activation_function
        ).to(device=self.device)
        
        self.value = linear_model(
            input_size=self.fc_layer_input_size,
            output_size=1,
            hidden_layer_dim=self.hidden_layers_dim, 
            activation_function=self.activation_function
        ).to(device=self.device)
    
    def feature_size(self) -> int:
        dummy_input = torch.randn((1,) + self.input_shape).to(device=self.device)
        return self.cnn_model(dummy_input).view(1, -1).size(1)