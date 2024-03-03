import torch.nn as nn


def cnn_model(
        conv_params: list[tuple[int, int, tuple[int, int], tuple[int, int]]],
        activation_function: nn.Module = nn.ReLU
    ) -> nn.Sequential:
    """
    Creates a convolutional neural network with ReLU activation functions.

    Parameters:
    -----------
    conv_params: list[tuple[int, int, tuple[int, int], tuple[int, int]]]
        A list of tuples where each tuple represents parameters for a convolutional layer.
        Each tuple contains (input_shape, output_channels, kernel_size, stride).
    activation_function: nn.Module = nn.ReLU
        The activation function to use for the hidden layers.
    """
    activation_func = getattr(nn, activation_function)

    # Define convolutional layers
    conv_layers = []
    for i, (in_channels, out_channels, kernel_size, stride) in enumerate(conv_params):
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        conv_layers.append(activation_func())
        conv_layers.append(nn.MaxPool2d(kernel_size=2) if i < len(conv_params) - 1 else nn.Flatten())

    return nn.Sequential(*conv_layers)

def linear_model(
        input_size: int,
        output_size: int,
        hidden_layer_dim: int,
        activation_function: nn.Module = nn.ReLU
    ) -> nn.Sequential:
    """
    Creates a fully connected neural network with ReLU activation functions.
    """
    activation_func = getattr(nn, activation_function)
    
    # Define fully connected layers
    fc_layers = [
        nn.Linear(input_size, hidden_layer_dim),
        activation_func(),
        nn.Linear(hidden_layer_dim, output_size)
    ]

    return nn.Sequential(*fc_layers)