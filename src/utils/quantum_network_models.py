import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import RandomLayers
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.model_setup import quantum_linear_model


class QuantumCNN(nn.Module):
    """
    Quantum Neural Network implementation.
    """
    def __init__(
            self,
            input_shape: int,
            n_qubits: int,
            n_layers: int,
            learning_rate: float,
            output_size: int,
            hidden_layer_dim: int,
            activation_function: Optional[nn.Module] = nn.ReLU
        ):
        super().__init__()
        self.input_shape = input_shape
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.hidden_layer_dim = hidden_layer_dim
        self.activation_function = activation_function
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_quantum_network()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum neural network.
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        # Flatten each sample in the batch
        observation = observation.view(observation.shape[0], -1)

        # Get Q-values from the Quantum Neural Network
        q_values = self.qnn(observation)

        # Get Q-values from the Linear Model
        linear_model = quantum_linear_model(
            input_size=q_values.shape[1],
            output_size=self.output_size,
            hidden_layer_dim=self.hidden_layer_dim,
            activation_function=self.activation_function
        ).to(device=self.device)

        q_values = linear_model(q_values)

        return q_values.to(self.device)

    def _build_quantum_network(self) -> None:
        """Builds the quantum neural network model."""
        q_device = qml.device("default.qubit.torch", wires=self.n_qubits, torch_device='cuda')

        @qml.qnode(q_device, interface='torch')
        def _quantum_circuit1(inputs: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
            """
            Quantum circuit for the quantum neural network.
            """
            x = inputs * np.pi
            x = x.to(self.device)
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
                qml.RZ(x[i], wires=i)
                qml.Hadamard(wires=i)
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            RandomLayers(weights, wires=list(range(self.n_qubits)))

            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            return expvals
        
        def _quantum_circuit2(inputs: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
            """
            Quantum circuit for the quantum neural network.
            """
            x = inputs * np.pi
            x = x.to(self.device)
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i], wires=i)
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    qml.RZ(x[i], wires=i)

            RandomLayers(weights, wires=list(range(self.n_qubits)))

            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            return expvals
        
        def _quantum_circuit3(inputs: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
            """
            Quantum circuit for the quantum neural network.
            """
            x = inputs * np.pi
            x = x.to(self.device)
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i], wires=i)
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i], wires=i)
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i], wires=i)
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    qml.RZ(x[i], wires=i)

            RandomLayers(weights, wires=list(range(self.n_qubits)))

            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            return expvals

        self.qnn = qml.qnn.TorchLayer(
            _quantum_circuit1,
            self._weight_shapes()
        )

    def _weight_shapes(self) -> dict:
        """
        Returns the weight shapes for the quantum neural network.
        """
        return {"weights": (self.n_layers, self.n_qubits)}