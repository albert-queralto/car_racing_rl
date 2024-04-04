import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import RandomLayers


import torch
import torch.nn as nn
import torch.optim as optim


class QuantumCNN(nn.Module):
    """
    Quantum Neural Network implementation.
    """
    def __init__(
            self,
            n_qubits: int,
            n_layers: int,
            learning_rate: float,
        ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_quantum_network()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum neural network.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device=self.device)
        return self.qnn(x).to(self.device)

    def _build_quantum_network(self) -> None:
        """Builds the quantum neural network model."""
        
        q_device = qml.device("default.qubit.torch", wires=self.n_qubits, torch_device='cuda')

        @qml.qnode(q_device, interface='torch')
        def _quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """
            Quantum circuit for the quantum neural network.
            """
            x = inputs.to(self.device)
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            
            RandomLayers(weights, wires=list(range(self.n_qubits)))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.qnn = qml.qnn.TorchLayer(
            _quantum_circuit,
            self._weight_shapes()
        )

    def _weight_shapes(self) -> dict:
        """
        Returns the weight shapes for the quantum neural network.
        """
        return {"weights": (self.n_layers, self.n_qubits)}