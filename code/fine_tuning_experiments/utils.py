from llama_index.embeddings.adapter import BaseAdapter
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from typing import Dict

class CustomNN(BaseAdapter):
    """
    A three layer neural network for the adapter.

    Consists of two linear layers with ReLU activation functions, one dropout layer and one output layer.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        out_features: int,
        add_residual: bool = False,
        dropout: float = 0.1,
    ):
        super(CustomNN, self).__init__()
        self.in_features = in_features
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_features = out_features
        self._add_residual = add_residual

        self.fc1 = nn.Linear(in_features, hidden_dim_1, bias = True)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2, bias = True)
        self.fc3 = nn.Linear(hidden_dim_2, out_features, bias = True)
        self.dropout = nn.Dropout(dropout)
        # if add_residual, then add residual_weight (init to 0)
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        """
        output1 = F.relu(self.fc1(x))
        output2 = F.relu(self.fc2(output1))
        output2 = self.dropout(output2)

        output3 = self.fc3(output2)

        if self._add_residual:
            output3 = self.residual_weight * output3 + x

        return output3
    
    def get_config_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "out_features": self.out_features,
            "add_residual": self._add_residual,
            "dropout": self.dropout.p
        }