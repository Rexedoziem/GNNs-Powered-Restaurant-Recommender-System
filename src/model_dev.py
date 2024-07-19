import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from steps.config import TrainingParameters

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.node_embedding = nn.Linear(in_channels, hidden_channels)
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_embedding(x)
        for i, conv in enumerate(self.convs):
            x_dict_out = {}
            for node_type, x in x_dict.items():
                if node_type in edge_index_dict.keys():
                    edge_index = edge_index_dict[node_type]
                    if edge_index.size(1) > 0:
                        x_dict_out[node_type] = conv(x_dict, edge_index_dict)
                        if i < len(self.convs) - 1:  # Apply ReLU and dropout to all but the last layer
                            x_dict_out[node_type] = self.relu(x_dict_out[node_type])
                            x_dict_out[node_type] = self.dropout(x_dict_out[node_type])
                    else:
                        x_dict_out[node_type] = x  # Keep the original node features
                else:
                    x_dict_out[node_type] = x  # Keep the original node features
            x_dict = x_dict_out
        # Apply the output linear layer
        x_dict = {k: v for k, v in x_dict.items()}
        return x_dict