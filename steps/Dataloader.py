from zenml import step
import torch
from typing import Tuple
#import torch_sparse
#import torch_scatter
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit, ToUndirected
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

@step
def data_loader_step(combined_data: HeteroData) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    # Split the combined_data into train, validation, and test sets based on 'user' node type
    combined_data = T.RandomNodeSplit(split='random', num_val=0.1, num_test=0.1)(combined_data)
    
    # Convert the graphs to undirected (if needed)
    combined_data = T.ToUndirected()(combined_data)

    # Create the train loader
    train_loader = NeighborLoader(
        combined_data,
        # Sample 10 neighbors for each node and edge type for 2 iterations
        num_neighbors={key: [10] * 2 for key in combined_data.edge_types},
        # Use a batch size of 32 for sampling training nodes of type user
        batch_size=32,
        input_nodes=('user', combined_data['user'].train_mask),
    )

    # Create the validation loader
    valid_loader = NeighborLoader(
        data=combined_data,
        num_neighbors={key: [10] * 2 for key in combined_data.edge_types},
        batch_size=32,
        input_nodes=('user', combined_data['user'].val_mask),
    )

    # Create the test loader
    test_loader = NeighborLoader(
        data=combined_data,
        num_neighbors={key: [10] * 2 for key in combined_data.edge_types},
        batch_size=32,
        input_nodes=('user', combined_data['user'].test_mask),
    )

    # Sample a batch from the train loader to verify it's working
    #sampled_hetero_data = next(iter(train_loader))
    #print(f"Sampled batch size: {sampled_hetero_data['user'].batch_size}")

    return train_loader, valid_loader, test_loader