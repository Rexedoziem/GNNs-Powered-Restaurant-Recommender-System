from zenml import step
import torch
from tqdm import tqdm
from torcheval.metrics import HitRate
from torch_geometric.loader import NeighborLoader
from src.model_dev import GraphSAGE
from typing import Tuple
import logging
import mlflow
from zenml.client import Client
from typing_extensions import Annotated
from src.evaluation import ndcg_at_k
from steps.config import TrainingParameters

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: GraphSAGE,
    test_loader: NeighborLoader,
    criterion: torch.nn.Module,
) -> Tuple[
    Annotated[float, 'final_test_hit_rate'],
    Annotated[float, 'ndcg_k'],
    Annotated[float, 'test_loss'],
]:
    # Initialize the HitRate metric for testing
    test_hit_rate = HitRate(k=5)  # Top 5 results for testing
    
    model.eval()
    
    # Get predictions on the test set
    with torch.no_grad():
        total_loss = 0
        total_examples = 0
        final_outputs = []
        all_y_true = []
        
        for batch in tqdm(test_loader):
            batch = batch.to(TrainingParameters.device)
            # Forward pass
            output = model(batch.x_dict, batch.edge_index_dict)
            user_output = output['user']
            
            # Filter out invalid user_ids
            valid_mask = (batch['user'].n_id < user_output.size(0))
            valid_user_ids = batch['user'].n_id[valid_mask]
            valid_observed_items = batch['user'].y[valid_mask].long()
            
            y_pred = user_output[valid_mask]
            unobserved_items = torch.randint(0, user_output.size(1), (valid_user_ids.size(0),))
            
            # Compute positive and negative scores
            pos_scores = user_output[valid_user_ids, valid_observed_items]
            neg_scores = user_output[valid_user_ids, unobserved_items]
            # Compute BPR loss
            loss = criterion(pos_scores, neg_scores)
            
            # Update metrics
            total_loss += loss.item() * valid_user_ids.size(0)
            total_examples += valid_user_ids.size(0)
            final_outputs.append(y_pred.cpu())
            all_y_true.append(valid_observed_items.cpu())
            
            # Update the HitRate metric with the predicted and true labels
            test_hit_rate.update(y_pred, valid_observed_items)
    
    # Calculate average losses and metrics
    test_loss = total_loss / total_examples
    logging.info("test_loss: {}".format(test_loss))
    mlflow.log_metric("final_test_loss", test_loss)
    
    # Compute the final hit rate
    final_test_hit_rate = test_hit_rate.compute()
    if final_test_hit_rate.shape == torch.Size([1]):
        final_test_hit_rate = final_test_hit_rate.item()
    else:
        final_test_hit_rate = final_test_hit_rate.mean().item()
    
    logging.info("final_test_hit_rate: {}".format(final_test_hit_rate))
    mlflow.log_metric("final_test_hit_rate", final_test_hit_rate)
    # Concatenate the tensors in the final_outputs and all_y_true lists
    final_outputs = torch.cat(final_outputs)
    all_y_true = torch.cat(all_y_true)
    
    # Calculate predicted labels for the entire dataset
    final_predicted_labels = torch.argmax(final_outputs, dim=1)
    
    # Compute NDCG@k
    ndcg_class = ndcg_at_k()
    ndcg_k = ndcg_class.calculate_scores(all_y_true, final_predicted_labels, k=5)
    mlflow.log_metric("final_ndcg_k", ndcg_k)
    
    return final_test_hit_rate, ndcg_k, test_loss