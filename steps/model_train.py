import torch
import os
import torch.nn as nn
from torcheval.metrics import HitRate
from tqdm import tqdm
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from .config import TrainingParameters
from src.model_dev import GraphSAGE
import pandas as pd
from src.evaluation import ndcg_at_k, rmse
from zenml.logger import get_logger
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch.optim as optim
from typing import Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

logger = get_logger(__name__)

# Set up MLflow tracking URI
mlflow.set_tracking_uri(get_tracking_uri())

# Enable autologging
mlflow.pytorch.autolog()

# Set the experiment name or ID
experiment_name = "Experiment_Name"  # Replace with your experiment name
mlflow.set_experiment(experiment_name)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        """
        Computes the BPR loss.

        Args:
            pos_scores (torch.Tensor): Predicted scores for positive (observed) user-item interactions.
            neg_scores (torch.Tensor): Predicted scores for negative (unobserved) user-item interactions.

        Returns:
            torch.Tensor: BPR loss value.
        """
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))
        return loss.mean()



@step
def model_initialization_step(
    in_channels: Annotated[int, "Number of input features"],
    hidden_channels: Annotated[int, "Number of hidden units"] = 256,
    out_channels: Annotated[int, "Number of output features"] = 128,
    num_layers: Annotated[int, "Number of GraphSAGE layers"] = 3
) -> Tuple[
    Annotated[torch.nn.Module, "model"],
    Annotated[torch.nn.Module, "criterion"],
    Annotated[torch.optim.Optimizer, "optimizer"],
    Annotated[Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau], "lr_scheduler"]
]:
    """
    Initialize the GraphSAGE model, loss function, optimizer, and learning rate scheduler.
    
    Returns:
        Tuple containing:
        - model: Initialized GraphSAGE model
        - criterion: Loss function
        - optimizer: Optimizer for model parameters
        - lr_scheduler: Learning rate scheduler
    """
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(TrainingParameters.device)

    # Set requires_grad=True for all model parameters
    for param in model.parameters():
        param.requires_grad = True

    criterion = BPRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingParameters.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Manually log some hyperparameters (MLflow will autolog most of these, but it's good to be explicit)
    mlflow.log_param("in_channels", in_channels)
    mlflow.log_param("hidden_channels", hidden_channels)
    mlflow.log_param("out_channels", out_channels)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("learning_rate", TrainingParameters.learning_rate)

    return model, criterion, optimizer, lr_scheduler


@step
def training_step(
    train_loader: Annotated[torch.utils.data.DataLoader, "Training data loader"],
    model: Annotated[torch.nn.Module, "PyTorch model"],
    criterion: Annotated[torch.nn.Module, "Loss function"],
    optimizer: Annotated[torch.optim.Optimizer, "Model optimizer"],
    epoch: Annotated[int, "Current epoch number"]
) -> Tuple[
    Annotated[float, "train_loss"],
    Annotated[float, "train_hit_rate"]
]:
    """
    Perform one epoch of training.
    
    Returns:
        Tuple containing:
        - train_loss: Average training loss for the epoch
        - train_hit_rate: Hit rate metric for the training data
    """
    model.train()
    total_loss = 0
    total_examples = 0
    train_hit_rate = HitRate(k=10)

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        batch = batch.to(TrainingParameters.device)
        
        # Forward pass
        output = model(batch.x_dict, batch.edge_index_dict)
        user_output = output['user']

        valid_mask = (batch['user'].n_id < user_output.size(0))
        valid_user_ids = batch['user'].n_id[valid_mask]
        valid_observed_items = batch['user'].y[valid_mask].long()

        y_pred = user_output[valid_mask]
        unobserved_items = torch.randint(0, user_output.size(1), (valid_user_ids.size(0),))

        pos_scores = user_output[valid_user_ids, valid_observed_items]
        neg_scores = user_output[valid_user_ids, unobserved_items]

        loss = criterion(pos_scores, neg_scores)
        
        train_hit_rate.update(y_pred, valid_observed_items)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * valid_user_ids.size(0)
        total_examples += valid_user_ids.size(0)
        
    train_loss = total_loss / total_examples
    # Compute the final hit rate
    final_hit_rate1 = train_hit_rate.compute()
    if final_hit_rate1.shape == torch.Size([1]):
        final_hit_rate1 = final_hit_rate1.item()
    else:
        final_hit_rate1 = final_hit_rate1.mean().item()

    # Manually log some metrics (MLflow will autolog most of these, but it's good to be explicit)
    mlflow.log_metric(f"train_loss_epoch_{epoch}", train_loss)
    mlflow.log_metric(f"train_hit_rate_epoch_{epoch}", final_hit_rate1)

    return train_loss, final_hit_rate1

@step
def validation_step(
    valid_loader: Annotated[torch.utils.data.DataLoader, "Validation data loader"],
    model: Annotated[torch.nn.Module, "PyTorch model"],
    criterion: Annotated[torch.nn.Module, "Loss function"],
    epoch: Annotated[int, "Current epoch number"]
) -> Tuple[
    Annotated[float, "eval_loss"],
    Annotated[float, "val_hit_rate"],
    Annotated[float, "ndcg"]
]:
    """
    Validate the model on the validation data.
    
    Returns:
        Tuple containing:
        - eval_loss: Average validation loss
        - val_hit_rate: Hit rate metric for the validation data
        - ndcg: Normalized Discounted Cumulative Gain metric
    """
    model.eval()
    total_loss = 0
    total_examples = 0
    final_outputs = []
    all_y_true = []
    valid_hit_rate = HitRate(k=5)

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Validation"):
            batch = batch.to(TrainingParameters.device)

            output = model(batch.x_dict, batch.edge_index_dict)
            user_output = output['user']

            valid_mask = (batch['user'].n_id < user_output.size(0))
            valid_user_ids = batch['user'].n_id[valid_mask]
            valid_observed_items = batch['user'].y[valid_mask].long()

            y_pred = user_output[valid_mask]
            unobserved_items = torch.randint(0, user_output.size(1), (valid_user_ids.size(0),))

            pos_scores = user_output[valid_user_ids, valid_observed_items]
            neg_scores = user_output[valid_user_ids, unobserved_items]

            loss = criterion(pos_scores, neg_scores)

            valid_hit_rate.update(y_pred, valid_observed_items)

            total_loss += loss.item() * valid_user_ids.size(0)
            total_examples += valid_user_ids.size(0)

            final_outputs.append(y_pred.cpu())
            all_y_true.append(valid_observed_items.cpu())

    # Calculate average losses and metrics
    eval_loss = total_loss / total_examples

    # Concatenate the tensors in the final_outputs and all_y_true lists
    final_outputs = torch.cat(final_outputs)
    all_y_true = torch.cat(all_y_true)

    # Calculate predicted labels for the entire dataset
    final_predicted_labels = torch.argmax(final_outputs, dim=1)

    # Compute the final hit rate
    final_hit_rate1 = valid_hit_rate.compute()
    if final_hit_rate1.shape == torch.Size([1]):
        final_hit_rate1 = final_hit_rate1.item()
    else:
        final_hit_rate1 = final_hit_rate1.mean().item()
        
    # Compute NDCG@k
    ndcg_k = ndcg_at_k(all_y_true, final_predicted_labels, k=10)

    # Manually log some metrics (MLflow will autolog most of these, but it's good to be explicit)
    mlflow.log_metric(f"eval_loss_epoch_{epoch}", eval_loss)
    mlflow.log_metric(f"val_hit_rate_epoch_{epoch}", final_hit_rate1)
    mlflow.log_metric(f"ndcg_epoch_{epoch}", ndcg_k)

    return eval_loss, all_y_true, final_hit_rate1, final_predicted_labels, ndcg_k


@step
def model_training_loop(
    train_loader: Annotated[torch.utils.data.DataLoader, "Training data loader"],
    valid_loader: Annotated[torch.utils.data.DataLoader, "Validation data loader"],
    model: Annotated[torch.nn.Module, "PyTorch model"],
    criterion: Annotated[torch.nn.Module, "Loss function"],
    optimizer: Annotated[torch.optim.Optimizer, "Model optimizer"],
    lr_scheduler: Annotated[Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau], "Learning rate scheduler"]
) -> Annotated[torch.nn.Module, "Trained PyTorch model"]:
    best_score = float('inf')
    best_model = None
    OUTPUT_DIR = "C:/Users/HP/Desktop/RESTAURANT_COMPONENTS/saved_model"

    for epoch in range(TrainingParameters.num_epochs):
        if (epoch) % 5 == 0:
            train_loss, train_hit_rate = training_step(train_loader, model, criterion, optimizer, epoch)
            eval_loss, all_y_true, final_hit_rate1, final_predicted_labels, ndcg_k = validation_step(valid_loader, model, criterion, epoch)
            
            # Step the learning rate scheduler
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(eval_loss)
            else:
                lr_scheduler.step()
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Hit Rate = {train_hit_rate:.4f}")
            logger.info(f"Validation Loss = {eval_loss:.4f}, Validation Hit Rate = {final_hit_rate1:.4f}, NDCG = {ndcg_k:.4f}")
            
            # scoring
            rmse_class = rmse()
            score = rmse_class.calculate_scores(all_y_true, final_predicted_labels)
            logger.info(f'Epoch {epoch+1} - score: {score:.4f}')
            
            if score < best_score:
                best_score = score
                best_model = model.state_dict()
                logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                
                # Save the best model
                save_path = os.path.join(OUTPUT_DIR, f'best_model.pth')
                torch.save({
                    'model': best_model,
                    'final_predicted_labels': final_predicted_labels
                }, save_path)

    # Load the best model before returning
    model.load_state_dict(best_model)
    return model

        