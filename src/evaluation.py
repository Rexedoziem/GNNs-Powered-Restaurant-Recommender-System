import logging
from abc import ABC, abstractmethod
import torch


class Evaluation(ABC):
    """Abstract class defining strategy for evaluating our model"""

    @abstractmethod
    def calculate_scores(self, y_true, y_pred):
        """
        Calculate the scores for the model
        Args:
            y_true: True labels(Tensors)
            y_pred: Predicted labels(Tensors)
        Returns:
            None
        """
        pass

class rmse(Evaluation):
    """
    Calculates the Root Mean Squared Error (RMSE) between the actual and predicted tensors.
    
    Args:
        y_true (torch.Tensor): Tensor of actual values.
        y_pred (torch.Tensor): Tensor of predicted values.
        
    Returns:
        float: The Root Mean Squared Error.
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("calculating rmse score")
            y_true = y_true.float()  # Convert y_true to float
            y_pred = y_pred.float()  # Convert y_pred to float
            mse = torch.mean((y_true - y_pred) ** 2)
            rmse = torch.sqrt(mse)
            logging.info("rmse score: {}".format(rmse.item()))
            return rmse.item()
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e

class ndcg_at_k(Evaluation):
    def calculate_scores(self, y_true, y_pred, k):
        try:
            logging.info("calculating ndcg_at_k score")
            device = y_true.device
            
            # Ensure y_pred and y_true are 2D tensors for batch processing
            if y_pred.dim() == 1:
                y_pred = y_pred.view(1, -1)
            if y_true.dim() == 1:
                y_true = y_true.view(1, -1)
                
            batch_size = y_true.size(0)
            
            _, indices = torch.topk(y_pred, k, dim=1)
            
            relevance = y_true.gather(1, indices).float()  # Gather the ground truth relevance scores
            
            sorted_relevance, _ = torch.sort(relevance, dim=1, descending=True)
            
            # Compute the DCG
            discounts = torch.log2(torch.arange(2, k + 2, device=device).float())
            dcg = (sorted_relevance / discounts).sum(dim=1)
            
            # Compute the ideal DCG
            ideal_relevance = torch.sort(y_true, dim=1, descending=True)[0][:, :k]
            ideal_dcg = (ideal_relevance / discounts).sum(dim=1)
            ideal_dcg = torch.clamp(ideal_dcg, min=torch.finfo(torch.float32).eps)  # Avoid division by zero
            
            # Compute the NDCG
            ndcg = dcg / ideal_dcg
            logging.info("ndcg score: {}".format(ndcg.mean()))
            return ndcg.mean()
        except Exception as e:
            logging.error("Error in calculating ndcg_at_k: {}".format(e))
            raise e