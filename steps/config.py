import torch

class TrainingParameters:
    """Parameters for the training process."""
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT: str = 'C:/Users/HP/Desktop/RESTAURANT_COMPONENTS/saved_model'