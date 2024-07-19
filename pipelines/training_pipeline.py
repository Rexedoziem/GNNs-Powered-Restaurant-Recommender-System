from zenml import pipeline
from steps.Dataloader import data_loader_step
from zenml.logger import get_logger
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.preprocess_data import preprocess_data_step
from steps.create_bipartite_graph import create_bipartite_graphs, add_node_features, add_edge_features, create_combined_data
from steps.model_train import model_initialization_step, model_training_loop


from zenml import pipeline
from steps.config import TrainingParameters

from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
import mlflow

logger = get_logger(__name__)

# Enable autologging
mlflow.pytorch.autolog()


@pipeline
def train_pipeline(data_path: str):
    data = ingest_df(data_path)
    preprocessed_data = preprocess_data_step(data=data)
    user_restaurant_graph, user_category_graph, restaurant_mapping = create_bipartite_graphs(data=preprocessed_data)
    user_restaurant_graph, user_category_graph = add_node_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
    user_restaurant_graph, user_category_graph = add_edge_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
    combined_data, _, in_channels = create_combined_data(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, preprocessed_data=preprocessed_data)
    
    train_loader, valid_loader, test_loader = data_loader_step(combined_data)
    # Model initialization
    model, criterion, optimizer, lr_scheduler = model_initialization_step(in_channels=in_channels)
    # Model training
    trained_model = model_training_loop(train_loader, valid_loader, model, criterion, optimizer, lr_scheduler)
    # Model evaluation
    final_test_hit_rate, ndcg_k, test_loss = evaluate_model(
        model=trained_model, 
        test_loader=test_loader,
        criterion=criterion
    )