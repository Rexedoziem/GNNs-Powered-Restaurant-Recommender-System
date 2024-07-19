import numpy as np
from zenml.steps import step
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
import torch
import logging
#from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from typing import Dict

from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from pipelines.utils import get_data_for_test
from steps.create_bipartite_graph import create_bipartite_graphs, add_node_features, add_edge_features, create_combined_data, create_pyg_data
from steps.model_train import model_initialization_step, model_training_loop
from steps.Dataloader import data_loader_step
from steps.preprocess_data import preprocess_data_step

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger config"""
    min_accuracy: float = 0.5

@step(enable_cache=False)
def dynamic_importer():
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = 'model',
) -> MLFlowDeploymentService:
    
    # get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and model name

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    combined_data: HeteroData,
) -> Dict[int, List[Tuple[int, float]]]:
    try:
        service.start(timeout=10)
        model = service.load_model()
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            user_ids = combined_data['user'].n_id.cpu().numpy()
            restaurant_ids = combined_data['restaurant'].n_id.cpu().numpy()
            user_output = model(combined_data.x_dict, combined_data.edge_index_dict)['user'].cpu().numpy()
            
            for user_id in user_ids:
                scores = user_output[user_id]
                top_indices = scores.argsort()[::-1][:10]  # Get indices of the top-10 scores
                restaurant_scores = [(restaurant_ids[i], scores[i]) for i in top_indices]
                predictions[user_id] = restaurant_scores
        
        return predictions
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
   data = ingest_df(data_path=data_path)
   preprocessed_data = preprocess_data_step(data=data)
   user_restaurant_graph, user_category_graph, restaurant_mapping = create_bipartite_graphs(data=preprocessed_data)
   user_restaurant_graph, user_category_graph = add_node_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
   user_restaurant_graph, user_category_graph = add_edge_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
   combined_data, combined_graph, in_channels = create_combined_data(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, preprocessed_data=preprocessed_data)
   train_loader, valid_loader, test_loader = data_loader_step(combined_data)
   # Model initialization
   model, criterion, optimizer, lr_scheduler = model_initialization_step(in_channels=in_channels)
   # Model training
   trained_model = model_training_loop(train_loader, valid_loader, model, criterion, optimizer, lr_scheduler)
   # Model evaluation
   _, ndcg_k, _ = evaluate_model(
       model=trained_model,
       test_loader=test_loader,
       criterion=criterion
    )
   deploy_decision = deployment_trigger(ndcg_k)
   mlflow_model_deployer_step(
       trained_model,
       deploy_decision=deploy_decision,
       workers = workers,
       timeout = timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    combined_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    top_n_recommendations = predictor(service=model_deployment_service, combined_data=combined_data)