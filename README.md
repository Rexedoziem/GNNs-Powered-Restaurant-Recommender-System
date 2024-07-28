# Restaurant Recommender System using GNNs and ZenML

This project uses ZenML to an simple end-to-end pipeline for a Graph Neural Network (GNN) model to recommend restaurants based on user preferences and interactions.

## Training Pipeline

The training pipeline consists of several steps: data preprocessing, GNN model training, and model evaluation.

### Data

The data for this project comes from the following sources:

- User reviews
- Restaurant metadata (e.g., cuisine type, location, etc.)
- User metadata (e.g., preferences, demographics, etc.)

The data is preprocessed to create a bipartite graph, where users and restaurants are nodes, and edges represent interactions (e.g., reviews or ratings).

### Data Preparation

The data preparation step includes:

1. **Loading and cleaning the data**: Removing duplicates, handling missing values, and ensuring data consistency.
2. **Creating the graph**: Constructing a bipartite graph with users and restaurants as nodes, and interactions as edges.
3. **Feature engineering**: Adding node and edge features such as user preferences, restaurant attributes, etc.
4. **Normalization and transformation**: Normalizing features and transforming data for the GNN model.


### Model

The model for this project is a Graph Neural Network (GNN) implemented with PyTorch Geometric. The model and training metrics are logged using MLflow for experiment tracking.

The model's hyperparameters were tuned using Optuna, achieving a balance between precision and recall on a hold-out validation set.


### Evaluation

The trained model is evaluated on a hold-out validation set, with metrics such as precision, recall, and F1-score logged to MLflow.


## Deployment Pipeline

The deployment pipeline extends the training pipeline and implements a continuous deployment workflow. It preps the input data, trains a model, and (re)deploys the recommendation server that serves the model if it meets some evaluation criteria (minimum precision and recall).

### Deployment Trigger

After the model is trained and evaluated, the deployment trigger step checks whether the newly-trained model meets the criteria set for deployment.

### Model Deployer

This step deploys the model as a service using MLflow (if deployment criteria are met).

The MLflow deployment server runs locally as a daemon process and updates with the new model if it passes the evaluation checks.

## Inference Pipeline

This project uses a Streamlit application for inference, but also includes a separate inference pipeline for testing.


### Data

- **Label accuracy**: Ensure the correctness of labels for training data.
- **Duplicates**: Confirm no duplicates in training and test data.
- **Data validation**: Integrate data validation steps into the pipeline using tools like Deep Checks.

### Model/Training

- **Cloud training**: Move training to a cloud service (e.g., AWS SageMaker) for scalability.
- **Hyperparameter tuning**: Automate hyperparameter tuning with ZenML.
- **Model validation**: Implement a model validation step.
- **Performance improvements**: Optimize training efficiency and conduct error analysis.

### Deployment

- **Production deployment**: Transition from local MLflow deployment to a production environment like Seldon or KServe.
- **Dockerization**: Containerize the application for consistent deployment.

### Monitoring

- **Input validation**: Set up input data validation.
- **User feedback**: Improve the accuracy and relevance of user feedback.
- **App performance**: Monitor application performance metrics (e.g., latency, errors).

### Orchestration

- **Automate retraining**: Use Airflow or a similar tool to automate retraining with new data.
- **Continual learning**: Implement continual learning to avoid storing raw images.

### Misc

- **Testing**: Improve testing coverage and rigor.

## Repository Structure

- `data/`: Contains datasets and data validation reports.
- `models/`: Contains trained models and related artifacts.
- `notebooks/`: Jupyter notebooks for exploration and prototyping.
- `scripts/`: Scripts for data preprocessing, training, and inference.
- `zenml_pipelines/`: ZenML pipeline definitions and configurations.
- `_assets/`: Images used in the README.
- `README.md`: Project overview and instructions.

## Running the Project

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt


2. **Set up ZenML**:
    ```bash
    zenml init
    zenml integration install pytorch mlflow tensorflow

3. **Run the training pipeline**:
    ```bash
    python zenml_pipelines/training_pipeline.py

4. **Run the deployment pipeline**:
    ```bash
    python zenml_pipelines/deployment_pipeline.py


## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

