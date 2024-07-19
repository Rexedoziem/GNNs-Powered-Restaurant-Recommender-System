GNN-based Restaurant Recommender System
Table of Contents

Introduction
Project Overview
Installation
Project Structure
Usage
Model Architecture
Data Pipeline
Training
Evaluation
Deployment
Contributing
License

Introduction
This project implements a restaurant recommender system using Graph Neural Networks (GNNs) and ZenML. The system leverages the power of GNNs to capture complex relationships between users, restaurants, and various features to provide personalized restaurant recommendations.
Project Overview
The recommender system uses a GNN to learn embeddings for users and restaurants based on their interactions and features. These embeddings are then used to predict user preferences for restaurants they haven't visited yet. The entire pipeline, from data ingestion to model deployment, is managed using ZenML, ensuring reproducibility and scalability.
Key features:

Graph-based representation of user-restaurant interactions
GNN model for learning user and restaurant embeddings
ZenML pipeline for end-to-end ML workflow management
Scalable architecture for handling large datasets
Personalized restaurant recommendations based on user history and preferences

Installation
To set up the project, follow these steps:

Clone the repository:
