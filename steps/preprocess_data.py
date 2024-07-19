import torch
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from zenml import step
import pandas as pd
from steps.text_encoding import encode_text

# Define scaler and encoder outside the function
scaler = StandardScaler()
encoder = OneHotEncoder(sparse=False)

def scale_numerical_features(data, features, scaler):
    """Scale numerical features using StandardScaler"""
    scaled_data = data.copy()
    scaled_data[features] = scaler.fit_transform(scaled_data[features])
    return scaled_data

def encode_categorical_features(data, features, encoder):
    """Encode categorical features using OneHotEncoder"""
    encoded_data = data.copy()
    encoded_data[features] = encoder.fit_transform(encoded_data[features])
    return encoded_data

@step
def preprocess_data_step(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features."""
    
    # Handle missing values
    data = data.fillna({'text': '', 'categories': ''})
    
    # Split categories
    data['categories'] = data['categories'].str.split(', ')
    
    # Scale numerical features
    numerical_features = ['longitude', 'latitude', 'review_count_x', 'review_count']
    data = scale_numerical_features(data, numerical_features, scaler)
    
    # Encode categorical features
    categorical_features = ['city']
    data = encode_categorical_features(data, categorical_features, encoder)
    
    return data
