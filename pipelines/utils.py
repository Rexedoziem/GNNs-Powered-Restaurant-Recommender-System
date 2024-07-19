import logging
import pandas as pd
import torch
from steps.preprocess_data import preprocess_data_step
from steps.create_bipartite_graph import create_bipartite_graphs, add_node_features, add_edge_features, create_combined_data


def get_data_for_test():
    try:
        # Load data
        df = pd.read_csv('C:/Users/HP/Desktop/RESTAURANT_COMPONENTS/Data/subset_review_merged1(2).csv')
        
        # Sample 20% of the data
        sample_size = int(0.2 * len(df))
        df = df.sample(n=sample_size, random_state=42)  # random_state for reproducibility
        
        preprocessed_data = preprocess_data_step(df)
        
        # Create bipartite graphs
        user_restaurant_graph, user_category_graph, restaurant_mapping = create_bipartite_graphs(preprocessed_data)
        # Add the node features
        user_restaurant_graph, user_category_graph = add_node_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
        # Add the edge features
        user_restaurant_graph, user_category_graph = add_edge_features(user_restaurant_graph=user_restaurant_graph, user_category_graph=user_category_graph, data=preprocessed_data)
    
        # Create the combined HeteroData object
        combined_data, _ = create_combined_data(user_restaurant_graph, user_category_graph, preprocessed_data)
        
        # Save the combined data object
        torch.save(combined_data, 'combined_data.pt')
        
        # Convert to dictionary for easy viewing and return
        result = combined_data.to_dict()
        return result
    except Exception as e:
        logging.error(e)
        raise e