import pandas as pd
import torch
from steps.preprocess_data import preprocess_data_step
import networkx as nx
from typing import Tuple, Dict, Any
from steps.text_encoding import encode_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict
from torch_geometric.data import HeteroData
from zenml import step
import logging
from typing import Tuple, Dict


encoder = OneHotEncoder(sparse=False)

@step
def create_bipartite_graphs(data: pd.DataFrame) -> Tuple[nx.Graph, nx.Graph, Dict]:
    """Create user-restaurant and user-category bipartite graphs."""
    user_restaurant_graph = nx.Graph()
    user_category_graph = nx.Graph()
    
    unique_users = data['user_id'].unique()
    unique_restaurants = data['business_id'].unique()
    unique_categories = set(category for categories in data['categories'] for category in categories)

    user_restaurant_graph.add_nodes_from(unique_users, bipartite='user')
    user_restaurant_graph.add_nodes_from(unique_restaurants, bipartite='restaurant')
    user_category_graph.add_nodes_from(unique_users, bipartite='user')
    user_category_graph.add_nodes_from(unique_categories, bipartite='category')

    restaurant_mapping = {
        idx + 1: {business_id: name}
        for idx, (business_id, name) in enumerate(zip(data['business_id'], data['name']))
    }

    user_restaurant_edges = [(row['user_id'], row['business_id']) for _, row in data.iterrows()]
    user_restaurant_graph.add_edges_from(user_restaurant_edges)
    user_star_x = dict(zip(data['user_id'], data['stars_x']))
    nx.set_node_attributes(user_restaurant_graph, user_star_x, name='y')

    user_category_edges = [(row['user_id'], category) for _, row in data.iterrows() for category in row['categories']]
    user_category_graph.add_edges_from(user_category_edges)

    star_y_edges = [(row['user_id'], category, {'star_y': row['stars_y']}) for _, row in data.iterrows() for category in row['categories']]
    user_category_graph.add_edges_from(star_y_edges)

    return user_restaurant_graph, user_category_graph, restaurant_mapping

@step
def add_node_features(user_restaurant_graph: nx.Graph, user_category_graph: nx.Graph, data: pd.DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    """Add node features to the bipartite graphs."""
    """Add node features to the bipartite graphs."""
    user_review_count = dict(zip(data['user_id'], data['review_count_x']))
    restaurant_review_count = dict(zip(data['business_id'], data['review_count']))
    nx.set_node_attributes(user_restaurant_graph, user_review_count, name='review_count_user')
    nx.set_node_attributes(user_restaurant_graph, restaurant_review_count, name='review_count_restaurant')

    restaurant_city = dict(zip(data['business_id'], data['city']))
    nx.set_node_attributes(user_restaurant_graph, restaurant_city, name='city_restaurant')

    restaurant_longitude = dict(zip(data['business_id'], data['longitude']))
    restaurant_latitude = dict(zip(data['business_id'], data['latitude']))
    nx.set_node_attributes(user_restaurant_graph, restaurant_longitude, name='longitude_restaurant')
    nx.set_node_attributes(user_restaurant_graph, restaurant_latitude, name='latitude_restaurant')

    user_categories = defaultdict(list)
    for _, row in data.iterrows():
        user_categories[row['user_id']].extend(row['categories'])
    nx.set_node_attributes(user_category_graph, {user: list(set(categories)) for user, categories in user_categories.items()}, name='categories_user')

    return user_restaurant_graph, user_category_graph

@step
def add_edge_features(user_restaurant_graph: nx.Graph, user_category_graph: nx.Graph, data: pd.DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    """Add edge features to the bipartite graphs."""    
    user_restaurant_text = {(row['user_id'], row['business_id']): row['text'] for _, row in data.iterrows()}
    user_category_text = {(row['user_id'], category): row['text'] for _, row in data.iterrows() for category in row['categories']}
    nx.set_edge_attributes(user_restaurant_graph, user_restaurant_text, name='review_text')
    nx.set_edge_attributes(user_category_graph, user_category_text, name='review_text')

    user_restaurant_name = {(row['user_id'], row['business_id']): row['name'] for _, row in data.iterrows()}
    user_restaurant_categories = {(row['user_id'], row['business_id']): row['categories'] for _, row in data.iterrows()}
    nx.set_edge_attributes(user_restaurant_graph, user_restaurant_name, name='restaurant_name')
    nx.set_edge_attributes(user_restaurant_graph, user_restaurant_categories, name='restaurant_categories')

    star_x_edges = [(row['user_id'], row['business_id'], {'star_x': row['stars_x']}) for _, row in data.iterrows()]
    user_restaurant_graph.add_edges_from(star_x_edges)

    star_y_edges = [(row['user_id'], category, {'star_y': row['stars_y']}) for _, row in data.iterrows() for category in row['categories']]
    user_category_graph.add_edges_from(star_y_edges)

    return user_restaurant_graph, user_category_graph

def encode_node_features(node_data: Dict[str, Any]) -> torch.Tensor:
#def encode_node_features(node_data):
    """Encode node features using node attributes and review text."""
    node_attributes = []
    # Encode categorical attributes
    city = node_data.get('city', '')
    if city:
        city_encoding = encoder.fit_transform([[city]]).toarray().flatten()
        node_attributes.extend(city_encoding)
    #else:
       # node_attributes.extend([0] * len(encoder.categories_))

    # Encode numerical attributes
    review_count = node_data.get('review_count', 0)
    node_attributes.append(review_count)

    # Encode boolean attributes
    is_open = node_data.get('is_open', 0)
    node_attributes.append(float(is_open))

    # Encode review text
    review_text = encode_text(node_data.get('review_text', ''))
    node_attributes.extend(review_text.tolist())
    return torch.tensor(node_attributes)
          
def create_pyg_data(graph: nx.Graph, node_type1: str, node_type2: str, node_type3: str) -> HeteroData:
    data = HeteroData()

    node_type1_nodes = set([n for n, d in graph.nodes(data=True) if d.get('bipartite') == 'user'])
    node_type2_nodes = set([n for n, d in graph.nodes(data=True) if d.get('bipartite') == 'restaurant'])
    node_type3_nodes = set([n for n, d in graph.nodes(data=True) if d.get('bipartite') == 'category'])

    node_type1_mapping = {node: idx for idx, node in enumerate(node_type1_nodes)}
    node_type2_mapping = {node: idx for idx, node in enumerate(node_type2_nodes)}
    node_type3_mapping = {node: idx for idx, node in enumerate(node_type3_nodes)}

    data['user'].num_nodes = len(node_type1_nodes)
    data['restaurant'].num_nodes = len(node_type2_nodes)
    data['category'].num_nodes = len(node_type3_nodes)

    user_y = [graph.nodes[node].get('y', 0) for node in node_type1_nodes]
    data['user'].y = torch.tensor(user_y)

    for node_type, node_set, mapping in [('user', node_type1_nodes, node_type1_mapping),
                                         ('restaurant', node_type2_nodes, node_type2_mapping),
                                         ('category', node_type3_nodes, node_type3_mapping)]:
        node_features = []
        for node in node_set:
            node_data = graph.nodes[node]
            node_features.append(encode_node_features(node_data))
        data[node_type].x = torch.stack(node_features)

    edges_user_restaurant = [(node_type1_mapping[u], node_type2_mapping[v]) for u, v in graph.edges() 
                             if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'restaurant']
    edges_user_category = [(node_type1_mapping[u], node_type3_mapping[v]) for u, v in graph.edges() 
                           if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'category']

    data['user', 'to', 'restaurant'].edge_index = torch.tensor(list(zip(*edges_user_restaurant)), dtype=torch.long)
    data['user', 'to', 'category'].edge_index = torch.tensor(list(zip(*edges_user_category)), dtype=torch.long)

    edge_features_user_restaurant = [encode_text(graph.edges[u, v]['review_text']) 
                                     for u, v in graph.edges() if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'restaurant']
    edge_features_user_category = [encode_text(graph.edges[u, v]['review_text']) 
                                   for u, v in graph.edges() if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'category']

    data['user', 'to', 'restaurant'].edge_attr = torch.stack(edge_features_user_restaurant)
    data['user', 'to', 'category'].edge_attr = torch.stack(edge_features_user_category)

    user_restaurant_star_x = [graph.edges[u, v].get('star_x', 0) for u, v in graph.edges() 
                              if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'restaurant']
    user_category_star_y = [graph.edges[u, v].get('star_y', 0) for u, v in graph.edges() 
                            if graph.nodes[u]['bipartite'] == 'user' and graph.nodes[v]['bipartite'] == 'category']

    data['user', 'to', 'restaurant'].edge_weight = torch.tensor(user_restaurant_star_x)
    data['user', 'to', 'category'].edge_weight = torch.tensor(user_category_star_y)

    return data

@step
def create_combined_data(user_restaurant_graph: nx.Graph, user_category_graph: nx.Graph, preprocessed_data: pd.DataFrame) -> Tuple[HeteroData, nx.Graph, int]:
    """Create the combined HeteroData object."""
    combined_graph = nx.compose(user_restaurant_graph, user_category_graph)
    combined_data = create_pyg_data(combined_graph, 'user', 'restaurant', 'category')
    in_channels = combined_data['user'].x.size(-1)  # Assuming user features are representative
    logging.info(f"combined_data: {combined_data}")
    logging.info(f"in_channels: {in_channels}")
    return combined_data, combined_graph, in_channels
