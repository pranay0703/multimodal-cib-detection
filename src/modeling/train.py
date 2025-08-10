import torch
from torch_geometric.data import Data
from src.graph_construction.graph_builder import create_graph, add_user_node, add_tweet_node, add_posted_edge
from src.modeling.gnn_model import GNNModel
# Note: In a real scenario, you'd use the feature extraction functions.
# For this example, we'll create dummy features.

def prepare_data(graph):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    This involves mapping node IDs to indices and extracting features.
    """
    # Create a mapping from node ID to a continuous index
    node_mapping = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    # --- Feature Engineering ---
    # In a real implementation, you would use the functions from
    # feature_extraction.py to get real embeddings.
    # Here, we'll use dummy features for demonstration.
    
    # Assuming user nodes have a 50-dim feature vector and tweets have a 768-dim (like BERT)
    # We will need to unify feature sizes later, e.g., by padding or projection.
    # For now, let's assume a unified feature size of 128 for simplicity.
    
    node_features = []
    for node_id, data in graph.nodes(data=True):
        if data['node_type'] == 'user':
            # Dummy user features
            feature_vec = torch.randn(1, 128)
        elif data['node_type'] == 'tweet':
            # Dummy tweet features
            feature_vec = torch.randn(1, 128)
        else:
            feature_vec = torch.zeros(1, 128)
        node_features.append(feature_vec)
    
    x = torch.cat(node_features, dim=0)

    # --- Edge Index ---
    # PyG requires edge_index in a specific format: [2, num_edges]
    edge_list = []
    for u, v in graph.edges():
        edge_list.append([node_mapping[u], node_mapping[v]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # --- Labels (dummy) ---
    # We'll assign dummy labels (0 for authentic, 1 for inauthentic) to each node
    y = torch.randint(0, 2, (x.size(0),))

    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == '__main__':
    # 1. Create a sample graph
    G = create_graph()
    add_user_node(G, 'user_A')
    add_user_node(G, 'user_B')
    add_tweet_node(G, 'tweet_1')
    add_posted_edge(G, 'user_A', 'tweet_1')
    add_posted_edge(G, 'user_B', 'tweet_1')
    
    print("Created a sample NetworkX graph.")
    
    # 2. Convert to PyTorch Geometric Data object
    data = prepare_data(G)
    print("Converted graph to PyG Data object:")
    print(data)

    # 3. Instantiate the GNN model
    # The number of features must match the dummy features we created (128)
    # Number of classes is 2 (authentic/inauthentic)
    model = GNNModel(num_node_features=128, num_classes=2)
    print("
Instantiated GNN model:")
    print(model)

    # 4. Perform a forward pass (a single step of training)
    output = model(data)
    print("
Output from a single forward pass (log probabilities):")
    print(output)
    print(f"Output shape: {output.shape}") # Should be [num_nodes, num_classes]

