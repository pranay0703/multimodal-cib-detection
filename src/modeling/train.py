import torch
import torch.optim as optim
from torch_geometric.data import Data
from src.graph_construction.graph_builder import create_graph, add_user_node, add_tweet_node, add_posted_edge
from src.modeling.gnn_model import GNNModel

def prepare_data(graph):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    """
    node_mapping = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    node_features = []
    for node_id, data in graph.nodes(data=True):
        if data['node_type'] == 'user':
            feature_vec = torch.randn(1, 128)
        elif data['node_type'] == 'tweet':
            feature_vec = torch.randn(1, 128)
        else:
            feature_vec = torch.zeros(1, 128)
        node_features.append(feature_vec)
    
    x = torch.cat(node_features, dim=0)

    edge_list = []
    for u, v in graph.edges():
        edge_list.append([node_mapping[u], node_mapping[v]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    y = torch.randint(0, 2, (x.size(0),))

    # Add a train mask to specify which nodes to use for training
    # Here, we'll just use all nodes for simplicity.
    train_mask = torch.ones(x.size(0), dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

def train_model(model, data, epochs=100):
    """
    Trains the GNN model.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # We only calculate loss on the training nodes
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

if __name__ == '__main__':
    # 1. Create a sample graph
    G = create_graph()
    add_user_node(G, 'user_A')
    add_user_node(G, 'user_B')
    add_user_node(G, 'user_C')
    add_tweet_node(G, 'tweet_1')
    add_posted_edge(G, 'user_A', 'tweet_1')
    add_posted_edge(G, 'user_B', 'tweet_1')
    add_posted_edge(G, 'user_C', 'tweet_1')
    
    # 2. Convert to PyTorch Geometric Data object
    data = prepare_data(G)
    
    # 3. Instantiate the GNN model
    model = GNNModel(num_node_features=data.num_node_features, num_classes=2)
    print("Instantiated GNN model:")
    print(model)

    # 4. Train the model
    print("
Starting training...")
    train_model(model, data)
    
    # 5. Evaluate the model (simple accuracy check)
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()
    print(f'
Training complete. Accuracy: {acc:.4f}')

