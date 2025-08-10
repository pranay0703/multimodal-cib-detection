import torch
import torch.optim as optim
from torch_geometric.data import Data
from src.graph_construction.graph_builder import create_graph, add_user_node, add_tweet_node, add_posted_edge
from src.modeling.gnn_model import GNNModel
from src.evaluation.evaluate import evaluate_model

def prepare_data(graph):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object,
    including train/test masks.
    """
    node_mapping = {node_id: i for i, node_id in enumerate(graph.nodes())}
    
    node_features = []
    for node_id, data in graph.nodes(data=True):
        feature_vec = torch.randn(1, 128) # Dummy features
        node_features.append(feature_vec)
    x = torch.cat(node_features, dim=0)

    edge_list = []
    for u, v in graph.edges():
        edge_list.append([node_mapping[u], node_mapping[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    y = torch.randint(0, 2, (x.size(0),))

    # Create train and test masks
    num_nodes = x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Let's use 80% for training, 20% for testing
    train_indices = torch.randperm(num_nodes)[:int(num_nodes * 0.8)]
    test_indices = torch.tensor(list(set(range(num_nodes)) - set(train_indices.tolist())))
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

def train(model, data, epochs=100):
    """ Trains the model. """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

def test(model, data):
    """ Tests the model and returns predictions and true labels. """
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, pred = out.max(dim=1)
    
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    
    return y_true, y_pred

if __name__ == '__main__':
    G = create_graph()
    for i in range(10): add_user_node(G, f'user_{i}')
    for i in range(20): add_tweet_node(G, f'tweet_{i}')
    for i in range(10): add_posted_edge(G, f'user_{i}', f'tweet_{i}')
    
    data = prepare_data(G)
    
    model = GNNModel(num_node_features=data.num_node_features, num_classes=2)
    
    print("Starting training...")
    train(model, data)
    
    print("
Training complete. Running evaluation on the test set...")
    y_true, y_pred = test(model, data)
    
    evaluate_model(y_true, y_pred)

