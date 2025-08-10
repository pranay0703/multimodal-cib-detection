import networkx as nx

def create_graph():
    """
    Creates and returns an empty NetworkX graph.
    We'll use a MultiDiGraph to allow for multiple directed edges 
    between the same two nodes (e.g., multiple retweets).
    """
    return nx.MultiDiGraph()

def add_user_node(graph, user_id, **kwargs):
    """
    Adds a user node to the graph.
    'user_id' is the unique identifier for the user.
    'kwargs' can be used to add attributes like username, profile_description, etc.
    """
    graph.add_node(user_id, node_type='user', **kwargs)

def add_tweet_node(graph, tweet_id, **kwargs):
    """
    Adds a tweet node to the graph.
    'tweet_id' is the unique identifier for the tweet.
    'kwargs' can be used to add attributes like text, created_at, etc.
    """
    graph.add_node(tweet_id, node_type='tweet', **kwargs)

def add_posted_edge(graph, user_id, tweet_id):
    """
    Adds a 'posted' edge from a user to a tweet.
    """
    graph.add_edge(user_id, tweet_id, edge_type='posted')

if __name__ == '__main__':
    G = create_graph()
    
    # Example Usage:
    # 1. Add some users and tweets
    add_user_node(G, 'user_A', username='Alice')
    add_user_node(G, 'user_B', username='Bob')
    add_tweet_node(G, 'tweet_1', text='This is the first tweet!')
    add_tweet_node(G, 'tweet_2', text='Hello world!')
    
    # 2. Add edges representing actions
    add_posted_edge(G, 'user_A', 'tweet_1')
    add_posted_edge(G, 'user_B', 'tweet_2')
    add_posted_edge(G, 'user_A', 'tweet_2') # Alice posts tweet_2 as well
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Print out the nodes and their attributes
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Type: {data['node_type']}")
        
    # Print out the edges
    for u, v, data in G.edges(data=True):
        print(f"Edge: ({u}) -> ({v}), Type: {data['edge_type']}")

