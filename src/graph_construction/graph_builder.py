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

def add_retweet_edge(graph, user_id, tweet_id):
    """
    Adds a 'retweeted' edge from a user to a tweet.
    """
    graph.add_edge(user_id, tweet_id, edge_type='retweeted')

def add_mention_edge(graph, source_tweet_id, target_user_id):
    """
    Adds a 'mentioned' edge from a tweet to a user.
    """
    graph.add_edge(source_tweet_id, target_user_id, edge_type='mentioned')

def add_content_similarity_edge(graph, tweet_id_1, tweet_id_2, **kwargs):
    """
    Adds a 'similar_content' edge between two tweets.
    'kwargs' can include the similarity score.
    """
    graph.add_edge(tweet_id_1, tweet_id_2, edge_type='similar_content', **kwargs)

if __name__ == '__main__':
    G = create_graph()
    
    # 1. Add nodes
    add_user_node(G, 'user_A', username='Alice')
    add_user_node(G, 'user_B', username='Bob')
    add_user_node(G, 'user_C', username='Charlie')
    add_tweet_node(G, 'tweet_1', text='This is the first tweet! @user_B')
    add_tweet_node(G, 'tweet_2', text='Hello world!')
    add_tweet_node(G, 'tweet_3', text='Hello world! #testing')

    # 2. Add edges
    add_posted_edge(G, 'user_A', 'tweet_1')
    add_posted_edge(G, 'user_B', 'tweet_2')
    add_retweet_edge(G, 'user_C', 'tweet_1')
    add_mention_edge(G, 'tweet_1', 'user_B')
    add_content_similarity_edge(G, 'tweet_2', 'tweet_3', score=0.95)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("
--- Nodes ---")
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Type: {data.get('node_type', 'N/A')}")
        
    print("
--- Edges ---")
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'N/A')
        details = ''
        if edge_type == 'similar_content':
            details = f" (Score: {data.get('score', 'N/A')})"
        print(f"Edge: ({u}) -> ({v}), Type: {edge_type}{details}")

