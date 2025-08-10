import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from src.graph_construction.graph_builder import create_graph, add_user_node, add_tweet_node, add_posted_edge, add_retweet_edge, add_mention_edge

def create_sample_graph():
    """ Creates a sample graph for visualization. """
    G = create_graph()
    add_user_node(G, 'user_A', username='Alice')
    add_user_node(G, 'user_B', username='Bob')
    add_user_node(G, 'user_C', username='Charlie')
    add_tweet_node(G, 'tweet_1', text='Hello @user_B')
    add_tweet_node(G, 'tweet_2', text='Hi back!')
    
    add_posted_edge(G, 'user_A', 'tweet_1')
    add_posted_edge(G, 'user_B', 'tweet_2')
    add_retweet_edge(G, 'user_C', 'tweet_1')
    add_mention_edge(G, 'tweet_1', 'user_B')
    return G

def main():
    st.set_page_config(layout="wide")
    st.title("Multimodal CIB Detection Dashboard")
    
    st.write("Welcome to the CIB detection dashboard.")
    
    st.header("Graph Visualization")
    
    # Create the graph
    nx_graph = create_sample_graph()

    # Create a Pyvis network
    pyvis_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    pyvis_net.from_nx(nx_graph)
    
    # Customize node appearance
    for node in pyvis_net.nodes:
        node_id = node["id"]
        node_type = nx_graph.nodes[node_id].get('node_type')
        if node_type == 'user':
            node["color"] = "#0072B2" # Blue
            node["shape"] = "dot"
            node["size"] = 25
        elif node_type == 'tweet':
            node["color"] = "#D55E00" # Orange
            node["shape"] = "square"
            node["size"] = 15

    # Generate the HTML file
    try:
        pyvis_net.save_graph("graph.html")
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=800)
    except Exception as e:
        st.error(f"An error occurred while generating the graph: {e}")

if __name__ == '__main__':
    main()

