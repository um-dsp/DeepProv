import pickle
import sys
from Graph_Extraction import GenericInferenceGraph  

def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph

# Example usage
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_graph.py <path_to_graph_pickle>")
        sys.exit(1)
    graph_path = sys.argv[1]
    graph = load_graph(graph_path)
    print(graph)
