from uncertainGraph import UncertainGraph
from implicatedGraph import ImplicatedGraph
import networkx as nx
import matplotlib.pyplot as plt


def visualizeGraphs(uncertainGraph):
    """
    Visualizes all possible implicated graphs of an uncertain graph.
    """
    # Generate all possible implicated graphs
    uncertainGraph.generateImplicatedGraphs()
    implicatedGraphs = uncertainGraph.getImplicatedGraphs()
    
    # Create a figure for the plots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot each implicated graph
    for i, implicatedGraph in enumerate(implicatedGraphs):
        # Create a subplot for the graph
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_title(f"Implicated Graph {implicatedGraph.getId()}")
        
        # Create a networkx graph
        nxGraph = nx.Graph()
        for vertex in implicatedGraph.getVertices():
            nxGraph.add_node(vertex.getId(), label=vertex.getLabel())
        for edge in implicatedGraph.getEdges():
            v1, v2 = edge.getVertices()
            nxGraph.add_edge(v1.getId(), v2.getId(), label=edge.getLabel())
        
        # Draw the graph
        pos = nx.spring_layout(nxGraph)
        nx.draw(nxGraph, pos, ax=ax, with_labels=True)
        edge_labels = nx.get_edge_attributes(nxGraph, 'label')
        nx.draw_networkx_edge_labels(nxGraph, pos, edge_labels=edge_labels, ax=ax)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create an uncertain graph
    uncertainGraph = UncertainGraph(1)
    v1 = uncertainGraph.addVertex('A')
    v2 = uncertainGraph.addVertex('B')
    v3 = uncertainGraph.addVertex('C')
    v4 = uncertainGraph.addVertex('D')
    uncertainGraph.addEdge(v1, v2, 'AB', 0.5)
    uncertainGraph.addEdge(v1, v3, 'AC', 0.3)
    uncertainGraph.addEdge(v2, v3, 'BC', 0.7)
    uncertainGraph.addEdge(v3, v4, 'CD', 0.9)
    
    # Visualize all possible implicated graphs
    visualizeGraphs(uncertainGraph)