import networkx as nx
import matplotlib.pyplot as plt

class graphDatabase:

    def __init__(self, iFile):
        self.graphs = []
        current_graph = {'vertices': [], 'edges': []}

        with open(iFile, 'r') as file:
            for line in file:
                if line.startswith('t #'):
                    if current_graph['vertices'] or current_graph['edges']:
                        self.graphs.append(current_graph)
                        current_graph = {'vertices': [], 'edges': []}
                elif line.startswith('v'):
                    _, v_id, label = line.split()
                    current_graph['vertices'].append((int(v_id), int(label)))
                elif line.startswith('e'):
                    _, v1, v2, label = line.split()
                    current_graph['edges'].append((int(v1), int(v2), int(label)))

        if current_graph['vertices'] or current_graph['edges']:
            self.graphs.append(current_graph)

    def printStats(self):
        for i, graph in enumerate(self.graphs):
            print(f"Graph {i}:")
            num_vertices = len(graph['vertices'])
            num_edges = len(graph['edges'])
            vertex_labels = set(label for _, label in graph['vertices'])
            edge_labels = set(label for _, _, label in graph['edges'])

            print(f"  Number of vertices: {num_vertices}")
            print(f"  Number of edges: {num_edges}")
            print(f"  Unique vertex labels: {vertex_labels}")
            print(f"  Unique edge labels: {edge_labels}")

if __name__ == '__main__':
    file_path = 'Chemical_340.txt'
    obj = graphDatabase(file_path)
    obj.printStats()