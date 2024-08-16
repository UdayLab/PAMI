import random

def generate_random_graphs(num_graphs, num_vertices, num_edges, num_vertex_labels, num_edge_labels):
    graph_data = ""
    for i in range(num_graphs):
        graph_data += f"t # {i}\n"
        vertices = list(range(num_vertices))
        random.shuffle(vertices)
        for v in vertices:
            label = random.randint(1, num_vertex_labels)
            graph_data += f"v {v} {label}\n"
        for _ in range(num_edges):
            v1, v2 = random.sample(vertices, 2)
            e_label = random.randint(1, num_edge_labels)
            graph_data += f"e {v1} {v2} {e_label}\n"
    return graph_data
