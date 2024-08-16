import unittest
import os
from PAMI.subgraphMining.topK.tkg import TKG
from generate import generate_random_graphs


class TestTKG(unittest.TestCase):

    def setUp(self):
        self.input_file = "test_input.txt"
        self.output_file = "test_output.txt"

    def tearDown(self):
        # Clean up the test files
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def generate_graph_data(self, num_graphs=5, num_vertices=10, num_edges=5, num_vertex_labels=5, num_edge_labels=3):
        graph_data = generate_random_graphs(num_graphs, num_vertices, num_edges, num_vertex_labels, num_edge_labels)
        with open(self.input_file, 'w') as f:
            f.write(graph_data)

    def test_initialization(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        self.assertEqual(tkg.iFile, self.input_file)
        self.assertEqual(tkg.k, 3)

    def test_mine(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()

        num_patterns = tkg.getKSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)

    def test_save(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()
        tkg.save(self.output_file)

        with open(self.output_file, 'r') as f:
            saved_data = f.read()
        self.assertGreater(len(saved_data), 0)

    def test_get_memory_uss(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()
        mem_uss = tkg.getMemoryUSS()
        self.assertGreater(mem_uss, 0)

    def test_get_memory_rss(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()
        mem_rss = tkg.getMemoryRSS()
        self.assertGreater(mem_rss, 0)

    def test_get_runtime(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()
        runtime = tkg.getRuntime()
        self.assertGreater(runtime, 0)

    def test_get_min_support(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3)
        tkg.mine()
        min_support = tkg.getMinSupport()
        self.assertGreaterEqual(min_support, 1) 

    def test_max_number_of_edges(self):
        self.generate_graph_data()
        tkg = TKG(iFile=self.input_file, k=3, maxNumberOfEdges=1)
        tkg.mine()
        num_patterns = tkg.getKSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)


if __name__ == '__main__':
    unittest.main()
