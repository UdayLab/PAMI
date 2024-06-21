import unittest
from unittest.mock import patch, mock_open
import os
from PAMI.subgraphMining.basic.gspan import GSpan
from generate import generate_random_graphs

class TestGSpan(unittest.TestCase):

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
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        self.assertEqual(gspan.inPath, self.input_file)
        self.assertEqual(gspan.minSup, 0.5)

    def test_mine(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        gspan.mine()

        num_patterns = gspan.getFrequentSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)

    def test_save(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        gspan.mine()
        gspan.save(self.output_file)

        with open(self.output_file, 'r') as f:
            saved_data = f.read()
        self.assertGreater(len(saved_data), 0)

    def test_get_memory_uss(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        gspan.mine()
        mem_uss = gspan.getMemoryUSS()
        self.assertGreater(mem_uss, 0)

    def test_get_memory_rss(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        gspan.mine()
        mem_rss = gspan.getMemoryRSS()
        self.assertGreater(mem_rss, 0)

    def test_get_runtime(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5)
        gspan.mine()
        runtime = gspan.getRuntime()
        self.assertGreater(runtime, 0)

    def test_max_number_of_edges(self):
        self.generate_graph_data()
        gspan = GSpan(iFile=self.input_file, minSupport=0.5, outputSingleVertices=True, maxNumberOfEdges=1)
        gspan.mine()
        num_patterns = gspan.getFrequentSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)


if __name__ == '__main__':
    unittest.main()
