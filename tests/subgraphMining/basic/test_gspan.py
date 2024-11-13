# To test simply use the following command:
# python -m unittest PathToPAMI/PAMI/tests/subgraphMining/basic/test_gspan.py

import unittest
from unittest.mock import mock_open, patch
from PAMI.subgraphMining.basic.gspan import GSpan
import os

class TestGSpan(unittest.TestCase):

    def setUp(self):
        # Mock data for the graph input file
        self.mock_graph_data = """
        t # 0
        v 0 1
        v 1 2
        e 0 1 0
        t # 1
        v 0 1
        v 1 2
        e 0 1 0
        t # 2
        v 0 1
        v 1 3
        e 0 1 0
        t # 3
        v 0 2
        v 1 3
        e 0 1 0
        t # 4
        v 0 1
        v 1 4
        e 0 1 0
        """
        self.input_file = "test_input.txt"
        self.output_file = "test_output.txt"

        # Write mock data to input file
        with open(self.input_file, 'w') as f:
            f.write(self.mock_graph_data)

    def tearDown(self):
        # Clean up the test files
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    @patch("builtins.open", new_callable=mock_open, read_data="t # 0\nv 0 1\nv 1 2\ne 0 1 0\n")
    def test_initialization(self, mock_file):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3)
        self.assertEqual(gspan.inPath, self.input_file)
        self.assertEqual(gspan.minSup, 0.3)

    def test_mine(self):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3)
        gspan.mine()

        num_patterns = gspan.getFrequentSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)
        # Expected number of subgraphs is 4
        self.assertEqual(num_patterns.count('#'), 4)

    @patch("builtins.open", new_callable=mock_open)
    def test_save(self, mock_file):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3)
        gspan.mine()
        gspan.save(self.output_file)

        # Ensure save method writes data to the output file
        mock_file.assert_called_with(self.output_file, 'w')

    def test_get_memory_uss(self):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3)
        gspan.mine()
        mem_uss = gspan.getMemoryUSS()
        self.assertGreater(mem_uss, 0)

    def test_get_memory_rss(self):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3)
        gspan.mine()
        mem_rss = gspan.getMemoryRSS()
        self.assertGreater(mem_rss, 0)

    def test_edge_only_output(self):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3, outputSingleVertices=False)
        gspan.mine()
        num_patterns = gspan.getFrequentSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)
        # Expected number of subgraphs is 1
        self.assertEqual(num_patterns.count('#'), 1)

    def test_max_number_of_edges(self):
        gspan = GSpan(iFile=self.input_file, minSupport=0.3, outputSingleVertices=True, maxNumberOfEdges=1)
        gspan.mine()
        num_patterns = gspan.getFrequentSubgraphs()
        self.assertGreater(num_patterns.count('#'), 0)
        # Expected number of subgraphs is 3 given max uses less than
        self.assertEqual(num_patterns.count('#'), 3)



if __name__ == '__main__':
    unittest.main()
