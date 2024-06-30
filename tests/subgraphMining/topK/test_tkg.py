# To test simply use the following command:
# python -m unittest PathToPAMI/PAMI/tests/subgraphMining/topK/test_tkg.py

import unittest
from unittest.mock import mock_open, patch
from PAMI.subgraphMining.topK.tkg import TKG
import os


class TestTKG(unittest.TestCase):
    def setUp(self):
        # Mock data for the graph input file
        self.mock_graph_data = """
        t # 0
        v 0 1
        v 1 2
        v 2 3
        v 3 4
        v 4 5
        e 0 1 0
        e 1 2 0
        e 2 3 0
        e 3 4 0
        t # 1
        v 0 1
        v 1 2
        v 2 3
        v 3 4
        v 4 5
        e 0 1 0
        e 1 2 0
        e 2 3 0
        e 3 4 0
        t # 2
        v 0 1
        v 1 2
        v 2 4
        v 3 5
        v 4 6
        e 0 1 0
        e 1 2 0
        e 2 3 0
        e 3 4 0
        t # 3
        v 0 1
        v 1 3
        v 2 4
        v 3 5
        v 4 6
        e 0 1 0
        e 1 2 0
        e 2 3 0
        e 3 4 0
        t # 4
        v 0 1
        v 1 2
        v 2 3
        v 3 4
        v 4 5
        e 0 1 0
        e 1 2 0
        e 2 3 0
        e 3 4 0
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
        tkg = TKG(iFile=self.input_file, k=4)
        self.assertEqual(tkg.iFile, self.input_file)
        self.assertEqual(tkg.k, 4)

    @patch("builtins.open", new_callable=mock_open)
    def test_save(self, mock_file):
        tkg = TKG(iFile=self.input_file, k=4)
        tkg.mine()
        tkg.save(self.output_file)

        # Ensure save method writes data to the output file
        mock_file.assert_called_with(self.output_file, 'w')

    def test_get_memory_uss(self):
        tkg = TKG(iFile=self.input_file, k=4)
        tkg.mine()
        mem_uss = tkg.getMemoryUSS()
        self.assertGreater(mem_uss, 0)

    def test_get_memory_rss(self):
        tkg = TKG(iFile=self.input_file, k=4)
        tkg.mine()
        mem_rss = tkg.getMemoryRSS()
        self.assertGreater(mem_rss, 0)

    def test_get_min_support(self):
        tkg = TKG(iFile=self.input_file, k=4)
        tkg.mine()
        min_support = tkg.getMinSupport()
        self.assertEqual(min_support, 4)

    def test_get_k_subgraphs(self):
        tkg = TKG(iFile=self.input_file, k=4)
        tkg.mine()
        k_subgraphs = tkg.getKSubgraphs()
        self.assertIsInstance(k_subgraphs, str)
        self.assertGreater(len(k_subgraphs), 0)
        self.assertEqual(k_subgraphs.count('#'), 4)


if __name__ == '__main__':
    unittest.main()
