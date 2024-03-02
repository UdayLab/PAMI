from abc import ABC, abstractmethod
from graph import Graph
from dfs_code import DFSCode
from frequent_subgraph import FrequentSubgraph
from vertex import Vertex
from edge import Edge
from extended_edge import ExtendedEdge
from sparse_triangular_matrix import SparseTriangularMatrix
import time
import math
import matplotlib.pyplot as plt
import networkx as nx
import resource

class _gSpan(ABC):

    @abstractmethod
    def run(self, inPath, outPath, minSupport, outputSingleVertices, maxNumberOfEdges, outputGraphIds):
        """
        Run the gSpan algorithm.
        :param inPath: Path to the input file containing the graphs.
        :param outPath: Path to the output file for the frequent subgraphs.
        :param minSupport: Minimum support threshold as a fraction.
        :param outputSingleVertices: Boolean indicating whether to output single vertices as subgraphs.
        :param maxNumberOfEdges: Maximum number of edges for the subgraphs to find.
        :param outputGraphIds: Boolean indicating whether to include graph IDs in the output.
        """
        pass

    @abstractmethod
    def readGraphs(self, path):
        """
        Read graphs from a file.
        :param path: Path to the file containing the graphs.
        """
        pass

    @abstractmethod
    def writeResultToFile(self, outputPath):
        """
        Write the result of the gSpan algorithm to a file.
        :param outputPath: Path to the output file.
        """
        pass

    @abstractmethod
    def gSpan(self, graphDb, outputFrequentVertices):
        """
        The main gSpan function to find frequent subgraphs.
        :param graphDb: The database of graphs to mine.
        :param outputFrequentVertices: Boolean indicating whether to output single vertices as subgraphs.
        """
        pass


