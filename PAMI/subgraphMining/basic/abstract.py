from abc import ABC, abstractmethod
from .graph import Graph
from .dfsCode import DFSCode
from .frequentSubgraph import FrequentSubgraph
from .vertex import Vertex
from .edge import Edge
from .extendedEdge import ExtendedEdge
from .sparseTriangularMatrix import SparseTriangularMatrix
import time
import math
import matplotlib.pyplot as plt
import psutil as _psutil
import os as _os


class _gSpan(ABC):

    @abstractmethod
    def mine(self):
        """
        Run the gSpan algorithm.
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
    def save(self, outputPath):
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


