class FrequentSubgraph:
    def __init__(self, dfsCode, setOfGraphsIds, support):
        """

       Represents a frequent subgraph discovered during the mining process.

        :Attributes:

                dfsCode (DFSCode): The DFS code representing the subgraph.

                setOfGraphsIds (set): Set of graph IDs containing this subgraph.

                support (int): Support count indicating the frequency of occurrence of the subgraph.
        """
        self.dfsCode = dfsCode
        self.setOfGraphsIds = setOfGraphsIds
        self.support = support
    
    def __eq__(self, other):
        """
        Checks if two FrequentSubgraph instances are equal based on support.
        """
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support == other.support

    def __lt__(self, other):
        """
        Checks if this FrequentSubgraph instance has lower support than another instance.
        """
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support < other.support

    def __gt__(self, other):
        """
        Checks if this FrequentSubgraph instance has greater support than another instance.
        """
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support > other.support
