class FrequentSubgraph:
    def __init__(self, dfsCode, setOfGraphsIds, support):
        self.dfsCode = dfsCode
        self.setOfGraphsIds = setOfGraphsIds
        self.support = support
    
    def __eq__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support == other.support

    def __lt__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support < other.support

    def __gt__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support > other.support
