class Edge:
    def __init__(self, v1, v2, edgeLabel, probability):
        self.v1 = v1
        self.v2 = v2
        self.edgeLabel = edgeLabel
        self.probability = probability

    def another(self, v):
        return self.v2 if v == self.v1 else self.v1

    def getEdgeLabel(self):
        return self.edgeLabel
    
    def getExistenceProbability(self):
        return self.probability
    
    def getVertices(self):
        return (self.v1, self.v2)
    
    def getLabel(self):
        return self.edgeLabel

    def __hash__(self):
        return hash((self.v1, self.v2, self.edgeLabel, self.probability)) 

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.v1 == other.v1 and
                self.v2 == other.v2 and
                self.edgeLabel == other.edgeLabel and
                self.probability == other.probability)

    def __repr__(self):
        return f"Edge(v1: {self.v1}, v2: {self.v2}, Label: {self.edgeLabel})"
