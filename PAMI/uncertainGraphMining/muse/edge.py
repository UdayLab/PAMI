class Edge:
    def __init__(self, v1, v2, edgeLabel, existenceProbability=1.0):
        self.v1 = v1
        self.v2 = v2
        self.edgeLabel = edgeLabel
        self.existenceProbability = existenceProbability

    def getEdgeLabel(self):
        return self.edgeLabel

    def getExistenceProbability(self):
        return self.existenceProbability

    def __hash__(self):
        return hash((self.v1, self.v2, self.edgeLabel))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.v1 == other.v1 and
                self.v2 == other.v2 and
                self.edgeLabel == other.edgeLabel)

    def __repr__(self):
        return f"Edge(v1: {self.v1}, v2: {self.v2}, Label: {self.edgeLabel})"

