class Edge:
    def __init__(self, v1, v2, edgeLabel):
        self.v1 = v1
        self.v2 = v2
        self.edgeLabel = edgeLabel
        self.hashCode = (v1 + 1) * 100 + (v2 + 1) * 10 + edgeLabel

    def another(self, v):
        return self.v2 if v == self.v1 else self.v1

    def getEdgeLabel(self):
        return self.edgeLabel

    def __hash__(self):
        return self.hashCode

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.hashCode == other.hashCode and
                self.v1 == other.v1 and
                self.v2 == other.v2 and
                self.edgeLabel == other.edgeLabel)

    def __repr__(self):
        return f"Edge(v1: {self.v1}, v2: {self.v2}, Label: {self.edgeLabel})"
