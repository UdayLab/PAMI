class Edge:
    def __init__(self, v1, v2, edgeLabel):
        """
        Represents an edge in a graph with vertices v1 and v2 and a specified edge label.

        Attributes:
        v1(int): The index of the first vertex.
        v2(int): The index of the first vertex.
        edgeLabel(str): The label of the edge.
        """

        self.v1 = v1
        self.v2 = v2
        self.edgeLabel = edgeLabel
        self.hashCode = (v1 + 1) * 100 + (v2 + 1) * 10 + edgeLabel

    def another(self, v):
        """
        Returns the other vertex of the edge given one vertex.
        """
        return self.v2 if v == self.v1 else self.v1

    def getEdgeLabel(self):
        """
        Retrieves the label of the edge.
        """
        return self.edgeLabel

    def __hash__(self):
        """
        hashCode(int): A hash code generated based the vertices and edge label.
        """
        return self.hashCode

    def __eq__(self, other):
        """
        Checks if two edges are equal based on their vertices and edge labels.
        """
        if not isinstance(other, Edge):
            return False
        return (self.hashCode == other.hashCode and
                self.v1 == other.v1 and
                self.v2 == other.v2 and
                self.edgeLabel == other.edgeLabel)

    def __repr__(self):
        """
        Returns a string representation of the Edge object.
        """
        return f"Edge(v1: {self.v1}, v2: {self.v2}, Label: {self.edgeLabel})"
