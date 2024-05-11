from edge import Edge
from vertex import Vertex

class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertexCount = 0
        self.edgeCount = 0

    def addVertex(self, vLabel):
        vertex = Vertex(self.vertexCount, vLabel)
        self.vertices[self.vertexCount] = vertex
        self.vertexCount += 1
        return vertex

    def addEdge(self, v1, v2, edgeLabel, probability):
        edge = Edge(v1, v2, edgeLabel, probability)
        self.edges[edge] = edge
        self.vertices[v1.getId()].addEdge(edge)
        self.vertices[v2.getId()].addEdge(edge)
        self.edgeCount += 1
        return edge
    
    def getVertex(self, id):
        return self.vertices[id]

    def getVertices(self):
        return self.vertices.values()

    def getEdges(self):
        return self.edges.values()

    def __repr__(self):
        return f"Graph(Vertices: {self.vertices}, Edges: {self.edges})"

    def __eq__(self, other):
        if not isinstance(other, Graph):
            return False
        return (self.vertices == other.vertices and
                self.edges == other.edges and
                self.vertexCount == other.vertexCount and
                self.edgeCount == other.edgeCount)

    def __hash__(self):
        return hash((self.vertices, self.edges, self.vertexCount, self.edgeCount))