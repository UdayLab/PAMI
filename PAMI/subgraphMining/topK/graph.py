from .edge import Edge
from .vertex import Vertex

class Graph:
    EMPTY_VERTEX_LIST = []
    EMPTY_INTEGER_ARRAY = []

    def __init__(self, id, vMap=None, dfsCode=None):
        self.vMap = {}
        self.id = id
        if vMap is not None:
            self.vMap = vMap
        elif dfsCode is not None:
            for ee in dfsCode.getEeList():
                v1, v2, v1Label, v2Label, eLabel = ee.getV1(), ee.getV2(), ee.getVLabel1(), ee.getVLabel2(), ee.getEdgeLabel()
                
                e = Edge(v1, v2, eLabel)
                if v1 not in self.vMap:
                    self.vMap[v1] = Vertex(v1, v1Label)
                if v2 not in self.vMap:
                    self.vMap[v2] = Vertex(v2, v2Label)

                self.vMap[v1].addEdge(e)
                self.vMap[v2].addEdge(e)

            self.id = -1

        self.vertices = []
        self.neighborCache = {}
        self.mapLabelToVertexIds = {}
        self.edgeCount = 0

        self.precalculateVertexList()
        self.precalculateVertexNeighbors()
        self.precalculateLabelsToVertices()

    def getId(self):
        return self.id

    def removeInfrequentLabel(self, label):
        toRemove = [key for key, vertex in self.vMap.items() if vertex.getLabel() == label]
        for key in toRemove:
            del self.vMap[key]

        for vertex in self.vMap.values():
            edgesToRemove = [edge for edge in vertex.getEdgeList() 
                               if edge.v1 not in self.vMap or edge.v2 not in self.vMap]

            for edge in edgesToRemove:
                vertex.getEdgeList().remove(edge)

    def precalculateVertexNeighbors(self):
        self.neighborCache = {}
        self.edgeCount = 0

        for vertexId, vertex in self.vMap.items():
            neighbors = []

            for edge in vertex.getEdgeList():
                neighborVertex = self.vMap[edge.another(vertexId)]
                neighbors.append(neighborVertex)

            neighbors.sort(key=lambda x: x.id)

            self.neighborCache[vertexId] = neighbors
            self.edgeCount += len(neighbors)

        self.edgeCount //= 2    
    
    def precalculateVertexList(self):
        self.vertices = []
        for _, vertex in self.vMap.items():
            self.vertices.append(vertex)

    def precalculateLabelsToVertices(self):
        self.mapLabelToVertexIds = {}
        for vertex in self.vertices:
            label = vertex.getLabel()
            if label not in self.mapLabelToVertexIds:
                sameIds = [v.getId() for v in self.vertices if v.getLabel() == label]
                self.mapLabelToVertexIds[label] = sameIds

    def findAllWithLabel(self, targetLabel):
        if targetLabel in self.mapLabelToVertexIds:
            return self.mapLabelToVertexIds[targetLabel]
        else:
            return []
        
    def getAllNeighbors(self, v):
        try:
            neighbors = self.neighborCache[v]
        except KeyError:
            neighbors = []
        return neighbors
    
    def getVLabel(self, v):
        return self.vMap[v].getLabel()
    
    def getEdgeLabel(self, v1, v2):
        for e in self.vMap.get(v1).getEdgeList():
            if e.v1 == v1 and e.v2 == v2:
                return e.getEdgeLabel()
            elif e.v1 == v2 and e.v2 == v1:
                return e.getEdgeLabel()
        return -1
    

    def getEdge(self, v1, v2):
        for e in self.vMap.get(v1).getEdgeList():
            if e.v1 == v1 and e.v2 == v2:
                return e
            elif e.v1 == v2 and e.v2 == v1:
                return e
        return None
    
    def getNonPrecalculatedAllVertices(self):
        return list(self.vMap.values())
    
    def isNeighboring(self, v1, v2):
        neighborsOfV1 = self.neighborCache.get(v1, [])
        low = 0
        high = len(neighborsOfV1) - 1

        while high >= low:
            middle = (low + high) // 2
            val = neighborsOfV1[middle].id
            if val == v2:
                return True
            if val < v2:
                low = middle + 1
            if val > v2:
                high = middle - 1
        return False

    def getAllVertices(self):
        return self.vertices
    
    def getEdgeCount(self):
        return self.edgeCount
