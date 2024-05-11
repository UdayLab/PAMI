from graph import Graph

class ImplicatedGraph(Graph):
    def __init__(self, implicatedGraphId, uncertainGraphInstance, uncertainGraphId):
        super().__init__()
        self.uncertainGraphId = uncertainGraphId
        self.implicatedGraphId = implicatedGraphId
        self.implicatedProbability = None
        self.createFromUncertain(uncertainGraphInstance)

    def createFromUncertain(self, uncertainGraphInstance):
        # Map original vertex IDs to new vertex objects
        vertexMap = {} 

        # Add all vertices from the uncertain graph to this graph
        for vertex in uncertainGraphInstance.getVertices():
            newVertex = self.addVertex(vertex.getLabel())
            vertexMap[vertex.getId()] = newVertex  

        # Add all edges using the new vertex objects
        for edge in uncertainGraphInstance.getEdges():
            origV1, origV2 = edge.getVertices()
            newV1 = vertexMap[origV1.getId()]
            newV2 = vertexMap[origV2.getId()]
            self.addEdge(newV1, newV2, edge.getLabel())

    def findImplicatedProbability(self, probability):
        self.implicatedProbability = probability


    def getId(self):
        return self.implicatedGraphId
    
    def getUncertainGraphId(self):
        return self.uncertainGraphId
    
    def getImplicatedProbability(self):
        return self.implicatedProbability



