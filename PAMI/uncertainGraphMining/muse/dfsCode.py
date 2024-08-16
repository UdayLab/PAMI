class DFSCode:
    def __init__(self):
        self.code = []

    def addEdge(self, v1, v2, v1Label, v2Label, eLabel, eProb):
        self.code.append((v1, v2, v1Label, v2Label, eLabel, eProb))

    def buildCode(self, graph):
        self.code = []
        start_node = next(iter(graph.keys()))
        self.dfs([], set(), graph, start_node, 0)
        self.code.sort()

    def dfs(self, path, visited, graph, node, depth):
        visited.add(node)
        for edge in graph[node].getEdgeList():
            neighbor = edge.v2 if edge.v1 == node else edge.v1
            if neighbor not in visited:
                if neighbor in graph:
                    self.code.append((depth, node, neighbor, graph[node].getLabel(), edge.edgeLabel, graph[neighbor].getLabel()))
                    self.dfs(path + [edge], visited, graph, neighbor, depth + 1)

    def isSubcodeOf(self, other):
        return all(e in other.code for e in self.code)

    def getEeList(self):
        """Return the list of edges in the DFS code format."""
        return [EE(*e) for e in self.code]

class EE:
    def __init__(self, depth, v1, v2, v1Label, eLabel, v2Label):
        self.v1 = v1
        self.v2 = v2
        self.v1Label = v1Label
        self.v2Label = v2Label
        self.eLabel = eLabel

    def getV1(self):
        return self.v1

    def getV2(self):
        return self.v2

    def getVLabel1(self):
        return self.v1Label

    def getVLabel2(self):
        return self.v2Label

    def getEdgeLabel(self):
        return self.eLabel
