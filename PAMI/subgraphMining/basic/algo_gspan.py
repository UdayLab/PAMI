from PAMI.subgraphMining.basic import abstract as _ab
# import abstract as _ab

class Gspan(_ab._gSpan):

    eliminate_infrequent_vertices = True
    eliminate_infrequent_vertex_pairs = True
    eliminate_infrequent_edge_labels = True
    edge_count_pruning = True

    def __init__(self) -> None:
        self.minSup = 0
        self.frequentSubgraphs = []
        self.runtime = 0
        self.maxMem = 0
        self.graphCount = 0
        self.patternCount = 0
        self.frequentVertexLabels = []
        self.infrequentVerticesRemovedCount = 0
        self.infrequentVertexPairsRemoved = 0
        self.edgeRemovedByLabel = 0
        self.eliminatedWithMaxSize = 0
        self.emptyGraphsRemoved = 0
        self.pruneByEdgeCount = 0
        self.maxNumberOfEdges = float('inf')
        self.outputGraphIds = True


    def run(self, inPath, outPath, minSupport, outputSingleVertices, maxNumberOfEdges, outputGraphIds):
        if maxNumberOfEdges <= 0:
            return
    
        mem1 = _ab.resource.getrusage(_ab.resource.RUSAGE_SELF).ru_maxrss

        self.maxNumberOfEdges = maxNumberOfEdges
        self.outputGraphIds = outputGraphIds

        self.infrequentVertexPairsRemovedCount = 0
        self.infrequentVerticesRemovedCount = 0
        self.edgeRemovedByLabel = 0
        self.eliminatedWithMaxSize = 0
        self.emptyGraphsRemoved = 0
        self.pruneByEdgeCount = 0

        self.frequentSubgraphs = []

        self.patternCount = 0

        # Record the start time
        t1 = _ab.time.time()

        # Read graphs
        graphDb = self.readGraphs(inPath)

        # Calculate minimum support as a number of graphs
        self.minSup = _ab.math.ceil(minSupport * len(graphDb))

        # Mining
        self.gSpan(graphDb, outputSingleVertices)

        # Output
        self.writeResultToFile(outPath)

        t2 = _ab.time.time()

        self.runtime = (t2 - t1)

        self.patternCount = len(self.frequentSubgraphs)

        mem2 = _ab.resource.getrusage(_ab.resource.RUSAGE_SELF).ru_maxrss

        self.maxMem = (mem2 - mem1)/1024



    def writeResultToFile(self, outputPath):
        with open(outputPath, 'w') as bw:
            i = 0
            for subgraph in self.frequentSubgraphs:
                sb = []

                dfsCode = subgraph.dfsCode
                sb.append(f"t # {i} * {subgraph.support}\n")
                if dfsCode.size == 1:
                    ee = dfsCode.getEeList()[0]
                    if ee.edgeLabel == -1:
                        sb.append(f"v 0 {ee.vLabel1}\n")
                    else:
                        sb.append(f"v 0 {ee.vLabel1}\n")
                        sb.append(f"v 1 {ee.vLabel2}\n")
                        sb.append(f"e 0 1 {ee.edgeLabel}\n")
                else:
                    vLabels = dfsCode.getAllVLabels()
                    for j, vLabel in enumerate(vLabels):
                        sb.append(f"v {j} {vLabel}\n")
                    for ee in dfsCode.getEeList():
                        sb.append(f"e {ee.v1} {ee.v2} {ee.edgeLabel}\n")

                if self.outputGraphIds:
                    sb.append("x " + " ".join(str(id) for id in subgraph.setOfGraphsIds))

                sb.append("\n\n")
                bw.write("".join(sb))
                i += 1


    def readGraphs(self, path):
        with open(path, 'r') as br:
            graphDatabase = []
            vMap = {}
            gId = None

            for line in br:
                line = line.strip()

                if line.startswith("t"):
                    if vMap:  # If vMap is not empty, it means a graph was read
                        graphDatabase.append(_ab.Graph(gId, vMap))
                        vMap = {}  # Reset for the next graph

                    gId = int(line.split(" ")[2]) 

                elif line.startswith("v"):
                    items = line.split(" ")
                    vId = int(items[1])
                    vLabel = int(items[2])
                    vMap[vId] = _ab.Vertex(vId, vLabel)

                elif line.startswith("e"):
                    items = line.split(" ")
                    v1 = int(items[1])
                    v2 = int(items[2])
                    eLabel = int(items[3])
                    e = _ab.Edge(v1, v2, eLabel)
                    vMap[v1].addEdge(e)
                    vMap[v2].addEdge(e)

            if vMap:
                graphDatabase.append(_ab.Graph(gId, vMap))

        self.graphCount = len(graphDatabase)
        return graphDatabase


    def subgraphIsomorphisms(self, c: _ab.DFSCode, g: _ab.Graph):
        isoms = []    
        startLabel = c.getEeList()[0].getVLabel1()
        for vId in g.findAllWithLabel(startLabel):
            hMap = {}
            hMap[0] = vId
            isoms.append(hMap)
        for ee in c.getEeList():
            v1, v2, v2Label, eLabel = ee.getV1(), ee.getV2(), ee.getVLabel2(), ee.getEdgeLabel() 
            updateIsoms = []

            for iso in isoms:
                mappedV1 = iso.get(v1)
                if v1 < v2:
                    mappedVertices = list(iso.values())
                    for mappedV2 in g.getAllNeighbors(mappedV1):
                        if (v2Label == mappedV2.getLabel() and
                            mappedV2.getId() not in mappedVertices and
                            eLabel == g.getEdgeLabel(mappedV1, mappedV2.getId())):

                            tempM = iso.copy()
                            tempM[v2] = mappedV2.getId()

                            updateIsoms.append(tempM)

                else:
                    mappedV2 = iso.get(v2)
                    if g.isNeighboring(mappedV1, mappedV2) and eLabel == g.getEdgeLabel(mappedV1, mappedV2):
                        updateIsoms.append(iso)

            isoms = updateIsoms
        return isoms


    def rightMostPathExtensionsFromSingle(self, c: _ab.DFSCode, g: _ab.Graph):
        gid = g.getId()
        extensions = {}

        if c.isEmpty():
            for vertex in g.vertices:
                for e in vertex.getEdgeList():
                    v1Label = g.getVLabel(e.v1)
                    v2Label = g.getVLabel(e.v2)
                    if v1Label < v2Label:
                        ee1 = _ab.ExtendedEdge(0, 1, v1Label, v2Label, e.getEdgeLabel())
                    else:
                        ee1 = _ab.ExtendedEdge(0, 1, v2Label, v1Label, e.getEdgeLabel())

                    setOfGraphIds = extensions.get(ee1, set())
                    setOfGraphIds.add(gid)
                    extensions[ee1] = setOfGraphIds
        else:
            rightMost = c.getRightMost()
            isoms = self.subgraphIsomorphisms(c, g)

            for isom in isoms:
                invertedIsom = {v: k for k, v in isom.items()}
                mappedRm = isom[rightMost]
                mappedRmLabel = g.getVLabel(mappedRm)
                for x in g.getAllNeighbors(mappedRm):
                    invertedX = invertedIsom.get(x.getId())
                    if invertedX is not None and c.onRightMostPath(invertedX) and c.notPreOfRm(invertedX) and not c.containEdge(rightMost, invertedX):
                        ee = _ab.ExtendedEdge(rightMost, invertedX, mappedRmLabel, x.getLabel(), g.getEdgeLabel(mappedRm, x.getId()))
                        extensions.setdefault(ee, set()).add(gid)

                mappedVertices = set(isom.values())
                for v in c.getRightMostPath():
                    mappedV = isom[v]
                    mappedVLabel = g.getVLabel(mappedV)
                    for x in g.getAllNeighbors(mappedV):
                        if x.getId() not in mappedVertices:
                            ee = _ab.ExtendedEdge(v, rightMost + 1, mappedVLabel, x.getLabel(), g.getEdgeLabel(mappedV, x.getId()))
                            extensions.setdefault(ee, set()).add(gid)

        return extensions


    def rightMostPathExtensions(self, c: _ab.DFSCode, graphDb, graphIds):
        extensions = {}
        if c.isEmpty():
            for id in graphIds:
                g = graphDb[id]
                if Gspan.edge_count_pruning and c.size >= g.getEdgeCount():
                    self.pruneByEdgeCount += 1
                    continue
                for v in g.vertices:
                    for e in v.getEdgeList():
                        v1L = g.getVLabel(e.v1)
                        v2L = g.getVLabel(e.v2)
                        if v1L < v2L:
                            ee1 = _ab.ExtendedEdge(0, 1, v1L, v2L, e.getEdgeLabel())
                        else:
                            ee1 = _ab.ExtendedEdge(0, 1, v2L, v1L, e.getEdgeLabel())
                                                
                        setOfGraphIds = extensions.get(ee1, set())
                        setOfGraphIds.add(id)
                        extensions[ee1] = setOfGraphIds
        else:
            rightMost = c.getRightMost()
            for id in graphIds:
                g = graphDb[id]
                if Gspan.edge_count_pruning and c.size >= g.getEdgeCount():
                    self.pruneByEdgeCount += 1
                    continue
                isoms = self.subgraphIsomorphisms(c, g)
                for isom in isoms:
                    invertedIsom = {}
                    for key, value in isom.items():
                        invertedIsom[value] = key
                    mappedRM = isom.get(rightMost)
                    mappedRMLabel = g.getVLabel(mappedRM)
                    for x in g.getAllNeighbors(mappedRM):
                        invertedX = invertedIsom.get(x.getId())
                        if invertedX is not None and c.onRightMostPath(invertedX) and \
                        c.notPreOfRm(invertedX) and not c.containEdge(rightMost, invertedX):
                            
                            ee = _ab.ExtendedEdge(rightMost, invertedX, mappedRMLabel, x.getLabel(),
                                            g.getEdgeLabel(mappedRM, x.getId()))

                            if ee not in extensions:
                                extensions[ee] = set()
                            extensions[ee].add(g.getId())

                    mappedVertices = isom.values()
                    for v in c.getRightMostPath():
                        mappedV = isom[v]
                        mappedVLabel = g.getVLabel(mappedV)
                        for x in g.getAllNeighbors(mappedV):
                            if x.getId() not in mappedVertices:
                                ee = _ab.ExtendedEdge(v, rightMost + 1, mappedVLabel, x.getLabel(),
                                                g.getEdgeLabel(mappedV, x.getId()))

                                if ee not in extensions:
                                    extensions[ee] = set()
                                extensions[ee].add(g.getId())
        return extensions


    def gspanDFS(self, c: _ab.DFSCode, graphDb, subgraphId):
        if c.size == self.maxNumberOfEdges - 1:
            return
        extensions = self.rightMostPathExtensions(c, graphDb, subgraphId)

        for extension, newGraphIds in extensions.items():
            sup = len(newGraphIds)
            
            if (sup >= self.minSup):
                newC = c.copy()
                newC.add(extension)
                
                if (self.isCanonical(newC)):
                    subgraph = _ab.FrequentSubgraph(newC, newGraphIds, sup)
                    self.frequentSubgraphs.append(subgraph)

                    self.gspanDFS(newC, graphDb, newGraphIds)


    def isCanonical(self, c: _ab.DFSCode):
        canC = _ab.DFSCode()
        for i in range(c.size):
            extensions = self.rightMostPathExtensionsFromSingle(canC, _ab.Graph(c))
            minEe = None
            for ee in extensions.keys():
                if minEe is None or ee.smallerThan(minEe):
                    minEe = ee

            if minEe is not None and minEe.smallerThan(c.getAt(i)):
                return False
            
            if minEe is not None:
                canC.add(minEe)
        return True
    

    def gSpan(self, graphDb, outputFrequentVertices):
        if outputFrequentVertices or Gspan.eliminate_infrequent_vertices:
            self.findAllOnlyOneVertex(graphDb, outputFrequentVertices)

        for g in graphDb:
            g.precalculateVertexList()

        if Gspan.eliminate_infrequent_vertex_pairs or Gspan.eliminate_infrequent_edge_labels:
            self.removeInfrequentVertexPairs(graphDb)

        graphIds = set()
        for i, g in enumerate(graphDb):
            if g.vertices is not None and len(g.vertices) != 0:
                if self.infrequentVerticesRemovedCount > 0:
                    g.precalculateVertexList()

                graphIds.add(i)
                g.precalculateVertexNeighbors()
                g.precalculateLabelsToVertices()
            else:
                self.emptyGraphsRemoved += 1

        if len(self.frequentVertexLabels) != 0:
            self.gspanDFS(_ab.DFSCode(), graphDb, graphIds)


    class Pair:
        def __init__(self, x, y):
            if x < y:
                self.x = x
                self.y = y
            else:
                self.x = y
                self.y = x

        def __eq__(self, other):
            if isinstance(other, Gspan.Pair):
                return self.x == other.x and self.y == other.y
            return False

        def __hash__(self):
            return self.x + 100 * self.y


    def findAllOnlyOneVertex(self, graphDb, outputFrequentVertices):
        self.frequentVertexLabels = []
        labelM = {} 
        for g in graphDb:
            for v in g.getNonPrecalculatedAllVertices():
                if v.getEdgeList():
                    vLabel = v.getLabel()
                    labelM.setdefault(vLabel, set()).add(g.getId())
        for label, tempSupG in labelM.items():                
            sup = len(tempSupG)
            if sup >= self.minSup:
                self.frequentVertexLabels.append(label)
                if outputFrequentVertices:
                    tempD = _ab.DFSCode()
                    tempD.add(_ab.ExtendedEdge(0, 0, label, label, -1))
                    self.frequentSubgraphs.append(_ab.FrequentSubgraph(tempD, tempSupG, sup))
            elif Gspan.eliminate_infrequent_vertices:
                for graphId in tempSupG:
                    g = graphDb[graphId]
                    g.removeInfrequentLabel(label)
                    self.infrequentVerticesRemovedCount += 1


    #TODO: This method needs some correction, it has bugs
    def removeInfrequentVertexPairs(self, graphDb):
        if Gspan.eliminate_infrequent_edge_labels:
            matrix = _ab.SparseTriangularMatrix()
            alreadySeenPair = set()

        if Gspan.eliminate_infrequent_edge_labels:
            mapEdgeLabelToSupport = {}
            alreadySeenEdgeLabel = set()

        for g in graphDb:
            vertices = g.getAllVertices()

            for v1 in vertices:
                labelV1 = v1.getLabel()

                for edge in v1.getEdgeList():
                    v2 = edge.another(v1.getId())
                    labelV2 = g.getVLabel(v2)

                    if Gspan.eliminate_infrequent_edge_labels:
                        pair = self.Pair(labelV1, labelV2)
                        if pair not in alreadySeenPair:
                            matrix.incrementCount(labelV1, labelV2)
                            alreadySeenPair.add(pair)

                    if Gspan.eliminate_infrequent_edge_labels:
                        edgeLabel = edge.getEdgeLabel()
                        if edgeLabel not in alreadySeenEdgeLabel:
                            alreadySeenEdgeLabel.add(edgeLabel)
                            edgeSupport = mapEdgeLabelToSupport.get(edgeLabel, 0)
                            mapEdgeLabelToSupport[edgeLabel] = edgeSupport + 1

            if Gspan.eliminate_infrequent_vertex_pairs:
                alreadySeenPair.clear()
            if Gspan.eliminate_infrequent_edge_labels:
                alreadySeenEdgeLabel.clear()

        if Gspan.eliminate_infrequent_vertex_pairs:
            matrix.removeInfrequentEntriesFromMatrix(self.minSup)

        if Gspan.eliminate_infrequent_vertex_pairs or Gspan.eliminate_infrequent_edge_labels:
            for g in graphDb:
                vertices = g.getAllVertices()

                for v1 in vertices:
                    iterEdges = iter(v1.getEdgeList())
                    for edge in iterEdges:
                        v2 = edge.another(v1.getId())
                        labelV2 = g.getVLabel(v2)
                        count = matrix.getSupportForItems(v1.getLabel(), labelV2)

                        if Gspan.eliminate_infrequent_vertex_pairs and count < self.minSup:
                            v1.removeEdge(edge)
                            self.infrequentVertexPairsRemoved += 1

                        elif Gspan.eliminate_infrequent_edge_labels and \
                                mapEdgeLabelToSupport.get(edge.getEdgeLabel(), 0) < self.minSup:
                            v1.removeEdge(edge)
                            self.edgeRemovedByLabel += 1
    

    def visualizeSubgraphs(self, outputPath):
        with open(outputPath, 'r') as file:
            lines = file.readlines()

        currentGraph = None
        graphs = []
        vertexLabels = {}  
        edgeLabels = {} 

        for line in lines:
            if line.startswith('t #'):
                if currentGraph is not None:
                    graphs.append((currentGraph, vertexLabels, edgeLabels))
                currentGraph = _ab.nx.Graph()
                vertexLabels = {}
                edgeLabels = {}
            elif line.startswith('v'):
                _, vertexId, label = line.split()
                currentGraph.add_node(int(vertexId))
                vertexLabels[int(vertexId)] = label
            elif line.startswith('e'):
                _, source, target, label = line.split()
                currentGraph.add_edge(int(source), int(target))
                edgeLabels[(int(source), int(target))] = label

        if currentGraph is not None:
            graphs.append((currentGraph, vertexLabels, edgeLabels))

        nRows = int(len(graphs) ** 0.5)
        nCols = (len(graphs) // nRows) + (len(graphs) % nRows > 0)

        _ab.plt.figure(figsize=(nCols * 4, nRows * 4))  

        for i, (graph, vertexLabels, edgeLabels) in enumerate(graphs):
            ax = _ab.plt.subplot(nRows, nCols, i + 1)
            pos = _ab.nx.spring_layout(graph)
            _ab.nx.draw(graph, pos, labels=vertexLabels, ax=ax, with_labels=True, nodeColor='lightblue', 
                    nodeSize=500, fontSize=10, fontWeight='bold')
            _ab.nx.drawNetworkxEdgeLabels(graph, pos, edgeLabels=edgeLabels, ax=ax, fontColor='black')
            ax.setTitle(f"Frequent Subgraph {i + 1}")

        _ab.plt.tightLayout()
        _ab.plt.show()




# gspanInstance = Gspan()

# inputFilePath = 'chemical_340.txt'

# outputFilePath = 'output_c340.txt'

# minSupport = 0.5
# outputSingleVertices = True  
# maxNumberOfEdges = 5
# outputGraphIds = True 

# gspanInstance.runAlgorithm(inputFilePath, outputFilePath, minSupport, 
#                              outputSingleVertices, maxNumberOfEdges, 
#                              outputGraphIds)

# print("Runtime:", gspanInstance.runtime, "seconds")
# print("Memory usage:", gspanInstance.maxmem, "MB")
# print("Number of frequent subgraphs found:", gspanInstance.patternCount)

# gspanInstance.visualizeSubgraphs(outputFilePath)
