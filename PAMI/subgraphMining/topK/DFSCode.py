import pickle

class DfsCode:
    def __init__(self):
        """
        Initializes the DFSCode object with default values.
        """
        self.rightMost = -1
        self.size = 0
        self.rightMostPath = []  
        self.eeList = []

    def copy(self):
        """
        Creates a deep copy of the DFSCode object.
        """
        return pickle.loads(pickle.dumps(self))

    def notPreOfRm(self, v):
        """
            Checks if a given vertex is not the second-to-last element on the rightmost path.

            Args:
                v (int): The vertex to check.

            Returns:
                bool: True if the vertex is not the second-to-last element on the rightmost path, False otherwise.
            """
        if len(self.rightMostPath) <= 1:
            return True
        return v != self.rightMostPath[-2]

    def getAllVLabels(self):
        """
            Retrieves all vertex labels from the extended edge list.

            Returns:
                list: A list of vertex labels.
            """
        labels = []
        vertexMap = {}
        for ee in self.eeList:
            v1, v1Label = ee.getV1(), ee.getVLabel1()
            v2, v2Label = ee.getV2(), ee.getVLabel2()
            vertexMap[v1] = v1Label
            vertexMap[v2] = v2Label
        
        count = 0
        while count in vertexMap:
            labels.append(vertexMap[count])
            count += 1
        return labels

    def add(self, ee):
        """
            Adds an extended edge to the DFSCode object.

            Args:
                ee (ExtendedEdge): The extended edge to add.
            """
        if self.size == 0:
            self.rightMost = 1
            self.rightMostPath.extend([0, 1])
        else:
            v1, v2 = ee.getV1(), ee.getV2()
            if v1 < v2:
                self.rightMost = v2
                while self.rightMostPath and self.rightMostPath[-1] > v1:
                    self.rightMostPath.pop()
                self.rightMostPath.append(v2)

        self.eeList.append(ee)
        self.size += 1

    def getAt(self, index):
        """
            Retrieves the extended edge at the specified index.

            Args:
                index (int): The index of the extended edge to retrieve.

            Returns:
                ExtendedEdge: The extended edge at the specified index.
            """

        return self.eeList[index]

    def onRightMostPath(self, v):
        """
            Checks if a vertex is present on the rightmost path.

            Args:
                v (int): The vertex to check.

            Returns:
                bool: True if the vertex is present on the rightmost path, False otherwise.
            """
        return v in self.rightMostPath

    def containEdge(self, v1, v2):
        """
            Checks if an edge between vertices v1 and v2 is present.

            Args:
                v1 (int): The first vertex of the edge.
                v2 (int): The second vertex of the edge.

            Returns:
                bool: True if the edge is present, False otherwise.
            """
        for ee in self.eeList:
            if (ee.getV1() == v1 and ee.getV2() == v2) or (ee.getV1() == v2 and ee.getV2() == v1):
                return True
        return False

    def isEmpty(self):
        """
            Checks if the DFSCode object is empty.

            Returns:
                bool: True if the DFSCode object is empty, False otherwise."""
        return not self.eeList

    def getRightMost(self):
        """
            Retrieves the rightmost vertex.

            Returns:
                int: The index of the rightmost vertex.
            """
        return self.rightMost

    def getRightMostPath(self):
        """
            Retrieves the rightmost path.

            Returns:
                list: A list containing the vertices on the rightmost path.
            """
        return self.rightMostPath

    def getEeList(self):
        """
            Retrieves the extended edge list.

            Returns:
                list: A list of ExtendedEdge objects.
            """
        return self.eeList

    def __str__(self):
        """
            Returns a string representation of the DFSCode object.

            Returns:
                str: A string representation of the DFSCode object.
            """
        return "DfsCode: " + " ".join(str(ee) for ee in self.eeList)
