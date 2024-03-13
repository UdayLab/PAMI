import pickle

class DfsCode:
    def __init__(self):
        self.rightMost = -1
        self.size = 0
        self.rightMostPath = []  
        self.eeList = []

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def notPreOfRm(self, v):
        if len(self.rightMostPath) <= 1:
            return True
        return v != self.rightMostPath[-2]

    def getAllVLabels(self):
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
        return self.eeList[index]

    def onRightMostPath(self, v):
        return v in self.rightMostPath

    def containEdge(self, v1, v2):
        for ee in self.eeList:
            if (ee.getV1() == v1 and ee.getV2() == v2) or (ee.getV1() == v2 and ee.getV2() == v1):
                return True
        return False

    def isEmpty(self):
        return not self.eeList

    def getRightMost(self):
        return self.rightMost

    def getRightMostPath(self):
        return self.rightMostPath

    def getEeList(self):
        return self.eeList

    def __str__(self):
        return "DfsCode: " + " ".join(str(ee) for ee in self.eeList)
