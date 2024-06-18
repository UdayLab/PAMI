# gSpan is a subgraph mining algorithm that uses DFS and DFS codes to mine subgraphs
#
# **Importing this algorithm into a python program**
#
#             from PAMI.subgraphMining.basic import gspan as alg
#
#             obj = alg.GSpan(iFile, minSupport)
#
#             obj.mine()
#
#             obj.run()
#
#             frequentGraphs = obj.getFrequentSubgraphs()
#
#             memUSS = obj.getMemoryUSS()
#
#             obj.save(oFile)
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
#


__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pickle

class DFSCode:
    def __init__(self):
        """
        Initializes the DFSCode object with default values.
        """
        self.rightMost = -1
        self.size = 0
        self.rightMostPath = []  
        self.eeList = []

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def notPreOfRm(self, v):
        """
        This function checks if a given value is not the second-to-last element on the
        `rightMostPath` given a vertex.
        """
        if len(self.rightMostPath) <= 1:
            return True
        return v != self.rightMostPath[-2]

    def getAllVLabels(self):
        """
        This function retrieves all vertex labels from the extended edge list and returns them in a list.
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
        The `add` function in adds elements to the EE list while updating the rightmost element and path
        based on certain conditions.
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
        return "DFSCode: " + " ".join(str(ee) for ee in self.eeList)
