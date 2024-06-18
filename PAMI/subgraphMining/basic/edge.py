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
class Edge:
    def __init__(self, v1, v2, edgeLabel):
        self.v1 = v1
        self.v2 = v2
        self.edgeLabel = edgeLabel
        self.hashcode = (v1 + 1) * 100 + (v2 + 1) * 10 + edgeLabel

    def another(self, v):
        return self.v2 if v == self.v1 else self.v1

    def getEdgeLabel(self):
        return self.edgeLabel

    def __hash__(self):
        return self.hashcode

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.hashcode == other.hashcode and
                self.v1 == other.v1 and
                self.v2 == other.v2 and
                self.edgeLabel == other.edgeLabel)

    def __repr__(self):
        return f"Edge(v1: {self.v1}, v2: {self.v2}, Label: {self.edgeLabel})"
