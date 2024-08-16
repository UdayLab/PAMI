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
class FrequentSubgraph:
    def __init__(self, dfsCode, setOfGraphsIds, support):
        self.dfsCode = dfsCode
        self.setOfGraphsIds = setOfGraphsIds
        self.support = support

    def __eq__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support == other.support

    def __lt__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support < other.support

    def __gt__(self, other):
        if not isinstance(other, FrequentSubgraph):
            return NotImplemented
        return self.support > other.support
