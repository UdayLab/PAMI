# The `SparseTriangularMatrix` class represents a matrix with sparse triangular structure and provides
# methods for incrementing counts, getting support for items, setting support values, and removing
# infrequent entries based on a minimum support threshold.
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
class SparseTriangularMatrix:
    def __init__(self):
        self.matrix = {}  

    def __str__(self):
        temp = []
        for key in sorted(self.matrix.keys()):
            subkeys = self.matrix[key]
            subkeyStr = " ".join(f"{subkey}:{count}" for subkey, count in subkeys.items())
            temp.append(f"{key}: {subkeyStr}\n")
        return "".join(temp)


    def incrementCount(self, i, j):
        if i < j:
            key, subkey = i, j
        else:
            key, subkey = j, i

        if key not in self.matrix:
            self.matrix[key] = {subkey: 1}
        else:
            if subkey not in self.matrix[key]:
                self.matrix[key][subkey] = 1
            else:
                self.matrix[key][subkey] += 1

    def getSupportForItems(self, i, j):
        smaller, larger = min(i, j), max(i, j)
        return self.matrix.get(smaller, {}).get(larger, 0)

    def setSupport(self, i, j, support):
        smaller, larger = min(i, j), max(i, j)
        
        if smaller not in self.matrix:
            self.matrix[smaller] = {larger: support}
        else:
            self.matrix[smaller][larger] = support

    def removeInfrequentEntriesFromMatrix(self, minsup):
        for key in list(self.matrix.keys()):
            for subkey, value in list(self.matrix[key].items()):
                if value < minsup:
                    del self.matrix[key][subkey]
            if not self.matrix[key]:
                del self.matrix[key]
