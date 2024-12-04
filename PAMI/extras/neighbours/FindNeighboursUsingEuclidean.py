# findNeighboursUsingEuclidean is a code used to create a neighbourhood file using Euclidean distance.
#
#  **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.neighbours import findNeighboursUsingEuclidean as db
#
#     obj = db.findNeighboursUsingGeodesic(iFile, oFile, 10, "\t")
#
#     obj.save()
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

import re
from math import sqrt
import time
import sys, psutil, os


class FindNeighboursUsingEuclidean:
    """
    This class create a neighbourhood file using euclid distance.

    :Attribute:

        :param iFile : file
            Input file name or path of the input file
        :param maxDist : int
            The user can specify maxEuclideanDistance.
            This program find pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace
            and store the pairs.
        :param  sep: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    :Methods:

        mine()
            find and store the pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace.
        getFileName()
            This function returns output file name.

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.neighbours import findNeighboursUsingEuclidean as db

            obj = db.findNeighboursUsingEuclidean(iFile, oFile, 10, "\t")

            obj.save()
    """

    def __init__(self, iFile: str, maxDist: int, sep='\t') -> None:
        self.iFile = iFile
        self.maxEucledianDistance = maxDist
        self.seperator = sep
        self.result = {}

    def create(self):
        self._startTime = time.time()
        coordinates = []
        with open(self.iFile, "r") as f:
            for line in f:
                l = line.rstrip().split(self.seperator)
                l[0] = re.sub(r'[^0-9. ]', '', l[0])
                coordinates.append(l[0].rstrip().split(' '))
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i != j:
                    firstCoordinate = coordinates[i]
                    secondCoordinate = coordinates[j]
                    x1 = float(firstCoordinate[0])
                    y1 = float(firstCoordinate[1])
                    x2 = float(secondCoordinate[0])
                    y2 = float(secondCoordinate[1])
                    ansX = x2 - x1
                    ansY = y2 - y1
                    dist = abs(pow(ansX, 2) - pow(ansY, 2))
                    norm = sqrt(dist)
                    if norm <= float(self.maxEucledianDistance):
                        self.result[tuple(firstCoordinate)] = self.result.get(tuple(firstCoordinate), [])
                        self.result[tuple(firstCoordinate)].append(secondCoordinate)
        self._endTime = time.time()

    def save(self,oFile: str) -> None:
        with open(oFile, "w+") as f:
            for i in self.result:
                string = "Point(" + i[0] + " " + i[1] + ")" + self.seperator
                f.write(string)
                for j in self.result[i]:
                    string = "Point(" + j[0] + " " + j[1] + ")" + self.seperator
                    f.write(string)
                f.write("\n")

    def getNeighbors():
        return self.result

    def getRuntime(self) -> float:
        """
        Get the runtime of the transactional database

        :return: the runtime of the transactional database


        :rtype: float
        """
        return self._endTime - self._startTime

    def getMemoryUSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        return self._memoryUSS

    def getMemoryRSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryRSS = process.memory_info().rss
        return self._memoryRSS


if __name__ == "__main__":
    obj = FindNeighboursUsingEuclidean(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
