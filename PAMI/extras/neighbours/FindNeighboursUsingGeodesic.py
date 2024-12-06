# findNeighboursUsingGeodesic is a code used to create a neighbourhood file using Geodesic distance.
#
#  **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.neighbours import findNeighboursUsingGeodesic as db
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
from geopy.distance import geodesic
import time
import sys
import psutil,os,tqdm
import pandas as pd


class FindNeighboursUsingGeodesic:
    """
    This class create a neighbourhood file using Geodesic distance.

    :Attribute:

        :param iFile : file
            Input file name or path of the input file
        :param maxDist : float
            The user can specify maxDist in Km(Kilometers).
            This program find pairs of values whose Geodesic distance is less than or equal to maxDistace
            and store the pairs.
        :param  sep: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    :Methods:

        mine()
            find and store the pairs of values whose Geodesic distance is less than or equal to maxDistace.
        getFileName()
            This function returns output file name.

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.neighbours import findNeighboursUsingGeodesic as db

             obj = db.findNeighboursUsingGeodesic(iFile, oFile, 10, "\t")

            obj.save()
    """

    def __init__(self, iFile: str, maxDist: float, sep='\t',DBtype="temp"):
        self.iFile = iFile
        self.maxGeodesicDistance = maxDist
        self.seperator = sep
        self.result = {}
        self.DBtype = DBtype
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()



    def create(self) -> None:
        self._startTime = time.time()
        coordinates = []
        with open(self.iFile, "r") as f:
            if self.DBtype == "temp":
                for line in f:
                    l = line.rstrip().split(self.seperator)
                    for i in l[1:]:
                        i = re.sub(r'[^0-9. ]', '', i)
                        if i not in coordinates:
                            coordinates.append(i.rstrip().split(' '))
            else:
                for line in f:
                    l = line.rstrip().split(self.seperator)
                    for i in l:
                        i = re.sub(r'[^0-9. ]', '', i)
                        if i not in coordinates:
                            coordinates.append(i.rstrip().split(' '))
        for i in tqdm.tqdm(range(len(coordinates))):
            for j in range(len(coordinates - i - 1)):
                    j = j + i + 1
                    firstCoordinate = coordinates[i]
                    secondCoordinate = coordinates[j]
                    long1 = float(firstCoordinate[0])
                    lat1 = float(firstCoordinate[1])
                    long2 = float(secondCoordinate[0])
                    lat2 = float(secondCoordinate[1])

                    dist = geodesic((lat1, long1), (lat2, long2)).kilometers

                    if dist <= float(self.maxGeodesicDistance):
                        self.result[tuple(firstCoordinate)] = self.result.get(tuple(firstCoordinate), [])
                        self.result[tuple(firstCoordinate)].append(secondCoordinate)
                        self.result[tuple(secondCoordinate)] = self.result.get(tuple(secondCoordinate), [])
                        self.result[tuple(secondCoordinate)].append(firstCoordinate)
        self._endTime = time.time()

    def save(self, oFile: str) -> None:
        with open(oFile, "w+") as f:
            for i in self.result:
                string = "Point(" + i[0] + " " + i[1] + ")" + self.seperator
                f.write(string)
                for j in self.result[i]:
                    string = "Point(" + j[0] + " " + j[1] + ")" + self.seperator
                    f.write(string)
                f.write("\n")

    def getNeighboringInformation(self):
        df = pd.DataFrame(['\t'.join(map(str, line)) for line in self.result], columns=['Neighbors'])
        return df
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
    obj = FindNeighboursUsingGeodesic(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
