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
import sys
import re
from math import sqrt
from geopy.distance import geodesic


class createNeighborhoodFileUsingGeodesicDistance:
    """
    This class create a neighbourhood file using Geodesic distance.

    Attribute:
    ----------
        :param iFile : file
            Input file name or path of the input file
        :param oFile : file
            Output file name or path pf the output file
        :param maxDistance : float
            The user can specify maxDistance in Km(Kilometers).
            This program find pairs of values whose Geodesic distance is less than or equal to maxDistace
            and store the pairs.
        :param  seperator: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    Methods:
    -------
        startMine()
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

    def __init__(self, iFile: str, oFile: str, maxDistance: float, seperator='\t'):
        self.iFile = iFile
        self.oFile = oFile
        self.maxDistance = maxDistance

        coordinates = []
        result = {}
        with open(self.iFile, "r") as f:
            for line in f:
                l = line.rstrip().split(seperator)
                # print(l)
                l[2] = re.sub(r'[^0-9. ]', '', l[2])
                coordinates.append(l[2].rstrip().split(' '))
                # print(l[0])
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i != j:
                    firstCoordinate = coordinates[i]
                    secondCoordinate = coordinates[j]
                    long1 = float(firstCoordinate[0])
                    lat1 = float(firstCoordinate[1])
                    long2 = float(secondCoordinate[0])
                    lat2 = float(secondCoordinate[1])

                    dist = geodesic((lat1, long1), (lat2, long2)).kilometers

                    if dist <= float(self.maxDistance):
                        result[tuple(firstCoordinate)] = result.get(tuple(firstCoordinate), [])
                        result[tuple(firstCoordinate)].append(secondCoordinate)

        with open(self.oFile, "w+") as f:
            for i in result:
                string = "Point(" + i[0] + " " + i[1] + ")" + seperator
                f.write(string)
                for j in result[i]:
                    string = "Point(" + j[0] + " " + j[1] + ")" + seperator
                    f.write(string)
                f.write("\n")

    def getFileName(self):
        return self.oFile


if __name__ == "__main__":
    obj = createNeighborhoodFileUsingGeodesicDistance(sys.argv[1], sys.argv[2], sys.argv[4])
