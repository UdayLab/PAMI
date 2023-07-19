import sys
import re
from math import sqrt
from geopy.distance import geodesic


class createNeighborhoodFileUsingGeodesicDistance:
    """
    This class create a neighbourhood file using Geodesic distance.

    Attribute:
    ----------
        iFile : file
            Input file name or path of the input file
        oFile : file
            Output file name or path pf the output file
        maxDistace : float
            The user can specify maxDistace in Km(Kilometers). 
            This program find pairs of values whose Geodesic distance is less than or equal to maxDistace
            and store the pairs.

    Methods:
    -------
        startMine()
            find and store the pairs of values whose Geodesic distance is less than or equal to maxDistace.
        getFileName()
            This function returns output file name.
    """

    def __init__(self,iFile,oFile,maxDistace, seperator='\t'):
        self.iFile = iFile
        self.oFile = oFile
        self.maxDistace = maxDistace

        coordinates = []
        result = {}
        with open(self.iFile,"r") as f:
            for line in f:
                l = line.rstrip().split(seperator)
                #print(l)
                l[2] = re.sub(r'[^0-9. ]', '', l[2])
                coordinates.append(l[2].rstrip().split(' '))
                #print(l[0])
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i != j:
                    firstCoordinate = coordinates[i]
                    secondCoordinate = coordinates[j]
                    long1 = float(firstCoordinate[0])
                    lat1 = float(firstCoordinate[1])
                    long2 = float(secondCoordinate[0])
                    lat2 = float(secondCoordinate[1])
                    
                    dist = geodesic((lat1,long1),(lat2,long2)).kilometers
                    
                    if dist <= float(self.maxDistace):
                        result[tuple(firstCoordinate)] = result.get(tuple(firstCoordinate),[])
                        result[tuple(firstCoordinate)].append(secondCoordinate)

        with open(self.oFile,"w+") as f:
            for i in result:
                string = "Point(" +i[0]+" "+i[1] + ")"+ seperator
                f.write(string)
                for j in result[i]:
                    string = "Point(" + j[0] + " " + j[1] + ")"+ seperator
                    f.write(string)
                f.write("\n")


    def getFileName(self):
        return self.oFile

if __name__ == "__main__":
    createNeighborhoodFileUsingGeodesicDistance('stationInfo.csv', 'road_points.txt',10, ',')