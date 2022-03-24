import sys
import re
from math import sqrt

class createNeighborhoodFileUsingEuclideanDistance:
    """
    This class create a neighbourhood file using euclid distance.

    Attribute:
    ----------
        iFile : file
            Input file name or path of the input file
        oFile : file
            Output file name or path pf the output file
        maxEucledianDistace : int
            The user can specify maxEucledianDistace.
            This program find pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace
            and store the pairs.

    Methods:
    -------
        startMine()
            find and store the pairs of values whose Euclidean distance is less than or equal to maxEucledianDistace.
        getFileName()
            This function returns output file name.
    """

    def __init__(self,iFile,oFile,maxEucledianDistace, seperator='\t'):
        self.iFile = iFile
        self.oFile = oFile
        self.maxEucledianDistace = maxEucledianDistace

        coordinates = []
        result = {}
        with open(self.iFile,"r") as f:
            for line in f:
                l = line.rstrip().split(seperator)
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
                    ansX = x2-x1
                    ansY = y2-y1
                    dist = abs(pow(ansX,2) - pow(ansY,2))
                    norm = sqrt(dist)
                    if norm <= float(self.maxEucledianDistace):
                        result[tuple(firstCoordinate)] = result.get(tuple(firstCoordinate),[])
                        result[tuple(firstCoordinate)].append(secondCoordinate)

        with open(self.oFile,"w") as f:
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
    createNeighborhoodFileUsingEuclideanDistance(sys.argv[1],sys.argv[2],sys.argv[3])

