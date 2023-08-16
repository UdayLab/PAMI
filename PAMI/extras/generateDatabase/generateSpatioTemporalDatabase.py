import random as rand
from typing import List, Dict, Tuple, Set, Union, Any, Generator
import pandas

class spatioTemporalDatabaseGenerator:
    coinFlip = [True, False]
    timestamp = list()
    items = list()
    alreadyAdded = set()
    outFileName=""

    def createPoint(self,xmin: int,xmax: int,ymin: int,ymax: int) -> Tuple[int, int]:
        x = rand.randint(xmin, xmax)
        y = rand.randint(ymin, ymax)
        coordinate = tuple([x, y])
        return coordinate

    def __init__(self,xmin: int,xmax: int,ymin: int,ymax: int,maxTimeStamp: int,numberOfItems: int, itemChanceLow: float,
                 itemChanceHigh: float, timeStampChanceLow: float,
                 timeStampChanceHigh: float) -> None:
        coinFlip = [True, False]
        timeStamp = 1
        self.timeStampList = list()
        self.itemList = list()

        while timeStamp != maxTimeStamp + 1:
            itemSet=list()
            for i in range(1, numberOfItems+1):
                #rand1=rand.rand(itemChanceLow,itemChanceHigh)
                #rand2 = rand.rand(timeStampChanceLow, timeStampChanceHigh)
                if rand.choices(coinFlip, weights=[itemChanceLow,itemChanceHigh], k=1)[0]:
                    coordinate=self.createPoint(xmin, xmax, ymin, ymax)
                    coordinate=tuple(coordinate)
                    if coordinate not in self.alreadyAdded:
                        coordinate=list(coordinate)
                        itemSet.append(coordinate)
                        coordinate=tuple(coordinate)
                        self.alreadyAdded.add(coordinate)
            if itemSet != []:
                self.timeStampList.append(
                    timeStamp)
                self.itemList.append(
                    itemSet)
            if rand.choices(coinFlip, weights=[itemChanceLow,itemChanceHigh], k=1)[0]:
                 timeStamp += 1
        self.outFileName = "temporal_" + str(maxTimeStamp // 1000) + \
                           "KI" + str(numberOfItems) + "C" + str(itemChanceLow) + "T" + str(timeStampChanceLow) + ".csv"




    def saveAsFile(self, outFileName="", sep="\t") -> None:
        if outFileName != "":
            self.outFileName = outFileName

        file = open(
            self.outFileName, "w")

        for i in range(len(self.timeStampList)):
            file.write(
                str(self.timeStampList[i]))
            for j in range(len(self.itemList[i])):
                file.write(
                    sep + str(self.itemList[i][j]))
            file.write('\n')

        file.close()


if __name__ == "__main__":
    xmin=0
    xmax=100
    ymin=0
    ymax=100
    maxTimeStamp = 10
    numberOfItems = 10
    itemChanceLow = 0.5
    itemChanceHigh = 0.9
    timeStampChanceLow = 0.5
    timeStampChanceHigh = 0.9
    generator = spatoTemporalDatabaseGenerator(xmin,xmax,ymin,ymax,maxTimeStamp, numberOfItems,
                                          itemChanceLow, itemChanceHigh, timeStampChanceLow, timeStampChanceHigh)
    generator.saveAsFile(outFileName='temp.txt')
