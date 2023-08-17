# generateSpatioTemporalDatabase this code is used to generates a SpatioTemporal Database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.generateDatabase import generateSpatioTemporalDatabase as db
#
#     obj = db.generateSpatioTemporalDatabase(xmin,xmax,ymin,ymax,maxTimeStamp,numberOfItems, itemChanceLow,
#                  itemChanceHigh, timeStampChanceLow,
#                  timeStampChanceHigh)
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

import random as rand
from typing import List, Dict, Tuple, Set, Union, Any, Generator
import pandas


class spatioTemporalDatabaseGenerator:
    """
        Description:
        -------------
            generateSpatioTemporalDatabase this code is used to generates a SpatioTemporal Database.

            :param xmin: int
                     To assign a minimum value for x according to the data size.
            :param xmax: int
                     To assign a maximum value for x according to the data size.
            :param ymin: int
                     To assign a minimum value for y according to the data size.
            :param ymax: int
                     To assign a maximum value for y according to the data size.
            :param maxTimeStamp: int
                     maxTimeStamp can be given here
            :param numberOfItems: int
                    it shows number of items in database
            :param itemChanceLow: float or int
                    it shows the least chance for item
            :param itemChanceHigh: float or int
                    it shows the highest chance for item
            :param timeStampChanceLow: float or int
                    it shows the least chance of timeStamp
            :param timeStampChanceHigh: float or int
                    it shows the highest chance of timeStamp

        Methods:
        ---------
            createPoint():
                creates a point in SpatioTemporalDatabase
            saveAsFile:
                saves the data as a file

        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

         from PAMI.extras.generateDatabase import generateSpatioTemporalDatabase as db

        obj = db.generateSpatioTemporalDatabase(xmin,xmax,ymin,ymax,maxTimeStamp,numberOfItems, itemChanceLow,
                 itemChanceHigh, timeStampChanceLow,
                 timeStampChanceHigh)

        obj.save(oFile)

        """
    coinFlip = [True, False]
    timestamp = list()
    items = list()
    alreadyAdded = set()
    outFileName = ""

    def createPoint(self, xmin: int, xmax: int, ymin: int, ymax: int) -> Tuple[int, int]:
        x = rand.randint(xmin, xmax)
        y = rand.randint(ymin, ymax)
        coordinate = tuple([x, y])
        return coordinate

    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int, maxTimeStamp: int, numberOfItems: int,
                 itemChanceLow: float,
                 itemChanceHigh: float, timeStampChanceLow: float,
                 timeStampChanceHigh: float) -> None:
        coinFlip = [True, False]
        timeStamp = 1
        self.timeStampList = list()
        self.itemList = list()

        while timeStamp != maxTimeStamp + 1:
            itemSet = list()
            for i in range(1, numberOfItems + 1):
                # rand1=rand.rand(itemChanceLow,itemChanceHigh)
                # rand2 = rand.rand(timeStampChanceLow, timeStampChanceHigh)
                if rand.choices(coinFlip, weights=[itemChanceLow, itemChanceHigh], k=1)[0]:
                    coordinate = self.createPoint(xmin, xmax, ymin, ymax)
                    coordinate = tuple(coordinate)
                    if coordinate not in self.alreadyAdded:
                        coordinate = list(coordinate)
                        itemSet.append(coordinate)
                        coordinate = tuple(coordinate)
                        self.alreadyAdded.add(coordinate)
            if itemSet != []:
                self.timeStampList.append(
                    timeStamp)
                self.itemList.append(
                    itemSet)
            if rand.choices(coinFlip, weights=[itemChanceLow, itemChanceHigh], k=1)[0]:
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
    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    maxTimeStamp = 10
    numberOfItems = 10
    itemChanceLow = 0.5
    itemChanceHigh = 0.9
    timeStampChanceLow = 0.5
    timeStampChanceHigh = 0.9
    generator = spatioTemporalDatabaseGenerator(xmin, xmax, ymin, ymax, maxTimeStamp, numberOfItems,
                                               itemChanceLow, itemChanceHigh, timeStampChanceLow, timeStampChanceHigh)
    generator.saveAsFile(outFileName='temp.txt')