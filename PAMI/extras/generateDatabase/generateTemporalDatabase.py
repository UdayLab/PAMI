# generateTemporalDatabase this code is used to generates a temporal Database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.generateDatabase import generateTemporalDatabase as db
#
#     obj = db.generateTemporalDatabase(numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction,  oFile, percentage, "\t" , typeOfFile)
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

import random
import pandas as pd
from typing import Tuple, List, Union
import os


class generateTemporalDatabase:
    """
    Description:
    -------------
        generateTemporalDatabase creates a temporal database and outputs a database or a frame depending on input

        :param numOfTransactions: int
            number of transactions
        :param maxNumOfItem: int
            highest value an item can be
        :param maxNumOfItemsPerTransaction: int
            maximum number of items a transaction can be
        :param outputFile: str
            output file name
        :param percentage: int
            percentage of coinToss for TID of temporalDatabase
        :param sep: str
            seperator for database output file
        :param typeOfFile: str
            specify database or dataframe to get corresponding output

    Methods:
    ---------
        getFileName():
            returns filename
        createTemporalFile():
            creates temporal database file or dataframe
        getDatabaseAsDataFrame:
            returns dataframe

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

     from PAMI.extras.generateDatabase import generateTemporalDatabase as db

    obj = db.generateTemporalDatabase(numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction,  oFile, percentage, "\t" , typeOfFile)

    obj.save(oFile)

    """
    def __init__(self, numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction, outputFile, percentage=50,
                 sep='\t', typeOfFile="Database"):
        """

        :param numOfTransactions: number of transactions
        :type numOfTransactions: int
        :param maxNumOfItems: Highest value an item can be
        :type maxNumOfItems: int
        :param maxNumOfItemsPerTransaction: max number of items per transaction
        :type maxNumOfItemsPerTransaction: int
        :param outputFile: output file/filename
        :type outputFile: str
        :param percentage: Chance of coinFlip for temporal TID
        :type percentage: int
        :param sep: seperator
        :type sep: str
        :param typeOfFile: specify whether database or dataframe to create respective objects. Note: dataframe must be
                            retrieved later with getDatabase as Dataframe
        :type typeOfFile: str

        *
        """
        self.numOfTransactions = numOfTransactions
        self.maxNumOfItems = maxNumOfItems
        self.maxNumOfItemsPerTransaction = maxNumOfItemsPerTransaction
        self.outputFile = outputFile
        self.percentage = percentage
        self.sep = sep
        self.typeOfFile = typeOfFile.lower()

        def getFileName(self) -> str:
            """
            return filename
            :return:
            """
            return self.outputFile

        def getDatabaseAsDataFrame(self) -> pd.DataFrame:
            """
            return dataframe
            return: pd.dataframe
            """
            return self.df

        def createTemporalFile(self) -> None:
            """
            create Temporal database or dataframe depending on input
            :return:
            """
            with open(self.outputFile, "w") as outFile:
                itemFrameSet = list()
                timeStampList = list()
                # This hashset will be used to remember which items have
                # already been added to this item set.
                timestamp = 1
                coinFlip = [True, False]
                alreadyAdded = set()
                # create an arraylist to store items from the item set that will be generated
                itemSet = list()
                # We randomly decide how many items will appear in this transaction
                randNumOfItems = random.randint(1, self.maxNumOfItemsPerTransaction)
                # for the number of items that was decided above
                for j in range(randNumOfItems):
                    # we generate the item randomly and write it to disk
                    item = random.randint(1, self.maxNumOfItems)
                    # if we already added this item to this item set
                    # we choose another one
                    while item in alreadyAdded:
                        item = random.randint(1, self.maxNumOfItems)
                    alreadyAdded.add(item)
                    itemSet.append(item)
                # sort the item set
                itemSet.sort()
                if self.typeOfFile == "database":
                    outFile.write(str(timestamp) + self.sep)
                    for j in itemSet:
                        outFile.write(str(j) + self.sep)
                    outFile.write('\n')
                if self.typeOfFile == "dataframe":
                    timeStampList.append(timestamp)
                    itemFrameSet.append(itemSet)
                # add item
                for i in range(self.numOfTransactions - 1):
                    while random.choices(coinFlip, weights=[self.percentage, 100 - self.percentage], k=1)[0]:
                        timestamp += 1
                        nextTimestamp = timestamp + 1
                    if not random.choices(coinFlip, weights=[self.percentage, 100 - self.percentage], k=1)[0]:
                        timestamp += 1
                        nextTimestamp = timestamp + 1
                    alreadyAdded = set()
                    # create an arraylist to store items from the item set that will be generated
                    itemSet = list()
                    randNumOfItems = random.randint(1, self.maxNumOfItemsPerTransaction)
                    for j in range(randNumOfItems):
                        # we generate the item randomly and write it to disk
                        item = random.randint(1, self.maxNumOfItems)
                        # if we already added this item to this item set
                        # we choose another one
                        while item in alreadyAdded:
                            item = random.randint(1, self.maxNumOfItems)
                        alreadyAdded.add(item)
                        itemSet.append(item)
                    # sort the item set
                    itemSet.sort()
                    # writing the item set
                    if self.typeOfFile == "database":
                        outFile.write(str(timestamp) + self.sep)
                        for j in itemSet:
                            outFile.write(str(j) + self.sep)
                        outFile.write('\n')
                    if self.typeOfFile == "dataframe":
                        timeStampList.append(timestamp)
                        itemFrameSet.append(itemSet)

                if self.typeOfFile == "dataframe":
                    data = {
                        'timestamp': timeStampList,
                        'transactions': pd.Series(itemFrameSet)
                    }
                    self.df = pd.DataFrame(data)
            outFile.close()
            if self.typeOfFile == "dataframe":
                os.remove(outFileName)

    if __name__ == '__main__':
        numOfTransactions = 100
        maxNumOfItems = 10
        maxNumOfItemsPerTransaction = 6
        outFileName = 'temporal_out.txt'
        sep = '\t'
        frameOrBase = "database"

        temporalDB = generateTemporalDatabase(numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction,
                                              outFileName)

        temporalDB.createTemporalFile()

        numOfTransactions = 100
        maxNumOfItems = 10
        maxNumOfItemsPerTransaction = 6
        outFileName = 'temporal_ot.txt'
        sep = '\t'
        percent = 50
        frameOrBase = "dataframe"

        temporalDB = generateTemporalDatabase(numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction,
                                              outFileName, percent, sep, frameOrBase)

        temporalDB.createTemporalFile()

        print(temporalDB.getDatabaseAsDataFrame())