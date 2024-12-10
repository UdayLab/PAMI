import random as _rd
import sys as _sys
import time
import os
import psutil
import numpy as np
import tqdm
import pandas as pd

class GeoReferentialTemporalDatabase:
    """
    This class create synthetic geo-referential temporal database.

    :Attribute:

        totalTransactions : int
            No of transactions
        noOfItems : int or float
            No of items
        avgTransactionLength : str
            The length of average transaction
        outputFile: str
            Name of the output file.

    :Methods:

        GeoReferentialTemporalDatabase(outputFile)
            Create geo-referential temporal database and store into outputFile

    **Credits:**
    ---------------
             The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(
            self,
            databaseSize: int,
            avgItemsPerTransaction: int,
            numItems: int,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            sep: str = '\t',
            occurrenceProbabilityOfSameTimestamp: float = 0,
            occurrenceProbabilityToSkipSubsequentTimestamp: float = 0,
    ) -> None:
        self.databaseSize = databaseSize
        self.avgItemsPerTransaction = avgItemsPerTransaction
        self.numItems = numItems
        self.db = []
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.seperator = sep
        self.occurrenceProbabilityOfSameTimestamp = occurrenceProbabilityOfSameTimestamp
        self.occurrenceProbabilityToSkipSubsequentTimestamp = occurrenceProbabilityToSkipSubsequentTimestamp
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()
        if numItems > ((x2 - x1) * (y2 - y1)):
            raise ValueError("Number of points is less than the number of lines * average items per line")

        self.itemPoint = {}
        usedPoints = set()

        for i in range(1, numItems + 1):
            # self.itemPoint[i] = (np.random.randint(x1, x2), np.random.randint(y1, y2))
            point = self.getPoint(x1, y1, x2, y2)
            while point in usedPoints:
                point = self.getPoint(x1, y1, x2, y2)
            self.itemPoint[i] = point

    def getPoint(self, x1, y1, x2, y2):

        return (np.random.randint(x1, x2), np.random.randint(y1, y2))

    def performCoinFlip(self, probability: float) -> bool:
        """
        Perform a coin flip with the given probability.

        :param probability: Probability of the coin landing heads (i.e., the event occurring).
        :return: True if the coin lands heads, False otherwise.
        """
        result = np.random.choice([0, 1], p=[1 - probability, probability])
        return result == 1

    def tuning(self, array, sumRes) -> list:
        """
        Tune the array so that the sum of the values is equal to sumRes

        :param array: list of values

        :type array: list

        :param sumRes: the sum of the values in the array to be tuned

        :type sumRes: int

        :return: list of values with the tuned values and the sum of the values in the array to be tuned and sumRes is equal to sumRes

        :rtype: list
        """

        while np.sum(array) != sumRes:
            # get index of largest value
            randIndex = np.random.randint(0, len(array))
            # if sum is too large, decrease the largest value
            if np.sum(array) > sumRes:
                array[randIndex] -= 1
            # if sum is too small, increase the smallest value
            else:
                minIndex = np.argmin(array)
                array[randIndex] += 1
        return array

    def generateArray(self, nums, avg, maxItems) -> list:
        """
        Generate a random array of length n whose values average to m

        :param nums: number of values

        :type nums: list

        :param avg: average value

        :type avg: float

        :param maxItems: maximum value

        :type maxItems: int

        :return: random array

        :rtype: list
        """

        # generate n random values
        values = np.random.randint(1, maxItems, nums)

        sumRes = nums * avg

        self.tuning(values, sumRes)

        # if any value is less than 1, increase it and tune the array again
        while np.any(values < 1):
            for i in range(nums):
                if values[i] < 1:
                    values[i] += 1
            self.tuning(values, sumRes)

        while np.any(values > maxItems):
            for i in range(nums):
                if values[i] > maxItems:
                    values[i] -= 1
            self.tuning(values, sumRes)

        # if all values are same then randomly increase one value and decrease another
        while np.all(values == values[0]):
            values[np.random.randint(0, nums)] += 1
            self.tuning(values, sumRes)

        return values

    def create(self) -> None:
        """
        Generate the Temporal database
        :return: None
        """
        self._startTime = time.time()
        db = set()
        lineSize = [] #may be error. need to check it.
        sumRes = self.databaseSize * self.avgItemsPerTransaction  # Total number of items

        for i in range(self.databaseSize):
            # Determine the timestamp
            if self.performCoinFlip(self.occurrenceProbabilityOfSameTimestamp):
                timestamp = self.current_timestamp
            else:
                if self.performCoinFlip(self.occurrenceProbabilityToSkipSubsequentTimestamp):
                    self.current_timestamp += 2
                else:
                    self.current_timestamp += 1
                timestamp = self.current_timestamp

            self.db.append([timestamp])  # Start the transaction with the timestamp

            lineSize.append([i, 0])  # Initialize lineSize with 0 for each transaction

        # Adjust lineSize to ensure sum of sizes equals sumRes
        lineSize = self.tuning(lineSize, sumRes)

        # For each transaction, generate items
        for i in tqdm.tqdm(range(len(lineSize))):
            transaction_index = lineSize[i][0]
            num_items = lineSize[i][1]

            if num_items > self.numItems:
                raise ValueError(
                    "Error: Either increase numItems or decrease avgItemsPerTransaction or modify percentage")
            items = np.random.choice(range(1, self.numItems + 1), num_items, replace=False)
            self.db[transaction_index].extend(items)

        self._runTime = time.time() - self._startTime
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def save(self,filename, sep='\t') -> None:
        """
        Save the Temporal database to a file

        :param filename: name of the file

        :type filename: str

        :param sep: seperator for the items

        :type sep: str

        :return: None
        """

        with open(filename, 'w') as f:
            for line in self.db:
                # f.write(','.join(map(str, line)) + '\n')
                line = list(map(str, line))
                f.write(sep.join(line) + '\n')
    def getTransactions(self) -> pd.DataFrame:
        """
        Get the Temporal database

        :return: the Temporal database

        :rtype: pd.DataFrame
        """
        df = pd.DataFrame(['\t'.join(map(str, line)) for line in self.db], columns=['Transactions'])
        return df


    def getRuntime(self) -> float:
        """
        Get the runtime of the transactional database

        :return: the runtime of the transactional database


        :rtype: float
        """
        return self._endTime - self._startTime

    def getMemoryUSS(self) -> float:

        return self._memoryUSS

    def getMemoryRSS(self) -> float:

        return self._memoryRSS
# if __name__ == "__main__":
#     _ap = str()
#     _ap = createSyntheticGeoreferentialTemporal(100000, 870, 10)
#     _ap.GeoreferentialTemporalDatabase("T10_geo_temp.txt")
# else:
#     print("Error! The number of input parameters do not match the total number of parameters provided")
