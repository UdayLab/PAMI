# TransactionalDatabase is a collection of transactions. It only considers the data in  transactions and ignores the metadata.
#
#  **Importing this algorithm into a python program**
#  --------------------------------------------------------
#     from PAMI.extras.syntheticDataGenerator import TransactionalDatabase as db
#
#     obj = db(10, 5, 10)
#
#     obj.create()
#
#     obj.save('db.txt')
#
#     print(obj.getTransactions())
#

import numpy as np
import pandas as pd
import sys


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

class TransactionalDatabase:
    """
    :Description: TransactionalDatabase is a collection of transactions. It only considers the data in  transactions and ignores the metadata.
    :Attributes:

        numLines: int
            Number of lines
        avgItemsPerLine: int
            Average number of items per line
        numItems: int
            Total number of items

    :Methods:

        create: 
            Generate the transactional database
        save: 
            Save the transactional database to a user-specified file
        getTransactions: 
            Get the transactional database


    **Methods to execute code on terminal**
    ---------------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 TransactionalDatabase.py <numLines> <avgItemsPerLine> <numItems>

      Example Usage:

      (.venv) $ python3 TransactionalDatabase.py 50.0 10.0 100



    **Importing this algorithm into a python program**
    --------------------------------------------------------
        from PAMI.extras.syntheticDataGenerator import TransactionalDatabase as db

        obj = db.TransactionalDatabase(10, 5, 10)

        obj.create()

        obj.save('db.txt')

        print(obj.getTransactions())

    
    """

    def __init__(self, databaseSize, avgItemsPerTransaction, numItems,seperator = "\t") -> None:
        """
        Initialize the transactional database with the given parameters

        :param databaseSize: total number of transactions in the database
        :type databaseSize: int
        :param avgItemsPerTransaction: average number of items per transaction
        :type avgItemsPerTransaction: int
        :param numItems: total number of items
        :type numItems: int
        :param seperator: separator to distinguish the items in a transaction
        :type seperator: str
        """

        self.databaseSize = databaseSize
        self.avgItemsPerTransaction = avgItemsPerTransaction
        self.numItems = numItems
        self.seperator = seperator
        self.db = []
    
    def _tuning(self, array, sumRes) -> list:
        """
        Tune the array so that the sum of the values is equal to sumRes

        :param array: list of values
        :type array: list
        :param sumRes: target sum
        :type sumRes: int

        Returns:
        array: list - tuned array
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
        

    def _generateArray(self, nums, avg, maxItems) -> list:
        """
        Generate a random array of length n whose values average to m

        :param nums: number of values
        :type nums: int
        :param avg: average value
        :type avg: int
        :param maxItems: maximum value
        :type maxItems: int

        Returns:
        values: list - random array
        """

        # generate n random values
        values = np.random.randint(1, maxItems, nums)

        sumRes = nums * avg

        self._tuning(values, sumRes)

        # if any value is less than 1, increase it and tune the array again
        while np.any(values < 1):
            for i in range(nums):
                if values[i] < 1:
                    values[i] += 1
            self._tuning(values, sumRes)

        while np.any(values > maxItems):
            for i in range(nums):
                if values[i] > maxItems:
                    values[i] -= 1
            self._tuning(values, sumRes)


        # if all values are same then randomly increase one value and decrease another
        while np.all(values == values[0]):
            values[np.random.randint(0, nums)] += 1
            self._tuning(values, sumRes)

        return values

    def create(self) -> None:
        """
        Generate the transactional database with the given input parameters.
        Returns: None
        """
        db = set()

        values = self._generateArray(self.databaseSize, self.avgItemsPerTransaction, self.numItems)

        for value in values:
            line = np.random.choice(range(1, self.numItems + 1), value, replace=False)
            self.db.append(line)

    def save(self, filename) -> None:
        """
        Save the transactional database to a file

        :param filename: name of the file
        :type filename: str
        """

        with open(filename, 'w') as f:
            for line in self.db:
                f.write(str(self.seperator).join(map(str, line)) + '\n')

    def getTransactions(self) -> pd.DataFrame:
        """
        Get the transactional database in dataFrame format

        Returns:
        db: pd.dataFrame - transactional database
        """
        column = "Transactions"
        db = pd.DataFrame(columns=[column])
        self.db = [tuple(x) for x in self.db]
        for i in range(len(self.db)):
            db.at[i,column] = self.db[i]
        return db
        

if __name__ == "__main__":
    if len(sys.argv) == 5:
        obj = TransactionalDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        obj.create()
        obj.save(sys.argv[4])
    if len(sys.argv) == 6:
        obj = TransactionalDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        obj.create()
        obj.save(sys.argv[5])
    else:
        raise ValueError("Invalid number of arguments. Args: <numLines> <avgItemsPerLine> <numItems> <filename> or Args: <numLines> <avgItemsPerLine> <numItems> <seperator> <filename>")
    