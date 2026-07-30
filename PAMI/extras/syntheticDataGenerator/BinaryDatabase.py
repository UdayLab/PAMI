# BinaryDatabase generates a synthetic binary (0/1) transactional database. Each row is a transaction and each column is an item; a 1 means the item is present in that transaction. This is the native input format of the binary frequent-pattern miners (BinaryApriori, BinaryECLAT, BinaryFPGrowth).
#
#  **Importing this algorithm into a python program**
#  --------------------------------------------------------
#     from PAMI.extras.syntheticDataGenerator import BinaryDatabase as db
#
#     obj = db.BinaryDatabase(10, 5, 10)
#
#     obj.create()
#
#     obj.save('binaryDB.txt')
#
#     print(obj.getTransactions())
#
import numpy as np
import pandas as pd
import sys, psutil, os, time

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


class BinaryDatabase:
    """
        :Description: BinaryDatabase generates a synthetic binary (0/1) transactional database where rows are transactions
                      and columns are items. A 1 in column *j* of row *i* means item *j* is present in transaction *i*.
                      The output is directly consumable by the binary frequent-pattern miners (BinaryApriori, BinaryECLAT,
                      BinaryFPGrowth) and by the binaryDF2DB converter.

        :Attributes:

            databaseSize: int
                Number of transactions (rows) in the database
            avgItemsPerTransaction: int
                Average number of present items (1s) per transaction
            numItems: int
                Total number of items (columns)
            memoryUSS : float
                To store the total amount of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program
            startTime : float
                To record the start time of the generation process
            endTime : float
                To record the completion time of the generation process

        :Methods:

            create:
                Generate the binary transactional database
            save:
                Save the binary database to a user-specified file (one row of 0/1 values per line)
            getTransactions:
                Get the binary database as a pandas DataFrame (columns item1..itemN)
            getMemoryUSS()
                Total amount of USS memory consumed by the generation process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the generation process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the generation process will be retrieved from this function

        **Methods to execute code on terminal**
        ---------------------------------------------

        .. code-block:: console

          Format:

          (.venv) $ python3 BinaryDatabase.py <databaseSize> <avgItemsPerTransaction> <numItems> <outputFile>

          Example Usage:

          (.venv) $ python3 BinaryDatabase.py 50 10 100 binaryDB.txt


        **Importing this algorithm into a python program**
        --------------------------------------------------------
            from PAMI.extras.syntheticDataGenerator import BinaryDatabase as db

            obj = db.BinaryDatabase(10, 5, 10)

            obj.create()

            obj.save('binaryDB.txt')

            print(obj.getTransactions())


        """

    def __init__(self, databaseSize, avgItemsPerTransaction, numItems, sep="\t") -> None:
        """
        Initialize the binary database with the given parameters

        :param databaseSize: total number of transactions (rows) in the database
        :type databaseSize: int
        :param avgItemsPerTransaction: average number of present items (1s) per transaction
        :type avgItemsPerTransaction: int
        :param numItems: total number of items (columns)
        :type numItems: int
        :param sep: separator used between the 0/1 values of a row
        :type sep: str
        """

        self.databaseSize = databaseSize
        self.avgItemsPerTransaction = avgItemsPerTransaction
        self.numItems = numItems
        self.sep = sep
        self._matrix = None
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()

    @staticmethod
    def _generateArray(databaseSize, avgItemsPerTransaction):
        """
        Generate per-row 1-counts whose average equals avgItemsPerTransaction (same weighting scheme used by
        the other synthetic generators in this package).
        """
        transactionSize = np.random.rand(databaseSize)
        sumTransactions = np.sum(transactionSize)
        weights = transactionSize / sumTransactions
        sumResultant = avgItemsPerTransaction * databaseSize
        valuesInt = np.round(weights * sumResultant).astype(int)

        indexZero = np.where(valuesInt == 0)[0]
        for i in indexZero:
            valuesInt[i] += 1
        return valuesInt

    def create(self) -> None:
        """
        Generate the binary transactional database as a (databaseSize x numItems) 0/1 matrix.
        """
        self._startTime = time.time()
        counts = self._generateArray(self.databaseSize, self.avgItemsPerTransaction)
        # A transaction cannot hold more distinct items than exist.
        counts = np.clip(counts, 1, self.numItems)

        matrix = np.zeros((self.databaseSize, self.numItems), dtype=np.uint8)
        for i in range(self.databaseSize):
            chosen = np.random.choice(self.numItems, counts[i], replace=False)
            matrix[i, chosen] = 1
        self._matrix = matrix
        self._endTime = time.time()

    def save(self, filename) -> None:
        """
        Save the binary database to a file, one transaction per line as sep-separated 0/1 values.

        :param filename: name of the output file
        :type filename: str
        """
        with open(filename, 'w') as f:
            for row in self._matrix:
                f.write(str(self.sep).join(map(str, row.tolist())) + '\n')

    def getTransactions(self) -> pd.DataFrame:
        """
        Get the binary database as a pandas DataFrame whose columns are item1..itemN.

        :return: the binary database
        :rtype: pd.DataFrame
        """
        columns = ["item" + str(j + 1) for j in range(self.numItems)]
        return pd.DataFrame(self._matrix, columns=columns)

    def getMemoryUSS(self) -> float:
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        process = psutil.Process(os.getpid())
        self._memoryRSS = process.memory_info().rss
        return self._memoryRSS

    def getRuntime(self) -> float:
        return self._endTime - self._startTime


if __name__ == "__main__":
    if len(sys.argv) == 5:
        obj = BinaryDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        obj.create()
        obj.save(sys.argv[4])
        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS", obj.getMemoryRSS())
        print("Total ExecutionTime in ms:", obj.getRuntime())
    elif len(sys.argv) == 6:
        obj = BinaryDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        obj.create()
        obj.save(sys.argv[5])
        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS", obj.getMemoryRSS())
        print("Total ExecutionTime in ms:", obj.getRuntime())
    elif len(sys.argv) == 4:
        obj = BinaryDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        obj.create()
        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS", obj.getMemoryRSS())
        print("Total ExecutionTime in ms:", obj.getRuntime())
    else:
        raise ValueError(
            "Invalid number of arguments. Args: <databaseSize> <avgItemsPerTransaction> <numItems> <filename> "
            "or Args: <databaseSize> <avgItemsPerTransaction> <numItems> <sep> <filename>")
