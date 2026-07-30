# generateBinaryDatabase generates a synthetic binary (0/1) transactional database.
#
#  **Importing this algorithm into a python program**
#  --------------------------------------------------------
#     from PAMI.extras.generateDatabase import generateBinaryDatabase as db
#     obj = db.generateBinaryDatabase(10, 5, 10)
#     obj.create()
#     obj.save('\t', 'binaryDB.txt')
#     print(obj.getTransactions()) to get the binary database as a pandas dataframe

# **Running the code from the command line**
# --------------------------------------------------------
#     python generateBinaryDatabase.py 10 5 10 binaryDB.txt
#     cat binaryDB.txt
#


__copyright__ = """
Copyright (C)  2026 Rage Uday Kiran

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


import numpy as np
import pandas as pd
import sys


class generateBinaryDatabase:
    """
    :Description: Generate a synthetic binary (0/1) transactional database with the given number of lines
                  (transactions), average number of present items per line, and total number of items. Rows are
                  transactions and columns are items; a 1 means the item is present in that transaction.

    :Attributes:
    numLines: int
        - number of lines (transactions)
    avgItemsPerLine: int
        - average number of present items (1s) per line
    numItems: int
        - total number of items (columns)

    :Methods:
        create:
            Generate the binary database
        save:
            Save the binary database to a file
        getTransactions:
            Get the binary database as a pandas DataFrame
    """

    def __init__(self, numLines, avgItemsPerLine, numItems) -> None:
        """
        Initialize the binary database with the given parameters

        :param numLines: number of lines (transactions)
        :param avgItemsPerLine: average number of present items per line
        :param numItems: total number of items
        """

        self.numLines = numLines
        self.avgItemsPerLine = avgItemsPerLine
        self.numItems = numItems
        self.db = None

    def create(self) -> None:
        """
        Generate the binary database as a (numLines x numItems) 0/1 matrix.

        :return: None
        """
        # Per-line 1-counts averaging avgItemsPerLine, bounded by [1, numItems].
        counts = np.random.poisson(self.avgItemsPerLine, self.numLines)
        counts = np.clip(counts, 1, self.numItems)

        matrix = np.zeros((self.numLines, self.numItems), dtype=np.uint8)
        for i in range(self.numLines):
            chosen = np.random.choice(self.numItems, counts[i], replace=False)
            matrix[i, chosen] = 1
        self.db = matrix

    def save(self, sep, filename) -> None:
        """
        Save the binary database to a file, one transaction per line as sep-separated 0/1 values.

        :param sep: separator
        :type sep: str
        :param filename: name of the file
        :type filename: str
        :return: None
        """
        with open(filename, 'w') as f:
            for row in self.db:
                f.write(sep.join(map(str, row.tolist())) + '\n')

    def getTransactions(self) -> pd.DataFrame:
        """
        Get the binary database as a pandas DataFrame whose columns are item1..itemN.

        :return: the binary database
        :rtype: pd.DataFrame
        """
        columns = ["item" + str(j + 1) for j in range(self.numItems)]
        return pd.DataFrame(self.db, columns=columns)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        obj = generateBinaryDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        obj.create()
        obj.save('\t', sys.argv[4])
    else:
        # default demonstration
        obj = generateBinaryDatabase(10, 5, 10)
        obj.create()
        obj.save('\t', 'binaryDB.txt')
        print(obj.getTransactions())
