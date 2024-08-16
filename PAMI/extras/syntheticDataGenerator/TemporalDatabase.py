# TemporalDatabase is a collection of timestamps and along with data at particular time.
#
#  **Importing this algorithm into a python program**
#
#             from PAMI.extras.syntheticDataGenerator import TemporalDatabase as db
#
#             temporalDB = db.TemporalDatabase(numOfTransactions, avgTransactionLength, numItems, outFileName, percentage, sep, occurrenceProbabilityAtSameTimestamp, occurrenceProbabilityToSkipSubsequentTimestamp)
#
#             temporalDB.create()
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

import pandas as pd
import numpy as np
import sys


class TemporalDatabase:
    """
    :Description: - creates a temporal database with required parameter (e.g.,databaseSize, avgItemsPerTransaction, numItems and outputFile).
                  - output can be printed in two ways either in text file or dataframe depending on the input type.

    :Attributes:

        :param databaseSize: int
            number of transactions

        :param avgItemsPerTransaction: int
            average length of transactions

    This class generates a temporal database based on the given parameters and provides
    options to output the database in either a text file or a DataFrame format.

    **Importing this algorithm into a Python program**


        from PAMI.extras.syntheticDataGenerator import TemporalDatabase as db

        temporalDB = db.TemporalDatabase(numOfTransactions, avgTransactionLength, numItems, outFileName, percentage, sep, occurrenceProbabilityAtSameTimestamp, occurrenceProbabilityToSkipSubsequentTimestamp)

        temporalDB.create()


    **Methods to execute code on terminal**


    Format:

        (.venv) $ python3 TemporalDatabase.py <numOfTransactions> <avgLenOfTransactions> <numItems> <outputFile> <percentage> <sep> <typeOfFile> <occurrenceProbabilityAtSameTimestamp> <occurrenceProbabilityToSkipSubsequentTimestamp>


    Example Usage:

        (.venv) $ python3 TemporalDatabase.py 50 10 100 temporal.txt 50 \t database 0.1 0.1


    :param numOfTransactions: int
        Number of transactions to generate.

    :param avgLenOfTransactions: int
        Average length of transactions.

    :param numItems: int
        Number of items in the database.

    :param outputFile: str
        Name of the output file for the database.

    :param percentage: int
        Percentage for the coin toss to decide if a transaction will be included in the output.
        If the value is greater than 1, it is treated as a percentage (i.e., 50 for 50%).

    :param sep: str
        Separator for the output file (default is tab).

    :param typeOfFile: str
        Type of output file. Can be 'database' for a text file or 'dataframe' for a DataFrame output.

    :param occurrenceProbabilityAtSameTimestamp: float
        Probability that a new transaction will occur at the same timestamp as the previous one.

    :param occurrenceProbabilityToSkipSubsequentTimestamp: float
        Probability that the timestamp will be skipped for subsequent transactions.
    """

    def __init__(self, numOfTransactions: int, avgLenOfTransactions: int,
                 numItems: int, outputFile: str, percentage: int = 50,
                 sep: str = '\t', typeOfFile: str = "Database",
                 occurrenceProbabilityAtSameTimestamp: float = 0.1,
                 occurrenceProbabilityToSkipSubsequentTimestamp: float = 0.1) -> None:
        """
        Initialize the TemporalDatabase with required parameters.

        :param numOfTransactions: Number of transactions to generate.
        :param avgLenOfTransactions: Average length of transactions.
        :param numItems: Number of items in the database.
        :param outputFile: Name of the output file for the database.
        :param percentage: Percentage for the coin toss to include transactions.
        :param sep: Separator for the output file.
        :param typeOfFile: Type of output file ('database' or 'dataframe').
        :param occurrenceProbabilityAtSameTimestamp: Probability for same timestamp.
        :param occurrenceProbabilityToSkipSubsequentTimestamp: Probability to skip subsequent timestamp.
        """

        self.databaseSize = databaseSize
        self.avgItemsPerTransaction = avgItemsPerTransaction
        self.numItems = numItems
        self.outputFile = outputFile
        if percentage > 1:
            self.percentage = percentage / 100
        else:
            self.percentage = percentage
        self.sep = sep
        self.typeOfFile = typeOfFile.lower()
        self.occurrenceProbabilityAtSameTimestamp = occurrenceProbabilityAtSameTimestamp
        self.occurrenceProbabilityToSkipSubsequentTimestamp = occurrenceProbabilityToSkipSubsequentTimestamp

    def getFileName(self) -> str:
        """
        Returns the name of the output file.

        :return: Output file name.
        """
        return self.outputFile

    def getDatabaseAsDataFrame(self) -> pd.DataFrame:
        """
        Returns the database as a DataFrame.

        :return: pd.DataFrame containing the temporal database.
        """
        return self.df

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
        Tune the array to ensure that the sum of the values equals sumRes.

        :param array: List of values to be tuned.
        :type array: list
        :param sumRes: Target sum for the array values.
        :type sumRes: int
        :return: Tuned list of values.
        """
        values = np.random.randint(1, self.numItems, len(array))

        while np.sum(values) != sumRes:
            if np.sum(values) > sumRes:
                maxIndex = np.argmax(values)
                values[maxIndex] -= 1
            else:
                minIndex = np.argmin(values)
                values[minIndex] += 1

        for i in range(len(array)):
            array[i][1] = values[i]

        return array

    def create(self) -> None:
        """
        Create the temporal database or DataFrame based on the specified type of file.
        """
        db = []
        lineSize = []

        self.current_timestamp = 0  # Initialize current timestamp

        for i in range(self.numOfTransactions):
            if self.performCoinFlip(self.occurrenceProbabilityAtSameTimestamp):
                timestamp = self.current_timestamp
            else:
                if self.performCoinFlip(self.occurrenceProbabilityToSkipSubsequentTimestamp):
                    self.current_timestamp += 2
                else:
                    self.current_timestamp += 1
                timestamp = self.current_timestamp

            db.append([timestamp])
            if self.performCoinFlip(self.percentage):
                lineSize.append([i, 0])

        sumRes = self.numOfTransactions * self.avgLenOfTransactions

        self.tuning(lineSize, sumRes)

        for i in range(len(lineSize)):
            if lineSize[i][1] > self.numItems:

                raise ValueError(
                    "Error: Either increase numItems or decrease avgLenOfTransactions or modify percentage")
            line = np.random.choice(range(1, self.numItems + 1), lineSize[i][1], replace=False)
            db[lineSize[i][0]].extend(line)

        if self.typeOfFile == "database":
            with open(self.outputFile, "w") as outFile:
                for line in db:
                    outFile.write(self.sep.join(map(str, line)) + '\n')

        if self.typeOfFile == "dataframe":
            data = {
                'timestamp': [line[0] for line in db],
                'transactions': pd.Series([line[1:] for line in db])
            }
            self.df = pd.DataFrame(data)

        print("Temporal database created successfully")


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print("Usage: python TemporalDatabase.py <numOfTransactions> <avgLenOfTransactions> <numItems> <outputFile> <percentage> <sep> <typeOfFile> <occurrenceProbabilityAtSameTimestamp> <occurrenceProbabilityToSkipSubsequentTimestamp>")
        sys.exit(1)

    obj = TemporalDatabase(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
        percentage=int(sys.argv[5]), sep=sys.argv[6], typeOfFile=sys.argv[7],
        occurrenceProbabilityAtSameTimestamp=float(sys.argv[8]),
        occurrenceProbabilityToSkipSubsequentTimestamp=float(sys.argv[9])
    )
    obj.create()
