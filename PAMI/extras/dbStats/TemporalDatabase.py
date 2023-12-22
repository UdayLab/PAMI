# TemporalDatabase is a class used to get stats of database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.dbStats import TemporalDatabase as db
#
#     obj = db.TemporalDatabase(iFile, "\t")
#
#     obj.save(oFile)
#
#     obj.run()
#
#     obj.printStats()
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

import sys
import statistics
import pandas as pd
import validators
import numpy as np
from urllib.request import urlopen
from typing import Dict, Union


class TemporalDatabase:
    """
        Description:
        -------------
            TemporalDatabase is class to get stats of database.

        :param inputFile : file
            input file path

        :param sep : str
            separator in file. Default is tab space.

        Methods:
        -------
        run()
            execute readDatabase function
        readDatabase()
            read database from input file
        getDatabaseSize()
            get the size of database
        getMinimumTransactionLength()
            get the minimum transaction length
        getAverageTransactionLength()
            get the average transaction length. It is sum of all transaction length divided by database length.
        getMaximumTransactionLength()
            get the maximum transaction length
        getStandardDeviationTransactionLength()
            get the standard deviation of transaction length
        getSortedListOfItemFrequencies()
            get sorted list of item frequencies
        getSortedListOfTransactionLength()
            get sorted list of transaction length
        save(data, outputFile)
            store data into outputFile
        getMinimumPeriod()
            get the minimum period
        getAveragePeriod()
            get the average period
        getMaximumPeriod()
            get the maximum period
        getStandardDeviationPeriod()
            get the standard deviation period
        getNumberOfTransactionsPerTimestamp()
            get number of transactions per time stamp. This time stamp range is 1 to max period.

        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

                    from PAMI.extras.dbStats import TemporalDatabase as db

                    obj = db.TemporalDatabase(iFile, "\t")

                    obj.save(oFile)

                    obj.run()

                    obj.printStats()

    """

    def __init__(self, inputFile: Union[str, pd.DataFrame], sep: str = '\t') -> None:
        """
        :param inputFile: input file name or path
        :type inputFile: str
        :param sep:
        """
        self.inputFile = inputFile
        self.database = {}
        self.lengthList = []
        self.timeStampCount = {}
        self.periodList = []
        self.sep = sep
        self.periods = {}

    def run(self) -> None:
        self.readDatabase()

    def readDatabase(self) -> None:
        """
        read database from input file and store into database and size of each transaction.
        And store the period between transactions as list
        """
        numberOfTransaction = 0
        if isinstance(self.inputFile, pd.DataFrame):
            if self.inputFile.empty:
                print("its empty..")
            i = self.inputFile.columns.values.tolist()
            if 'TS' in i and 'Transactions' in i:
                self.database = self.inputFile.set_index('ts').T.to_dict(orient='records')[0]
            if 'TS' in i and 'Patterns' in i:
                self.database = self.inputFile.set_index('ts').T.to_dict(orient='records')[0]
            self.timeStampCount = self.inputFile.groupby('ts').count().T.to_dict(orient='records')[0]

        if isinstance(self.inputFile, str):
            if validators.url(self.inputFile):
                data = urlopen(self.inputFile)
                for line in data:
                    numberOfTransaction += 1
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.database[numberOfTransaction] = temp[1:]
                    self.timeStampCount[int(temp[0])] = self.timeStampCount.get(int(line[0]), 0)
                    self.timeStampCount[int(temp[0])] += 1
            else:
                try:
                    with open(self.inputFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            numberOfTransaction += 1
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            if len(temp) > 0:
                                self.database[numberOfTransaction] = temp[1:]
                                self.timeStampCount[int(temp[0])] = self.timeStampCount.get(int(line[0]), 0)
                                self.timeStampCount[int(temp[0])] += 1
                except IOError:
                    print("File Not Found")
                    quit()
        self.lengthList = [len(s) for s in self.database.values()]
        timeStampList = sorted(list(self.database.keys()))
        preTimeStamp = 0
        for ts in timeStampList:
            self.periodList.append(int(ts) - preTimeStamp)
            preTimeStamp = ts

        for x, y in self.database.items():
            for i in y:
                if i not in self.periods:
                    self.periods[i] = [x, x]
                else:
                    self.periods[i][0] = max(self.periods[i][0], x - self.periods[i][1])
                    self.periods[i][1] = x
        for key in self.periods:
            self.periods[key][0] = max(self.periods[key][0], abs(len(self.database) - self.periods[key][1]))
        self.periods = {k: v[0] for k, v in self.periods.items()}

    def getDatabaseSize(self) -> int:
        """
        get the size of database
        :return: database size
        """
        return len(self.database)

    def getMinimumTransactionLength(self) -> int:
        """
        get the minimum transaction length
        :return: minimum transaction length
        """
        return min(self.lengthList)

    def getAverageTransactionLength(self) -> float:
        """
        get the average transaction length. It is sum of all transaction length divided by database length.
        :return: average transaction length
        """
        totalLength = sum(self.lengthList)
        return totalLength / len(self.database)

    def getMaximumTransactionLength(self) -> int:
        """
        get the maximum transaction length
        :return: maximum transaction length
        """
        return max(self.lengthList)

    def getStandardDeviationTransactionLength(self) -> float:
        """
        get the standard deviation transaction length
        :return: standard deviation transaction length
        """
        return statistics.pstdev(self.lengthList)

    def getVarianceTransactionLength(self) -> float:
        """
        get the variance transaction length
        :return: variance transaction length
        """
        return statistics.variance(self.lengthList)

    def convertDataIntoMatrix(self) -> np.ndarray:
        singleItems = self.getSortedListOfItemFrequencies()
        itemsets = {}
        for tid in self.database:
            for item in singleItems:
                if item in itemsets:
                    if item in self.database[tid]:
                        itemsets[item].append(1)
                    else:
                        itemsets[item].append(0)
                else:
                    if item in self.database[tid]:
                        itemsets[item] = [1]
                    else:
                        itemsets[item] = [0]
        data = list(itemsets.values())
        an_array = np.array(data)
        return an_array

    def getSparsity(self) -> float:
        """
        get the sparsity of database. sparsity is percentage of 0 of database.
        :return: database sparsity
        """
        big_array = self.convertDataIntoMatrix()
        n_zeros = np.count_nonzero(big_array == 0)
        return (n_zeros / big_array.size)

    def getDensity(self) -> float:
        """
        get the sparsity of database. sparsity is percentage of 0 of database.
        :return: database sparsity
        """
        big_array = self.convertDataIntoMatrix()
        n_zeros = np.count_nonzero(big_array == 1)
        return (1.0 - n_zeros / big_array.size)

    def getTotalNumberOfItems(self) -> int:
        """
        get the number of items in database.
        :return: number of items
        """
        return len(self.getSortedListOfItemFrequencies())

    def getSortedListOfItemFrequencies(self) -> Dict[str, int]:
        """
        get sorted list of item frequencies
        :return: item frequencies
        """
        itemFrequencies = {}
        for tid in self.database:
            for item in self.database[tid]:
                itemFrequencies[item] = itemFrequencies.get(item, 0)
                itemFrequencies[item] += 1
        return {k: v for k, v in sorted(itemFrequencies.items(), key=lambda x: x[1], reverse=True)}

    def getFrequenciesInRange(self) -> Dict[int, int]:
        fre = self.getSortedListOfItemFrequencies()
        rangeFrequencies = {}
        maximum = max([i for i in fre.values()])
        values = [int(i * maximum / 6) for i in range(1, 6)]
        # print(maximum)
        va = len({key: val for key, val in fre.items() if val > 0 and val < values[0]})
        rangeFrequencies[va] = values[0]
        for i in range(1, len(values)):
            va = len({key: val for key, val in fre.items() if val < values[i] and val > values[i - 1]})
            rangeFrequencies[va] = values[i]
        return rangeFrequencies

    def getPeriodsInRange(self) -> Dict[int, int]:
        fre = {k: v for k, v in sorted(self.periods.items(), key=lambda x: x[1])}
        rangePeriods = {}
        maximum = max([i for i in fre.values()])
        values = [int(i * maximum / 6) for i in range(1, 6)]
        # print(maximum)
        va = len({key: val for key, val in fre.items() if val > 0 and val < values[0]})
        rangePeriods[va] = values[0]
        for i in range(1, len(values)):
            va = len({key: val for key, val in fre.items() if val < values[i] and val > values[i - 1]})
            rangePeriods[va] = values[i]
        return rangePeriods

    def getTransanctionalLengthDistribution(self) -> Dict[int, int]:
        """
        get transaction length
        :return: transaction length
        """
        transactionLength = {}
        for length in self.lengthList:
            transactionLength[length] = transactionLength.get(length, 0)
            transactionLength[length] += 1
        return {k: v for k, v in sorted(transactionLength.items(), key=lambda x: x[0])}

    def save(self, data: dict, outputFile: str) -> None:
        """
        store data into outputFile
        :param data: input data
        :type data: dict
        :param outputFile: output file name or path to store
        :type outputFile: str
        """
        with open(outputFile, 'w') as f:
            for key, value in data.items():
                f.write(f'{key}\t{value}\n')

    def getMinimumInterArrivalPeriod(self) -> int:
        """
        get the minimum inter arrival period
        :return: minimum inter arrival period
        """
        return min(self.periodList)

    def getAverageInterArrivalPeriod(self) -> float:
        """
        get the average inter arrival period. It is sum of all period divided by number of period.
        :return: average inter arrival period
        """
        totalPeriod = sum(self.periodList)
        return totalPeriod / len(self.periodList)

    def getMaximumInterArrivalPeriod(self) -> int:
        """
        get the maximum inter arrival period
        :return: maximum inter arrival period
        """
        return max(self.periodList)

    def getMinimumPeriodOfItem(self) -> int:
        """
        get the minimum period of the item
        :return: minimum period
        """
        return min([i for i in self.periods.values()])

    def getAveragePeriodOfItem(self) -> float:
        """
        get the average period of the item
        :return: average period
        """
        return sum([i for i in self.periods.values()]) / len(self.periods)

    def getMaximumPeriodOfItem(self) -> int:
        """
        get the maximum period of the item
        :return: maximum period
        """
        return max([i for i in self.periods.values()])

    def getStandardDeviationPeriod(self) -> float:
        """
        get the standard deviation period
        :return: standard deviation period
        """
        return statistics.pstdev(self.periodList)

    def getNumberOfTransactionsPerTimestamp(self) -> Dict[int, int]:
        """
        get number of transactions per time stamp
        :return: number of transactions per time stamp as dict
        """
        maxTS = max(list(self.timeStampCount.keys()))
        return {ts: self.timeStampCount.get(ts, 0) for ts in range(1, maxTS + 1)}

    def printStats(self) -> None:
        print(f'Database size : {self.getDatabaseSize()}')
        print(f'Number of items : {self.getTotalNumberOfItems()}')
        print(f'Minimum Transaction Size : {self.getMinimumTransactionLength()}')
        print(f'Average Transaction Size : {self.getAverageTransactionLength()}')
        print(f'Maximum Transaction Size : {self.getMaximumTransactionLength()}')
        print(f'Minimum Inter Arrival Period : {self.getMinimumInterArrivalPeriod()}')
        print(f'Average Inter Arrival Period : {self.getAverageInterArrivalPeriod()}')
        print(f'Maximum Inter Arrival Period : {self.getMaximumInterArrivalPeriod()}')
        print(f'Minimum periodicity : {self.getMinimumPeriodOfItem()}')
        print(f'Average periodicity : {self.getAveragePeriodOfItem()}')
        print(f'Maximum periodicicty : {self.getMaximumPeriodOfItem()}')
        print(f'Standard Deviation Transaction Size : {self.getStandardDeviationTransactionLength()}')
        print(f'Variance : {self.getVarianceTransactionLength()}')
        print(f'Sparsity : {self.getSparsity()}')

    def plotGraphs(self) -> None:
        itemFrequencies = self.getFrequenciesInRange()
        transactionLength = self.getTransanctionalLengthDistribution()
        plt.plotLineGraphFromDictionary(itemFrequencies, 100, 0, 'Frequency', 'no of items', 'frequency')
        plt.plotLineGraphFromDictionary(transactionLength, 100, 0, 'transaction length', 'transaction length',
                                        'frequency')


if __name__ == '__main__':
    data = {'tid': [1, 2, 3, 4, 5, 6, 7],

            'Transactions': [['a', 'd', 'e'], ['b', 'a', 'f', 'g', 'h'], ['b', 'a', 'd', 'f'], ['b', 'a', 'c'],
                             ['a', 'd', 'g', 'k'],

                             ['b', 'd', 'g', 'c', 'i'], ['b', 'd', 'g', 'e', 'j']]}

    # data = pd.DataFrame.from_dict('temporal_T10I4D100K.csv')
    import PAMI.extras.graph.plotLineGraphFromDictionary as plt

    if len(sys.argv) < 3:
        print("Please provide two arguments.")
    else:
        obj = TemporalDatabase(sys.argv[1], sys.argv[2])
        obj1 = TemporalDatabase(pd.DataFrame(data))
        obj1.run()
        if obj1.getDatabaseSize() > 0:
            obj1.printStats()
            obj1.plotGraphs()
        else:
            print("No data found in the database.")
