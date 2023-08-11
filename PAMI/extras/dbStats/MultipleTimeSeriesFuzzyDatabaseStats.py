import statistics
import pandas as pd
import validators
import numpy as np
from urllib.request import urlopen
import sys
import PAMI.extras.graph.plotLineGraphFromDictionary as plt


class MultipleTimeSeriesFuzzyDatabaseStats:
    """
        :Description:  transactionalDatabaseStats is class to get stats of database.


        :param inputFile: file :
            input file path
        :param database: dict :
            store time stamp and its transaction
        :param lengthList: list :
            store size of all transaction
        :param sep: str
            separator in file. Default is tab space.

           """

    def __init__(self, inputFile, sep='\t'):
        """
        :param inputFile: input file name or path
        :type inputFile: str
        """
        self.inputFile = inputFile
        self.lengthList = []
        self.sep = sep
        self.database = {}
        self.itemFrequencies = {}

    def run(self):
        self.readDatabase()

    def readDatabase(self):
        """
        read database from input file and store into database and size of each transaction.
        """
        self._transactions, self._fuzzyValues, self._Database, self._ts = [], [], [], []
        numberOfTransaction = 0
        if isinstance(self.inputFile, pd.DataFrame):
            if self.inputFile.empty:
                print("its empty..")
            i = self.inputFile.columns.values.tolist()
            if 'tid' in i and 'Transactions' in i:
                self.database = self.inputFile.set_index('tid').T.to_dict(orient='records')[0]
            if 'tid' in i and 'Patterns' in i:
                self.database = self.inputFile.set_index('tid').T.to_dict(orient='records')[0]
        if isinstance(self.inputFile, str):
            if validators.url(self.inputFile):
                data = urlopen(self.inputFile)
                for line in data:
                    numberOfTransaction += 1
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.database[numberOfTransaction] = temp
            else:
                try:
                    with open(self.inputFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            parts = line.split(":")
                            numberOfTransaction += 1
                            parts[0] = parts[0].strip()
                            parts[1] = parts[1].strip()
                            parts[2] = parts[2].strip()
                            times = parts[0].split(self.sep)
                            items = parts[1].split(self.sep)
                            quantities = parts[2].split(self.sep)
                            #print(times, items, quantities)
                            _time = [x for x in times if x]
                            items = [x for x in items if x]
                            quantities = [float(x) for x in quantities if x]
                            tempList = []
                            for k in range(len(_time)):
                                ite = "(" + _time[k] + "," + items[k] + ")"
                                tempList.append(ite)
                            self._ts.append([x for x in times])
                            self._transactions.append([x for x in tempList])
                            self._fuzzyValues.append([x for x in quantities])
                            self.database[numberOfTransaction] = tempList
                except IOError:
                    print("File Not Found")
                    quit()
        self.lengthList = [len(s) for s in self._transactions]

    def getDatabaseSize(self):
        """
        get the size of database
        :return: data base size
        """
        return len(self.database)

    def getTotalNumberOfItems(self):
        """
        get the number of items in database.
        :return: number of items
        """
        return len(self.getSortedListOfItemFrequencies())

    def getMinimumTransactionLength(self):
        """
        get the minimum transaction length
        :return: minimum transaction length
        """
        return min(self.lengthList)

    def getAverageTransactionLength(self):
        """
        get the average transaction length. It is sum of all transaction length divided by database length.
        :return: average transaction length
        """
        totalLength = sum(self.lengthList)
        return totalLength / len(self.database)

    def getMaximumTransactionLength(self):
        """
        get the maximum transaction length
        :return: maximum transaction length
        """
        return max(self.lengthList)

    def getStandardDeviationTransactionLength(self):
        """
        get the standard deviation transaction length
        :return: standard deviation transaction length
        """
        return statistics.pstdev(self.lengthList)

    def getVarianceTransactionLength(self):
        """
        get the variance transaction length
        :return: variance transaction length
        """
        return statistics.variance(self.lengthList)

    def getNumberOfItems(self):
        """
        get the number of items in database.
        :return: number of items
        """
        return len(self.getSortedListOfItemFrequencies())

    def convertDataIntoMatrix(self):
        singleItems = self.getSortedListOfItemFrequencies()
        # big_array = np.zeros((self.getDatabaseSize(), len(self.getSortedListOfItemFrequencies())))
        itemsets = {}
        for i in self.database:
            for item in singleItems:
                if item in itemsets:
                    if item in self.database[i]:
                        itemsets[item].append(1)
                    else:
                        itemsets[item].append(0)
                else:
                    if item in self.database[i]:
                        itemsets[item] = [1]
                    else:
                        itemsets[item] = [0]
        # new = pd.DataFrame.from_dict(itemsets)
        data = list(itemsets.values())
        an_array = np.array(data)
        return an_array

    def getSparsity(self):
        """
        get the sparsity of database. sparsity is percentage of 0 of database.
        :return: database sparsity
        """
        big_array = self.convertDataIntoMatrix()
        n_zeros = np.count_nonzero(big_array == 0)
        return (n_zeros / big_array.size)

    def getDensity(self):
        """
        get the sparsity of database. sparsity is percentage of 0 of database.
        :return: database sparsity
        """
        big_array = self.convertDataIntoMatrix()
        n_zeros = np.count_nonzero(big_array != 0)
        return (n_zeros / big_array.size)

    def getSortedListOfItemFrequencies(self):
        """
        get sorted list of item frequencies
        :return: item frequencies
        """
        itemFrequencies = {}
        for line in range(len(self._transactions)):
            times = self._ts[line]
            items = self._transactions[line]
            quantities = self._fuzzyValues[line]
            for i in range(0, len(items)):
                item = items[i]
                if item in itemFrequencies:
                    itemFrequencies[item] += quantities[i]
                else:
                    itemFrequencies[item] = quantities[i]
        self.itemFrequencies = {k: v for k, v in sorted(itemFrequencies.items(), key=lambda x: x[1], reverse=True)}
        return self.itemFrequencies
    
    def getFrequenciesInRange(self):
        fre = self.getSortedListOfItemFrequencies()
        rangeFrequencies = {}
        maximum = max([i for i in fre.values()])
        values = [int(i*maximum/6) for i in range(1,6)]
        va = len({key: val for key, val in fre.items() if val > 0 and val < values[0]})
        rangeFrequencies[va] = values[0]
        for i in range(1,len(values)):
            va = len({key: val for key, val in fre.items() if val < values[i] and val > values[i-1]})
            rangeFrequencies[va] = values[i]
        return rangeFrequencies

    def getTransanctionalLengthDistribution(self):
        """
        get transaction length
        :return: transaction length
        """
        transactionLength = {}
        for length in self.lengthList:
            transactionLength[length] = transactionLength.get(length, 0)
            transactionLength[length] += 1
        return {k: v for k, v in sorted(transactionLength.items(), key=lambda x: x[0])}

    def save(self, data, outputFile):
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
                   
    def printStats(self):
        print(f'Database size (total no of transactions) : {self.getDatabaseSize()}')
        print(f'Number of items : {self.getNumberOfItems()}')
        print(f'Minimum Transaction Size : {self.getMinimumTransactionLength()}')
        print(f'Average Transaction Size : {self.getAverageTransactionLength()}')
        print(f'Maximum Transaction Size : {self.getMaximumTransactionLength()}')
        print(f'Standard Deviation Transaction Size : {self.getStandardDeviationTransactionLength()}')
        print(f'Variance in Transaction Sizes : {self.getVarianceTransactionLength()}')
        print(f'Sparsity : {self.getSparsity()}')
  
    def plotGraphs(self):
        # itemFrequencies = self.getFrequenciesInRange()
        transactionLength = self.getTransanctionalLengthDistribution()
        plt.plotLineGraphFromDictionary(self.itemFrequencies, 100, 0, 'Frequency', 'No of items', 'frequency')
        plt.plotLineGraphFromDictionary(transactionLength, 100, 0, 'transaction length', 'transaction length', 'frequency')


if __name__ == '__main__':
    import PAMI.extras.graph.plotLineGraphFromDictionary as plt
    import pandas as pd
    obj = MultipleTimeSeriesFuzzyDatabaseStats(sys.argv[1])
    obj.run()
    obj.printStats()
    obj.plotGraphs()
