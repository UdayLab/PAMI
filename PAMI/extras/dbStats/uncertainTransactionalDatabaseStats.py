import statistics
import pandas as pd
import validators
import numpy as np
from urllib.request import urlopen
import PAMI.extras.graph.plotLineGraphFromDictionary as plt

class uncertainTransactionalDatabaseStats:
    """
        :Description:
         ------------
         uncertainTransactionalDatabaseStats is class to get stats of database.
        
        Attributes:
        ----------
        inputFile : file
            input file path
        database : dict
            store time stamp and its transaction
        lengthList : list
            store length of all transaction
        sep : str
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
        getVarianceTransactionLength()
            get the variance of transaction length
        getSparsity()
            get the sparsity of database
        getSortedListOfItemFrequencies()
            get sorted list of item frequencies
        getSortedListOfTransactionLength()
            get sorted list of transaction length
        save(data, outputFile)
            store data into outputFile
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

    def run(self):
        self.readDatabase()

    def readDatabase(self):
        """
        read database from input file and store into database and size of each transaction.
        """
        # self.creatingItemSets()
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
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = line.split(':')
                    temp1 = [i.rstrip() for i in temp[0].split(self.sep)]
                    temp1 = [x for x in temp if x]
                    self.database[numberOfTransaction] = temp1
            else:
                try:
                    with open(self.inputFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            numberOfTransaction += 1
                            line = line.strip()
                            temp = line.split(':')
                            temp1 = [i for i in temp[0].split(self.sep)]
                            self.database[numberOfTransaction] = temp1
                except IOError:
                    print("File Not Found")
                    quit()
        self.lengthList = [len(s) for s in self.database.values()]

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
        for tid in self.database:
            for item in self.database[tid]:
                itemFrequencies[item] = itemFrequencies.get(item, 0)
                itemFrequencies[item] += 1
        return {k: v for k, v in sorted(itemFrequencies.items(), key=lambda x: x[1], reverse=True)}

    def getFrequenciesInRange(self):
        fre = self.getSortedListOfItemFrequencies()
        rangeFrequencies = {}
        maximum = max([i for i in fre.values()])
        values = [int(i * maximum / 6) for i in range(1, 6)]
        va = len({key: val for key, val in fre.items() if val > 0 and val < values[0]})
        rangeFrequencies[va] = values[0]
        for i in range(1, len(values)):
            va = len({key: val for key, val in fre.items() if val < values[i] and val > values[i - 1]})
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
        itemFrequencies = self.getFrequenciesInRange()
        transactionLength = self.getTransanctionalLengthDistribution()
        plt.plotLineGraphFromDictionary(itemFrequencies, 100, 'Frequency', 'No of items', 'frequency')
        plt.plotLineGraphFromDictionary(transactionLength, 100, 'transaction length', 'transaction length', 'frequency')


if __name__ == '__main__':

    obj = uncertainTransactionalDatabaseStats('Uncertain_T10.csv', '\t')
    obj.run()
    obj.printStats()
    obj.plotGraphs()
