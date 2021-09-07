import statistics

class utilityDatabaseStats:
    """
    utilityDatabaseStats is class to get stats of database.

        Attributes:
        ----------
        inputFile : file
            input file path
        database : dict
            store time stamp and its transaction
        lengthList : list
            store length of all transaction
        utility : dict
            store utility each item
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
        getSortedListOfItemFrequencies()
            get sorted list of item frequencies
        getSortedListOfTransactionLength()
            get sorted list of transaction length
        storeInFile(data, outputFile)
            store data into outputFile
        getMinimumUtility()
            get the minimum utility
        getAverageUtility()
            get the average utility
        getMaximumUtility()
            get the maximum utility
        getSortedUtilityValuesOfItem()
            get sorted utility values each item
    """
    def __init__(self, inputFile, sep='\t'):
        """
        :param inputFile: input file name or path
        :type inputFile: str
        """
        self.inputFile = inputFile
        self.database = {}
        self.lengthList = []
        self.utility = {}
        self.sep = sep

    def run(self):
        self.readDatabase()

    def readDatabase(self):
        """
        read database from input file and store into database and size of each transaction.
        """
        numberOfTransaction = 0
        with open(self.inputFile, 'r') as f:
            for line in f:
                numberOfTransaction += 1
                line = [s for s in line.strip().split(':')]
                transaction = [s for s in line[0].split(self.sep)]
                utilities = [int(s) for s in line[2].split(self.sep)]
                self.database[numberOfTransaction] = transaction
                for i in range(len(transaction)):
                    self.utility[transaction[i]] = self.utility.get(transaction[i],0)
                    self.utility[transaction[i]] += utilities[i]
        self.lengthList = [len(s) for s in self.database.values()]
        self.utility = {k: v for k, v in sorted(self.utility.items(), key=lambda x:x[1], reverse=True)}

    def getDatabaseSize(self):
        """
        get the size of database
        :return: data base size
        """
        return len(self.database)

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

    def getSparsity(self):
        # percentage of 0 dense dataframe
        """
        get the sparsity of database
        :return: database sparsity
        """
        matrixSize = self.getDatabaseSize()*len(self.getSortedListOfItemFrequencies())
        return (matrixSize - sum(self.getSortedListOfItemFrequencies().values())) / matrixSize

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
        return {k: v for k, v in sorted(itemFrequencies.items(), key=lambda x:x[1], reverse=True)}

    def getTransanctionalLengthDistribution(self):
        """
        get transaction length
        :return: transaction length
        """
        transactionLength = {}
        for length in self.lengthList:
            transactionLength[length] = transactionLength.get(length, 0)
            transactionLength[length] += 1
        return {k: v for k, v in sorted(transactionLength.items(), key=lambda x:x[0])}

    def storeInFile(self, data, outputFile):
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

    def getTotalUtility(self):
        """
        get sum of utility
        :return: total utility
        """
        return sum(list(self.utility.values()))

    def getMinimumUtility(self):
        """
        get the minimum utility
        :return: minimum utility
        """
        return min(list(self.utility.values()))

    def getAverageUtility(self):
        """
        get the average utility
        :return: average utility
        """
        return sum(list(self.utility.values())) / len(self.utility)

    def getMaximumUtility(self):
        """
        get the maximum utility
        :return: maximum utility
        """
        return max(list(self.utility.values()))

    def getSortedUtilityValuesOfItem(self):
        """
        get sorted utility value each item. key is item and value is utility of item
        :return: sorted dictionary utility value of item
        """
        return self.utility