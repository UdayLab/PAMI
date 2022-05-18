from PAMI.frequentPattern.pyspark import abstract as _ab


class parallelApriori(_ab._frequentPatterns):
    """
    Attributes
    ----------
        iFile : file
            Input file name or path of the input file
        oFile : file
            Name of the output file or the path of output file
        minSup : float
            minSup is a proportion of database size.
        numWorkers : int
            the number of workers
        finalPatterns : dict
            Storing the complete set of patterns in a dictionary variable
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        storePatternsInFile(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsInDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        getAllFrequentPatterns(database,frequentItemsets)
            This function generates all frequent patterns
        genCandidateItemsets(frequentPatterns, length)
            This function generates candidate patterns from the frequentPatterns
        Mapper(transaction,candidateItemsets)
            This function map each itemset of candidateItemsets to (itemset,1) if itemset is in transaction

    Executing the code on terminal:
    -------------------------------
            
            Format:
            ------
            
                python3 parallelApriori.py <inputFile> <outputFile> <minSup> <numWorkers>
            
            Examples:
            ---------
                python3 parallelApriori.py sampleDB.txt patterns.txt 10.0 3   (minSup will be considered in times of minSup and count of database transactions)
                
                python3 parallelApriori.py sampleDB.txt patterns.txt 10 3     (minSup will be considered in support count or frequency)
        
   Sample run of the importing code:
   ---------------------------------
            
            import PAMI.frequentPattern.pyspark.parallelApriori as alg
            
            obj = alg.parallelApriori(iFile, minSup, numWorkers)
            
            obj.startMine()
            
            frequentPatterns = obj.getPatterns()
            
            print("Total number of Frequent Patterns:", len(frequentPatterns))
            
            obj.savePatterns(oFile)
            
            Df = obj.getPatternInDataFrame()
            
            memUSS = obj.getMemoryUSS()
            
            print("Total Memory in USS:", memUSS)
            
            memRSS = obj.getMemoryRSS()
            
            print("Total Memory in RSS", memRSS)
            
            run = obj.getRuntime()
            
            print("Total ExecutionTime in seconds:", run)
        Credits:
        --------
            The complete program was written by Yudai Masu  under the supervision of Professor Rage Uday Kiran.
    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _numWorkers = int()

    def __init__(self, iFile, minSup, numWorkers, sep = '\t'):
         super().__init__(iFile, float(minSup), int(numWorkers), sep)

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def savePatterns(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = str(x) + " : " + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    @staticmethod
    def _Mapper(transaction, candidateItemsets):
        """
        Map each candidate itemset of candidateItemsets to (itemset,1) if a candidate itemset is in transaction

        :param transaction: a transaction of database
        :type transaction: set
        :param candidateItemsets: candidate item sets
        :type candidateItemsets: list
        :return:set
        """

        candidates = set()
        for itemset in candidateItemsets:
            if set(itemset).issubset(transaction):
                candidates.add((itemset, 1))
        return candidates

    @staticmethod
    def _genCandidateItemsets(frequentPatterns, length):
        """
        Generate candidate itemsets from frequentPatterns

        :param frequentPatterns: set of all frequent patterns to generate candidate patterns of each of size is length
        :type frequentPatterns: list
        :param length: size of each candidate patterns to be generated
        :type length: int
        :return: list of candidate patterns
        :rtype: list
        """
        candidates = list(_ab._c(frequentPatterns, 2))
        candidates = set([tuple(set(item[0]).union(set(item[1]))) for item in [x for x in candidates]])
        candidates = list({item for item in candidates if len(item) == length})
        return candidates

    def _genFrequentItems(self, database):
        """
        Get frequent items which length is 1

        :param database: database
        :type database: RDD
        :return: frequent items which length is 1
        :rtype: dict
        """
        frequentItems = dict(database.flatMap(lambda x: [(item, 1) for item in x])
                             .reduceByKey(lambda x, y: x + y)
                             .filter(lambda c: c[1] >= self._minSup)
                             .collect())
        return frequentItems

    def _getAllFrequentPatterns(self, database, frequentItems):
        """
        Get all frequent patterns and save them to self.oFile

        :param database: database
        :type : RDD
        :param frequentItems: dict
        :return:
        """

        # Get candidate patterns that length is 2
        candidates = list(_ab._c(frequentItems.keys(), 2))
        length = 3
        while len(candidates) != 0:
            # if each itemset of candidates is in each transaction, then create (itemset,1)
            mappedDatabase = database.flatMap(lambda transaction: self._Mapper(transaction, candidates))

            # aggregate the values by key by reduceByKey() method
            frequentPatterns = dict(
                mappedDatabase.reduceByKey(lambda x, y: x + y).filter(lambda c: c[1] >= self._minSup).collect())
            self._finalPatterns.update(frequentPatterns)
            candidates = self._genCandidateItemsets(list(frequentPatterns.keys()), length)
            length += 1
            
    def _convert(self, dataLength, value):
        """
        To convert the user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (dataLength * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (dataLength * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        Frequent pattern mining process will start from here

        :return:
        """
        self._startTime = _ab._time.time()

        # setting SparkConf and SparkContext to process in parallel
        conf = _ab._SparkConf().setAppName("parallelApriori").setMaster("local[*]")
        sc = _ab._SparkContext(conf=conf)

        # read database from iFile
        database = sc.textFile(self._iFile, self._numWorkers).map(lambda x: {int(y) for y in x.rstrip().split(self._sep)})

        # Calculating minSup as a percentage
        self._minSup = self._convert(database.count(), self._minSup)
        oneFrequentItems = self._genFrequentItems(database)
        self._finalPatterns = oneFrequentItems
        self._getAllFrequentPatterns(database, oneFrequentItems)

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Parallel Apriori algorithm")
        sc.stop()
        


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = parallelApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = parallelApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _finalPatterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_finalPatterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
