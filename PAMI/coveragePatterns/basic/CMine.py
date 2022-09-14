from PAMI.coveragePatterns.basic import abstract as _ab

class CMine(_ab._coveragePatterns):
    """
        CMine algorithms aims to discover the coverage patterns in transactional databases.

    Reference:
    ---------
        P. Gowtham Srinivas, P. Krishna Reddy, A. V. Trinath, Bhargav Sripada, R. Uday Kiran:
        Mining coverage patterns from transactional databases. J. Intell. Inf. Syst. 45(3): 423-439 (2015)
    
    Attributes:
    -----------
        self.iFile : str
            Input file name or path of the input file
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
            However, the users can override their default separator.
        self.oFile : str
            Name of the output file or path of the output file
        self.startTime:float
            To record the start time of the mining process
        self.endTime:float
            To record the completion time of the mining process
        self.finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        self.memoryUSS : float
            To store the total amount of USS memory consumed by the program
        self.memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        self.Database : list
            To store the complete set of transactions available in the input database/file
    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of coverage patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of coverage patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        createCoverageItems()
            Generate coverage items
        tidToBitset(itemset)
            Convert tid list to bit set
        genPatterns(prefix, tidData)
            Generate coverage patterns
        generateAllPatterns(coverageItems)
            Generate all coverage patterns

    Executing the code on terminal:
    -------
        Format:
        ------
            python3 CMine.py <inputFile> <outputFile> <minRF> <minCS> <maxOR> <'\t'>

        Examples:
        --------
            python3 CMine.py sampleTDB.txt patterns.txt 0.4 0.7 0.5 ','

        Sample run of importing the code:
        -------------------

            from PAMI.coveragePattern.basic import CMine as alg

            obj = alg.CMine(iFile, minRF, minCS, maxOR, seperator)

            obj.startMine()

            coveragePatterns = obj.getPatterns()

            print("Total number of coverage Patterns:", len(coveragePatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.
    """

    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _minCS = str()
    _minRF = str()
    _maxOR = str()
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _mapSupport = {}
    _lno = 0


    def _convert(self, value):
        """
        To convert the user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = value
        if type(value) is str:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        return value

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        self._mapSupport = {}
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            self._lno += 1
                            splitter = [i.rstrip() for i in line.split(self._sep)]
                            splitter = [x for x in splitter if x]
                            self._Database.append(splitter)
                except IOError:
                    print("File Not Found")

    def creatingCoverageItems(self):
        """
        This function creates coverage items from _database.
        :return: coverageTidData that stores coverage items and their tid list.
        """
        tidData = {}
        self._lno = 0
        for transaction in self._Database:
            self._lno = self._lno + 1
            for item in transaction[1:]:
                if item not in tidData:
                    tidData[item] = [self._lno]
                else:
                    tidData[item].append(self._lno)
        coverageTidData = {k: v for k, v in tidData.items() if len(v) / len(self._Database) >= self._minRF}
        coverageTidData = dict(sorted(coverageTidData.items(), reverse=True, key=lambda x: len(x[1])))
        return coverageTidData

    def tidToBitset(self,itemset):
        """
        This function converts tid list to bitset.
        :param itemset:
        :return:
        """
        bitset = {}

        for k,v in itemset.items():
            bitset[k] = 0b1
            bitset[k] = (bitset[k] << int(v[0])) | 0b1
            for i in range(1,len(v)):
                diff = int(v[i]) - int(v[i-1])
                bitset[k] = (bitset[k] << diff) | 0b1
            bitset[k] = (bitset[k] << (self._lno - int(v[i])))
        return bitset

    def genPatterns(self,prefix,tidData):
        """
        This function generate coverage pattern about prefix.
        :param prefix: String
        :param tidData: list
        :return:
        """
        # variables to store coverage item set and
        itemset = prefix[0]

        # Get the length of tidData
        length = len(tidData)
        for i in range(length):
            tid = prefix[1] & tidData[i][1]
            tid1 = prefix[1] | tidData[i][1]
            andCount = bin(tid).count("1") - 1
            orCount = bin(tid1).count("1") - 1
            if orCount/len(self._Database) >= self._minCS and andCount / len(str(prefix[1])):
                coverageItemset = itemset + '\t' + tidData[i][0]
                if orCount / len(self._Database) >= self._minRF:
                    self._finalPatterns[coverageItemset] = andCount
                self.genPatterns((coverageItemset,tid),tidData[i+1:length])

    def generateAllPatterns(self,coverageItems):
        """
        This function generates all coverage patterns.
        :param coverageItems: coverage items
        :return:
        """
        tidData = list(coverageItems.items())
        length = len(tidData)
        for i in range(length):
            #print(i,tidData[i][0])
            self.genPatterns(tidData[i],tidData[i+1:length])

    def startMine(self):
        """ Main method to start """

        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._creatingItemSets()
        self._minCS = self._convert(self._minCS)
        self._minRF =  self._convert(self._minRF)
        self._maxOR = self._convert(self._maxOR)
        coverageItems = self.creatingCoverageItems()
        self._finalPatterns = {k: len(v) for k, v in coverageItems.items()}
        coverageItemsBitset = self.tidToBitset(coverageItems)
        self.generateAllPatterns(coverageItemsBitset)
        self.save('output.txt')
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Coverage patterns were generated successfully using CMine  algorithm")

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
        """Storing final coverage patterns in a dataframe
        :return: returning coverage patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """Complete set of coverage patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x.strip() + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of coverage patterns after completion of the mining process
        :return: returning coverage patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Coverage Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__=="__main__":
    _ap = str()
    if len(_ab._sys.argv) == 7 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 7:
            _ap = CMine(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = CMine(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.startMine()
        print("Total number of coverage Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
