from pyspark import SparkConf, SparkContext
from PAMI.frequentPattern.pyspark import abstract as _ab


class parallelECLAT(_ab._frequentPatterns):
    """
        Attributes
        ----------
            iFile : file
                Input file name or path of the input file
            oFile : file
                Name of the output file or the path of output file
            minSup : int
                minSup is a minimum support
            numPartitions : int
                The number of partitions
                On each worker node, an executor process is started and this process performs processing.
                The processing unit of worker node is partition
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
            lno : int
                the number of transactions
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
            genPatterns(suffix, pattern, data)
                Generate frequent pattern from the arguments
        Executing the code on terminal:
        -------------------------------
            Format:
            ------

                python3 parallelECLAT.py <inputFile> <outputFile> <minSup> <numWorkers>

            Examples:
            ---------

                python3 parallelECLAT.py sampleDB.txt patterns.txt 10.0 3   (minSup will be considered in times of minSup and count of database transactions)

                python3 paralleECLAT.py sampleDB.txt patterns.txt 10 3    (minSup will be considered in support count or frequency)

        Sample run of the importing code:
        ---------------------------------

            import PAMI.frequentPattern.pyspark.parallelECLAT as alg

            obj = alg.parallelECLAT(iFile, minSup, numWorkers)

            obj.startMine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.save(oFile)

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
    _numPartitions = int()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _lno = int()

    def __init__(self, iFile, minSup, numWorkers, sep='\t'):
        super().__init__(iFile, minSup, int(numWorkers), sep)

    def _genPatterns(self, suffix, pattern, data):
        freqPatterns = {}
        index = data.index(suffix)
        for i in range(index + 1, len(data)):
            tid = pattern[1].intersection(data[i][1])
            if len(tid) >= self._minSup:
                freqPattern = pattern[0] + ' ' + data[i][0]
                freqPatterns[freqPattern] = len(tid)
                freqPatterns.update(self._genPatterns(data[i], (freqPattern, tid), data))
        return freqPatterns

    def _convert(self, value):
        """
        To convert the user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        elif type(value) is float:
            value = (self._lno * value)
        elif type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._lno * value)
            else:
                value = int(value)
        else:
            print("None")
        return value

    def startMine(self):
        """
        Frequent pattern mining process will start from here
        """

        self._startTime = _ab._time.time()
        conf = SparkConf().setAppName("Parallel ECLAT").setMaster("local[*]")
        sc = SparkContext(conf=conf)

        data = sc.textFile(self._iFile, self._numPartitions) \
            .map(lambda line: [int(y) for y in line.rstrip().split(self._sep)]).persist()
        self._lno = data.count()
        self._minSup = self._convert(self._minSup)

        frequentItems = None
        if 'transactional' in self._iFile:
            frequentItems = data.zipWithIndex() \
                .flatMap(lambda x: [(str(item), x[1]) for item in x[0]]) \
                .groupByKey() \
                .filter(lambda x: len(x[1]) >= self._minSup) \
                .sortBy(lambda x: len(x[1])) \
                .mapValues(set) \
                .persist()
            data.unpersist()
        elif 'temporal' in self._iFile:
            frequentItems = data.flatMap(lambda trans: [(str(item), trans[0]) for item in trans[1:]]) \
                .groupByKey() \
                .filter(lambda x: len(x[1]) >= self._minSup) \
                .mapValues(set) \
                .persist()
            data.unpersist()
        else:
            pass
            # print("may be not able to process the input file")

        freqItems = dict(frequentItems.collect())
        self._finalPatterns = {k: len(v) for k, v in freqItems.items()}

        freqPatterns = list(frequentItems.map(lambda x: self._genPatterns(x, x, list(freqItems.items())))
                            .filter(lambda x: len(x) != 0).collect())

        for value in freqPatterns:
            self._finalPatterns.update(value)

        temp = {}
        for pattern, v in self._finalPatterns.items():
            s = ""
            if isinstance(pattern, str):
                s += pattern.replace(' ', '\t') + '\t'
            else:
                for item in pattern:
                    s += item + '\t'
            temp[s] = v
        self._finalPatterns = temp

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Parallel ECLAT algorithm")
        sc.stop()

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
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = parallelECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = parallelECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
