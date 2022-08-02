from PAMI.stablePeriodicFrequentPattern.basic import abstract as _ab


class SPPEclat:
    _iFile = " "
    _oFile = " "
    _minSup = str()
    _maxPer = str()
    _maxLa = float()
    _sep = " "
    _SPPList = {}
    _itemList = []
    _last = int()
    _finalPatterns = {}
    _tsList = {}
    _startTime = float()
    _endTime = float()
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []

    def __init__(self, inputFile, minSup, maxPer, maxLa, sep='\t'):
        self._iFile = inputFile
        self._minSup = minSup
        self._maxPer = maxPer
        self._maxLa = maxLa
        self._sep = sep

    def _readDatabase(self):
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            if 'Patterns' in i:
                self._Database = self._iFile['Patterns'].tolist()
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
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """
        to convert the type of user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value

    def _createSPPList(self):
        tidLast = {}
        la = {}
        for transaction in self._Database:
            ts = int(transaction[0])
            for item in transaction[1:]:
                if item not in self._SPPList:
                    la[item] = max(0, ts - self._maxPer)
                    self._SPPList[item] = [1, la[item]]
                    self._tsList[item] = [ts]
                else:
                    s = self._SPPList[item][0] + 1
                    la[item] = max(0, la[item] + ts - tidLast.get(item) - self._maxPer)
                    self._SPPList[item] = [s, max(la[item], self._SPPList[item][1])]
                    self._tsList[item].append(ts)
                tidLast[item] = ts
            self._last = ts
        for item in self._SPPList:
            la[item] = max(0, la[item] + self._last - tidLast[item] - self._maxPer)
            self._SPPList[item][1] = max(la[item], self._SPPList[item][1])
        self._SPPList = {k: v for k, v in self._SPPList.items() if v[0] >= self._minSup and v[1] <= self._maxLa}
        self._SPPList = {k: v for k, v in sorted(self._SPPList.items(), key=lambda x: x[1][0], reverse=True)}
        self._GPPF_DFS(list(self._SPPList), set())

    def _GPPF_DFS(self, GPPFList, CP):
        for i in range(len(GPPFList)):
            item = GPPFList[i]
            CP1 = CP | {item}
            if CP != set():
                self._tsList['\t'.join(CP1)] = list(set(self._tsList['\t'.join(CP)]) & set(self._tsList[item]))
            la = self._calculateLa(self._tsList['\t'.join(CP1)])
            support = len(self._tsList['\t'.join(CP1)])
            if la <= self._maxLa and len(self._tsList['\t'.join(CP1)]) >= self._minSup:
                #CP = CP1
                self._finalPatterns['\t'.join(CP1)] = [support, la]
                if i+1 < len(GPPFList):
                    self._GPPF_DFS(GPPFList[i+1:], CP1)

    def _calculateLa(self, tsList):
        previous = 0
        la = 0
        tsList = sorted(tsList)
        laList = []
        for ts in tsList:
            la = max(0, la + ts - previous - self._maxPer)
            laList.append(la)
            previous = ts
            
        la = max(0, la + self._last - previous - self._maxPer)
        laList.append(la)
        maxla = max(laList)
        return maxla

    def startMine(self):
        self._startTime = _ab._time.time()
        self._readDatabase()
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        self._maxLa = self._convert(self._maxLa)
        #print(self._minSup, self._maxPer, self._maxLa)
        self._createSPPList()
        print(len(self._SPPList))
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Stable Periodic Frequent patterns were generated successfully using SPPECLAT algorithm ")


    def getRuntime(self):
        return self._endTime - self._startTime

    def getPatterns(self):
        return self._finalPatterns

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def savePatterns(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS


if __name__ == '__main__':
    _ap = str()
    if len(_ab._sys.argv) == 6 or len(_ab._sys.argv) == 7:
        if len(_ab._sys.argv) == 7:
            _ap = SPPEclat(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = SPPEclat(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Patterns:", len(_Patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        '''ap = SPPEclat('https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_retail.csv', 0.001,
                       0.005, 0.004)
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.savePatterns('/Users/Likhitha/Downloads/output')
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)'''
        ap = SPPEclat("/Users/masuyudai/Code/Dataset/temporal_T10I4D100K.csv", 500, 300, 900)
        # ap = SPPEclat("/Users/masuyudai/Code/Spark/PartialPeriodicFrequent_wrong/sample.csv", 4,2,1," ")
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        print(ap.getRuntime())
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        # for k,v in Patterns.items():
        #     print(k,v)


        # print("Error! The number of input parameters do not match the total number of parameters provided")
