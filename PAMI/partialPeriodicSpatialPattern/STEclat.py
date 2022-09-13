from PAMI.partialPeriodicSpatialPattern import abstract as _ab


class STEclat(_ab._partialPeriodicSpatialPatterns):
    """

    ...

    Reference:
    ---------
        R. Uday Kiran, C. Saideep, K. Zettsu, M. Toyoda, M. Kitsuregawa and P. Krishna Reddy,
        "Discovering Partial Periodic Spatial Patterns in Spatiotemporal Databases," 2019 IEEE International
        Conference on Big Data (Big Data), 2019, pp. 233-238, doi: 10.1109/BigData47090.2019.9005693.

    Attributes :
    ----------
            iFile : str
                Input file name or path of the input file
            nFile: str:
               Name of Neighbourhood file name
            maxIAT: float or int or str
                The user can specify maxIAT either in count or proportion of database size.
                If the program detects the data type of maxIAT is integer, then it treats maxIAT is expressed in count.
                Otherwise, it will be treated as float.
                Example: maxIAT=10 will be treated as integer, while maxIAT=10.0 will be treated as float
            minPS: float or int or str
                The user can specify minPS either in count or proportion of database size.
                If the program detects the data type of minPS is integer, then it treats minPS is expressed in count.
                Otherwise, it will be treated as float.
                Example: minPS=10 will be treated as integer, while minPS=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
                However, the users can override their default separator.
            startTime:float
                To record the start time of the mining process
            endTime:float
                To record the completion time of the mining process
            finalPatterns: dict
                Storing the complete set of patterns in a dictionary variable
            oFile : str
                Name of the output file to store complete set of frequent patterns
            memoryUSS : float
                To store the total amount of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program
            Database : list
                To store the complete set of transactions available in the input database/file
    Methods :
    -------
            startMine()
                Mining process will start from here
            getPatterns()
                Complete set of patterns will be retrieved with this function
            save(oFile)
                Complete set of frequent patterns will be loaded in to a output file
            getPatternsAsDataFrames()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            creatingItemSets(iFileName)
                Storing the complete transactions of the database/input file in a database variable
            frequentOneItem()
                Generating one frequent patterns
            convert(value):
                To convert the given user specified value
            getNeighbourItems(keySet):
                A function to get common neighbours of a itemSet
             mapNeighbours(file):
                A function to map items to their neighbours

    Executing the code on terminal :
    ------------------------------
        Format:
            python3 STEclat.py <inputFile> <outputFile> <neighbourFile>  <minPS>  <maxIAT> 
        Examples:
            python3 STEclat.py sampleTDB.txt output.txt sampleN.txt 0.2 0.5 (maxIAT & minPS will be considered in percentage of database transactions)

            python3 STEclat.py sampleTDB.txt output.txt sampleN.txt  5 3 ( maxIAT & minPS will be considered in support count or frequency)
                                                                (it considers "\t" as separator)

            python3 STEclat.py sampleTDB.txt output.txt sampleN.txt 3 2 ',' (it will consider "," as a separator)

    Sample run of importing the code :
    -------------------------------

        import PAMI.partialPeriodicSpatialPattern.STEclat as alg

        obj = alg.STEclat("sampleTDB.txt", "sampleN.txt", 3, 4)

        obj.startMine()

        partialPeriodicSpatialPatterns = obj.getPatterns()

        print("Total number of Periodic Spatial Frequent Patterns:", len(partialPeriodicSpatialPatterns))

        obj.save("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by P. Likhitha under the supervision of Professor Rage Uday Kiran.
    """

    _maxIAT = " "
    _minPS = " "
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _nFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _sep = "\t"
    _lno = 0

    def __init__(self, iFile, nFile, minPS, maxIAT, sep="\t"):
        super().__init__(iFile, nFile, minPS, maxIAT,  sep)
        self._NeighboursMap = {}

    def _creatingItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable

            """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            data, ts = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                self._Database.append(tr)

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

    # function to get frequent one pattern
    def _frequentOneItem(self):
        """Generating one frequent patterns"""
        self._tidList = {}
        self._mapSupport = {}
        self._maxIAT = self._convert(self._maxIAT)
        for line in self._Database:
            s = line
            n = int(s[0])
            for i in range(1, len(s)):
                si = s[i]
                if self._mapSupport.get(si) is None:
                    self._mapSupport[si] = [0, n]
                    self._tidList[si] = [n]
                else:
                    lp = n - self._mapSupport[si][1]
                    if lp <= self._maxIAT:
                        self._mapSupport[si][0] += 1
                    self._mapSupport[si][1] = n
                    self._tidList[si].append(n)
        self._minPS = self._convert(self._minPS)
        self._mapSupport = {k: v[0] for k, v in self._mapSupport.items() if v[0] >= self._minPS}
        plist = [key for key, value in sorted(self._mapSupport.items(), key=lambda x: x[1], reverse=True)]
        return plist

    def _convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
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

    def _getPeriodicSupport(self, timeStamps):
        """calculates the support and periodicity with list of timestamps

            :param timeStamps: timestamps of a pattern
            :type timeStamps: list
        """
        timeStamps.sort()
        per = 0
        for i in range(len(timeStamps) - 1):
            j = i + 1
            if abs(timeStamps[j] - timeStamps[i]) <= self._maxIAT:
                per += 1
        return per

    def _save(self, prefix, suffix, tidSetX):
        """Saves the patterns that satisfy the periodic frequent property.

            :param prefix: the prefix of a pattern
            :type prefix: list or None
            :param suffix: the suffix of a patterns
            :type suffix: list
            :param tidSetX: the timestamp of a patterns
            :type tidSetX: list


        """
        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        val = self._getPeriodicSupport(tidSetX)
        if val >= self._minPS:
            self._finalPatterns[tuple(prefix)] = val

    def _Generation(self, prefix, itemSets, tidSets):
        if len(itemSets) == 1:
            i = itemSets[0]
            tidi = tidSets[0]
            self._save(prefix, [i], tidi)
            return
        for i in range(len(itemSets)):
            itemI = itemSets[i]
            if itemI is None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemI]
            for j in range(i + 1, len(itemSets)):
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetX).intersection(tidSetJ))
                val = self._getPeriodicSupport(y)
                if val >= self._minPS:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newprefix = list(set(itemSetX)) + prefix
            self._Generation(newprefix, classItemSets, classTidSets)
            self._save(prefix, list(set(itemSetX)), tidSetX)

    def _getNeighbourItems(self, keySet):
        """
            A function to get Neighbours of a item
            :param keySet:itemSet
            :type keySet:str or tuple
            :return: set of common neighbours
            :rtype:set
        """
        itemNeighbours = self._NeighboursMap.keys()
        if isinstance(keySet, str):
            if self._NeighboursMap.get(keySet) is None:
                return []
            itemNeighbours = list(set(itemNeighbours).intersection(set(self._NeighboursMap.get(keySet))))
        if isinstance(keySet, tuple):
            keySet = list(keySet)
            for j in range(0, len(keySet)):
                i = keySet[j]
                itemNeighbours = list(set(itemNeighbours).intersection(set(self._NeighboursMap.get(i))))
        return itemNeighbours

    def mapNeighbours(self):
        """
            A function to map items to their Neighbours
        """
        self._NeighboursMap = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data = []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for i in data:
                self._NeighboursMap[i[0]] = i[1:]
        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._NeighboursMap[temp[0]] = temp[1:]
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._NeighboursMap[temp[0]] = temp[1:]
                except IOError:
                    print("File Not Found")
                    quit()

    def startMine(self):
        """Frequent pattern mining process will start from here"""

        # global items_sets, endTime, startTime
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._creatingItemSets()
        #self._minSup = self._convert(self._minSup)
        self.mapNeighbours()
        self._finalPatterns = {}
        plist = self._frequentOneItem()
        for i in range(len(plist)):
            itemX = plist[i]
            tidSetX = self._tidList[itemX]
            itemSetX = [itemX]
            itemSets = []
            tidSets = []
            neighboursItems = self._getNeighbourItems(plist[i])
            for j in range(i + 1, len(plist)):
                if not plist[j] in neighboursItems:
                    continue
                itemJ = plist[j]
                tidSetJ = self._tidList[itemJ]
                y1 = list(set(tidSetX).intersection(tidSetJ))
                val = self._getPeriodicSupport(y1)
                if val >= self._minPS:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self._Generation(itemSetX, itemSets, tidSets)
            self._save(None, itemSetX, tidSetX)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm")

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
            pat = ""
            for i in a:
                pat += str(i) + '\t'
            data.append([pat, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'periodicSupport'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            pat = ""
            for i in x:
                pat += str(i) + '\t'
            patternsAndSupport = pat.strip() + ": " + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of  Spatial Partial Periodic Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 6 or len(_ab._sys.argv) == 7:
        if len(_ab._sys.argv) == 7:
            _ap = STEclat(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = STEclat(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.startMine()
        print("Total number of Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:",  _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")



