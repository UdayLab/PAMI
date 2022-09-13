#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.


from PAMI.frequentPattern.basic import abstract as _ab

class ECLATbitset(_ab._frequentPatterns):
    """
        ECLATbitset is one of the fundamental algorithm to discover frequent patterns in a transactional database.

    Reference:
    ----------


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
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        createFrequentItems()
            Generate frequent items
        tidToBitset(itemset)
            Convert tid list to bit set
        genPatterns(prefix, tidData)
            Generate frequent patterns
        genAllFrequentPatterns(frequentItems)
            Generate all frequent patterns
    """

    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _minSup = str()
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
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
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
        self._minSup = self._convert(self._minSup)

    def creatingFrequentItems(self):
        """
        This function creates frequent items from _database.
        :return: frequentTidData that stores frequent items and their tid list.
        """
        tidData = {}
        self._lno = 0
        for transaction in self._Database:
            self._lno = self._lno + 1
            for item in transaction:
                if item not in tidData:
                    tidData[item] = [self._lno]
                else:
                    tidData[item].append(self._lno)
        frequentTidData = {k: v for k, v in tidData.items() if len(v) >= self._minSup}
        frequentTidData = dict(sorted(frequentTidData.items(), reverse=True, key=lambda x: len(x[1])))
        return frequentTidData

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
        This function generate frequent pattern about prefix.
        :param prefix: String
        :param tidData: list
        :return:
        """
        # variables to store frequent item set and
        itemset = prefix[0]

        # Get the length of tidData
        length = len(tidData)

        for i in range(length):
            #tid = prefix[1].intersection(tidData[i][1])
            tid = prefix[1] & tidData[i][1]
            count = bin(tid).count("1") - 1
            #tidLength = len(tid)
            if count >= self._minSup:
                frequentItemset = itemset + '\t' + tidData[i][0]
                self._finalPatterns[frequentItemset] = count
                self.genPatterns((frequentItemset,tid),tidData[i+1:length])

    def genAllFrequentPatterns(self,frequentItems):
        """
        This function generates all frequent patterns.
        :param frequentItems: frequent items
        :return:
        """
        tidData = list(frequentItems.items())
        length = len(tidData)
        for i in range(length):
            #print(i,tidData[i][0])
            self.genPatterns(tidData[i],tidData[i+1:length])

    def startMine(self):
        """Frequent pattern mining process will start from here
                We start with the scanning the itemSets and store the bitsets respectively.
                We form the combinations of single items and  check with minSup condition to check the frequency of patterns
                """

        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")

        self._creatingItemSets()
        frequentItems = self.creatingFrequentItems()
        self._finalPatterns = {k: len(v) for k, v in frequentItems.items()}
        frequentItemsBitset = self.tidToBitset(frequentItems)
        self.genAllFrequentPatterns(frequentItemsBitset)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Eclat_bitset algorithm")

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
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x.strip() + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())

if __name__=="__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = ECLATbitset(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = ECLATbitset(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
