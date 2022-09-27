from PAMI.sequentialPatternMining.basic import abstract as _ab


class _Sequence:
    def __init__(self, sequence, support):
        """
        Attributes:
            sequence : list
                A input sequence on which we will create the Sequence object
            support : int
                Support of the given sequence
        """
        self.sequence = []
        self.support = support
        self.place_holder = '_'
        for element in sequence:
            self.sequence.append(list(element))

    def add(self, seq):
        """
        Used when projecting the database by extending a sequence.
        """
        if seq.sequence[0][0] == self.place_holder:
            fEle = seq.sequence[0]
            fEle.remove(self.place_holder)
            self.sequence[-1].extend(fEle)
            print(self.sequence[-1])
            self.sequence.extend(seq.sequence[1:])
        else:
            self.sequence.extend(seq.sequence)
            if self.support == None:
                self.support = seq.support
        self.support = min(self.support, seq.support)


class prefixSpan(_ab._sequentialPatterns):
    """
        PrefixSpan is one of the basic algorithm to discover the frequent sequential patterns.

    Reference:
    ---------
        J. Pei, J. Han, B. Mortazavi-Asl, J. Wang, H. Pinto, Q. Chen, U. Dayal,
        M. Hsu: Mining Sequential Patterns by Pattern-Growth: The PrefixSpan Approach.
        IEEE Trans. Knowl. Data Eng. 16(11): 1424-1440 (2004)

    Attribute:
    ---------
        iFile : str
            Input file name or path of the input file
        oFile : str
            Name of the output file or the path of output file
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        Database : list
            To store the transactions of a database in list

    Methods:
    -------
        startMine()
            Start the Mining Process By calling the recursion function
        recursion()
            The main Function to startMine the database level by level
        projectDataBase()
            This function will project the input databse based on the input pattern
        validateLengths()
            Find whether all the transactions in the sequence is less than the maximum pattern length
        getFrequentItems()
            This function will return the frequent items based in the minimum support that is called in the recursion function
        getTime()
            return the time taken for mining
        getMemory()
            return the maximum memory utilized while mining
    """
    _minSup = float()
    _maxPatternLength = int()
    _PLACEHOLDER = '_'
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            temp = []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                temp = self._iFile['Transactions'].tolist()

            for k in temp:
                self._Database.append(set(k))
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.strip()
                    tempLis = line.split(' -1 ')
                    tempLis = tempLis[:-1]
                    mLis = []
                    for ele in tempLis:
                        mLis.append(list(map(int, ele.split(self._sep))))
                    self._Database.append(mLis)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            tempLis = line.split(' -1 ')
                            tempLis = tempLis[:-1]
                            mLis = []
                            for ele in tempLis:
                                mLis.append(list(map(int, ele.split(self._sep))))
                            self._Database.append(mLis)
                except IOError:
                    print("File Not Found")
                    quit()

    def _powerset(self, iterable):
        s = list(iterable)
        return list(_ab._chain.from_iterable(_ab._combinations(s, r) for r in range(len(s) + 1)))

    def _recur(self, output, n, subs, lis):
        if n == len(subs):
            temp = str(output).replace('(), ', '')
            temp = temp.replace(', ()', '')
            temp = temp.replace(',)', ')')
            if temp != '(())' or len(temp) <= 2:
                lis.append(temp)
            return
        for x in subs[n]:
            self._recur(output + (x,), n + 1, subs, lis)

    def _isSubSequence1(self, string1, string2, m, n):
        if m == 0:
            return True
        if n == 0:
            return False
        if string1[m - 1] == string2[n - 1]:
            return self._isSubSequence(string1, string2, m - 1, n - 1)
        return self._isSubSequence(string1, string2, m, n - 1)

    def _isSubSequence(self, string1, string2, m, n):
        if m == 0:
            return 1
        if n == 0:
            return 0
        if string1[m - 1] == string2[n - 1]:
            return self._isSubSequence(string1, string2, m - 1, n - 1)
        return self._isSubSequence(string1, string2, m, n - 1)

    def _FPMUtil(self, sequences, outputFile):
        allPossibleSequences = []
        mainDict = {}
        for sequence in sequences:
            subs = []
            for x in sequence:
                subs.append(list(self._powerset(x)))

            i = len(subs)
            lis = []
            output = ()
            self._recur(output, 0, subs, lis)
            allPossibleSequences.extend(lis)

        allPossibleSequences = list(set(allPossibleSequences))

        for seq in allPossibleSequences:
            for row in sequences:
                row = str(row).replace(',)', ')')
                seq1 = seq[1:-1].split('), ')
                row1 = row[1:-1].split('), ')
                seq1 = [x + ')' for x in seq if x[-1] != ')']
                row1 = [x + ')' for x in row if x[-1] != ')']
                if (mainDict.get(seq)):
                    mainDict[seq] += self._isSubSequence(seq, row, len(seq), len(row))
                else:
                    mainDict[seq] = self._isSubSequence(seq, row, len(seq), len(row))
        self._finalPatterns = mainDict

    def _recursion(self, pattern, inp, threshold, maxPatternLength):
        '''The main Function to startMine and project the database at each level
            @ pattern : Pattern to build upon
            @ inp : Input Database at the current level
            @ threshold : Minimum support
            @ maxPattern Length : Maxium Pattern length
        '''
        resultList = []
        if self._validateLengths(pattern, maxPatternLength):
            freqList = self._getFrequentItems(inp, pattern, threshold, maxPatternLength)
            for item in freqList:
                retVal = _Sequence(pattern.sequence, pattern.support)
                retVal.add(item)
                if self._validateLengths(pattern, maxPatternLength):
                    resultList.append(retVal)
                projectedDB = self._projectDatabase(inp, retVal)
                newPatterns = self._recursion(retVal, projectedDB, threshold, maxPatternLength)
                resultList.extend(newPatterns)
        return resultList

    def _validateLengths(self, pattern, maxPatternLength):
        '''
            Find whether all the transactions in the sequence is less than the maximum pattern length
        '''
        for item in pattern.sequence:
            if len(item) > maxPatternLength:
                return False
        return True

    def _projectDatabase(self, S, pattern):
        '''
            This function will project the input databse based on the input pattern
        '''
        projectedDB = []
        lastEle = pattern.sequence[-1]
        last_item = lastEle[-1]
        for s in S:
            projectTempDB = []
            for element in s:
                isPrefix = False
                if self._PLACEHOLDER in element:
                    if last_item in element and len(pattern.sequence[-1]) > 1:
                        isPrefix = True
                else:
                    isPrefix = True
                    for item in lastEle:
                        if item not in element:
                            isPrefix = False
                            break
                if isPrefix:
                    currentIndex = s.index(element)
                    lastIndex = element.index(last_item)
                    if lastIndex == len(element) - 1:
                        projectTempDB = s[currentIndex + 1:]
                    else:
                        projectTempDB = s[currentIndex:]
                        index = element.index(last_item)
                        slicedEle = element[lastIndex:]
                        slicedEle[0] = self._PLACEHOLDER
                        projectTempDB[0] = slicedEle
                    break
            if len(projectTempDB) != 0:
                projectedDB.append(projectTempDB)
        return projectedDB

    '''def _loadDataFromFile(self, filename):
        sequences = []
        with open(filename) as f:
            for x in f:
                tempLis = x.split(' -1 ')
                tempLis = tempLis[:-1]
                mLis = []
                for ele in tempLis:
                    mLis.append(list(map(int, ele.split())))
                sequences.append(mLis)

        self.db = sequences'''

    def _save(self, output, lis):
        with open(output, 'w+') as file:
            i = 0
            for x in lis:
                st = ' -1 '.join(list(map(lambda z: str(z)[1:-1], x.sequence)))
                st += f' : {x.support}'
                st += '\n'
                st = st.replace(', ', '\t')
                file.write(st)

    def _getFrequentItems(self, S, pattern, threshold, maxPatternLength):
        """
            This function will return the frequent items based in the minimum support that is called in the recursion function
            @ pattern : Pattern to check for the frequency
            @ S : Input Database at the current level
            @ threshold : Minimum support
            @ maxPattern Length : Maxium Pattern length
        """
        items = {}
        parentItemDict = {}
        freqList = []
        if S is None or len(S) == 0:
            return []
        if len(pattern.sequence) != 0:
            lastEle = pattern.sequence[-1]
        else:
            lastEle = []
        for s in S:
            isPrefix = True
            for item in lastEle:
                if item not in s[0]:
                    isPrefix = False
                    break
            if isPrefix and len(lastEle) > 0:
                index = s[0].index(lastEle[-1])
                if index < len(s[0]) - 1:
                    for item in s[0][index + 1:]:
                        if item in parentItemDict:
                            parentItemDict[item] += 1
                        else:
                            parentItemDict[item] = 1
            if self._PLACEHOLDER in s[0]:
                for item in s[0][1:]:
                    if item in parentItemDict:
                        parentItemDict[item] += 1
                    else:
                        parentItemDict[item] = 1
                s = s[1:]
            counted = []
            for element in s:
                for item in element:
                    if item not in counted:
                        counted.append(item)
                        if item in items:
                            items[item] += 1
                        else:
                            items[item] = 1

        freqList.extend([_Sequence([[self._PLACEHOLDER, key]], value)
                         for key, value in parentItemDict.items() if value >= threshold])
        freqList.extend([_Sequence([[key]], value)
                         for key, value in items.items() if value >= threshold])

        freqList = [item for item in freqList if self._validateLengths(item, maxPatternLength)]
        return sorted(freqList, key=lambda x: x.support)

    def startMine(self):
        '''startMine and return the frequent sequence patterns.. '''
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._minSup = _ab._math.ceil(self._minSup * len(self._Database))
        if self._minSup == 1:
            mainTupe = []
            for x in self._Database:
                temp = []
                for y in x:
                    temp.append(tuple(y))
                temp = tuple(temp)
                mainTupe.append(temp)
            mainTupe = tuple(mainTupe)
            self._FPMUtil(mainTupe, self._oFile)
            #print('oFile: ', self._oFile)
        result = self._recursion(_Sequence([], None),
                                self._Database,
                                _ab._math.ceil(self._minSup * len(self._Database)),
                                self._maxPatternLength)
        self.endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print(result)
        return result

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
        # dataFrame = dataFrame.replace(r'\r+|\n+|\t+',' ', regex=True)
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Frequent Sequential Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())

if __name__ == "__main__":

    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = prefixSpan(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = prefixSpan(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        print("Total number of Sequential Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        _ap = prefixSpan('/Users/Likhitha/Downloads/MAIN_SPMF/InpDataPrefixSpan', 0.6, 100, ' ')
        _ap.startMine()
        print("Total number of Sequential Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save('/Users/Likhitha/Downloads/MAIN_SPMF/output.txt')
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
        print("Error! The number of input parameters do not match the total number of parameters provided")
