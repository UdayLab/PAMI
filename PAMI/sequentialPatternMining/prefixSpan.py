from PAMI.sequentialPatternMining import abstract as _ab


class prefixSpan(_ab._frequentPatterns):
    """
    Attributes:
    ----------
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
            patternBuffer : list
                To store the items and used to help while saving the patterns
            containsItemSetsWithMultipleItems : boolean
                Check whether the dataset contains multiple items or not

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
        minSupAbs()
            Convert the percentage value minimumSupport to the float or integer value
        save()
            save the patterns in the finalPatterns dictionary
        save1()
            save the patterns in the finalPatterns dictionary
        scanDataBase()
            scans the database and assign the database to the variable in the form of a list
        PseudoSequence()
            returns the parameters in the form of the list
        findSequenceContainingItems()
            returns the dictionary in which contains the single items with their sequence Ids
        prefixSpanWithSingleItems()
            removes the infrequent items and start the process to find the patterns with single items
        prefixSpanWithMultipleItems()
            removes the infrequent items and start the process to find the patterns with multiple items
        buildProjectedDataBaseSingleItems()
            build the projected database for the single items
        buildProjectedDataBaseFirstTimeMultipleItems()
            build the projected database for the multiple items
        findAllFrequentPairsSingleItems()
            calculates the all possible pattern pairs for the single items
        findAllFrequentPairs()
            calculate the all possible pattern pairs for the multiple items
        recursionSingleItems()
            recursive process will be done for the pattern if it is frequent for the single items
        recursion()
            recursive process will be done for the pattern if it is frequent for the multiple items
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
    _Database = []
    _patternBuffer = [None] * 50
    _containsItemSetsWithMultipleItems = False

    def _minSupAbs(self, file, support):
        """
        calculates and return the integer or float value for minimumSupport
        :param file: input file
        :param support: minimumSupport value as percentage
        """
        with open(file, 'r') as f:
            read = f.readlines()
        self.minSup = int(_ab._math.ceil(support * len(read)))

    def _save(self, item, support):
        """
        saves the frequent sequential pattern  in the 'output.txt' file
        :param item: item
        :param support: support value
        """
        a = ""
        a += str(item)
        a += " -1"
        self._finalPatterns[a] = str(support)

    def _save1(self, lastBufPos, pseudoSeq):
        """
        saves the frequent sequential pattern  in the 'output.txt' file
        :param lastBufPos: buffer position to save the sequential frequent pattern
        :param pseudoSeq: sequences containing the item
        """
        i = 0
        a = ""
        while i <= lastBufPos:
            a += str(self._patternBuffer[i])
            a += " "
            i += 1
        if self._patternBuffer[lastBufPos] != -1:
            a += str(-1)
        self._finalPatterns[a] = str(len(pseudoSeq))

    def _scanDataBase(self):
        """
        scans the database and stores the items in the list
        :return: returns the database in the form of the list
        """

        with open(self._iFile, 'r') as f:
            read = f.readlines()
            lis = []
            for i in read:
                li = list(map(int, i.split()))
                lis.append(li)
        print(lis, type(lis))
        return lis

    @staticmethod
    def _PseudoSequence(seqId, indFirItem):
        """
        returns the parameters in the form of the list
        :param seqId: sequence id in the database
        :param indFirItem: index of the item in the sequence
        :return: return the parameters in the form of the list
        """
        return [seqId, indFirItem]

    def _findSequenceContainItems(self, data):
        """
        calculates the items and their sequences contains single items in the database
        :param data: input data in the form of the list
        :return: returns the items contains single items in the database
        """
        sequenceID = _ab._dd(list)
        for i in range(len(data)):
            sequence = data[i]
            itemCount = 0
            for j in sequence:
                if j >= 0:
                    sequenceIDs = sequenceID[j]
                    if len(sequenceIDs) == 0 or sequenceIDs[len(sequenceIDs) - 1] != i:
                        sequenceIDs.append(i)
                    itemCount += 1
                    if itemCount > 1:
                        self.containsItemSetsWithMultipleItems = True
                elif j == -1:
                    itemCount = 0
        return sequenceID

    def _prefixSpanWithSingleItems(self, seqId, data):
        """
        remove the infrequent items and starts the process to find frequent the sequential patterns
        :param seqId: sequence containing the items and their sequences
        :param data: input data in the form of the list
        """
        for i in range(len(data)):
            seq = data[i]
            curPos = 0
            lis = []
            for j in seq:
                if j >= 0:
                    if j in seqId:
                        isFreq = len(seqId[j]) >= self.minSup
                        if isFreq:
                            lis.append(j)
                            curPos += 1
                elif j == -2:
                    if curPos > 0:
                        lis.append(-2)
                        data[i] = lis
                        continue
                    else:
                        data[i] = []
        self.Database = data
        for i in list(sorted(seqId)):
            if len(seqId[i]) >= self.minSup:
                self._save(i, len(seqId[i]))
                self._patternBuffer[0] = i
                projDataB = self._buildProjectedDatabaseSingleItems(i, seqId[i])
                self._recursionSingleItems(self._patternBuffer, projDataB, 0)

    def _prefixSpanWithMultipleItems(self, seqId, data):
        """
        remove the infrequent items and starts the process to find frequent the sequential patterns
        :param seqId: sequence containing the items and their sequences
        :param data: input data in the form of the list
        """
        for i in range(len(data)):
            seq = data[i]
            curPos = 0
            curItemCount = 0
            lis = []
            for j in seq:
                if j >= 0:
                    if j in seqId:
                        isFreq = len(seqId[j]) >= self._minSup
                        if isFreq:
                            lis.append(j)
                            curPos += 1
                            curItemCount += 1
                elif j == -1:
                    if curItemCount > 0:
                        lis.append(-1)
                        curPos += 1
                        curItemCount = 0
                elif j == -2:
                    if curPos > 0:
                        lis.append(-2)
                        data[i] = lis
                        continue
                    else:
                        data[i] = []
        self._Database = data
        for i in list(sorted(seqId)):
            if len(seqId[i]) >= self.minSup:
                self._save(i, len(seqId[i]))
                self._patternBuffer[0] = i
                projDataB = self._buildProjectedDatabaseFirstTimeMultipleItems(i, seqId[i])
                self._recursion(self._patternBuffer, projDataB, 0)

    def _buildProjectedDatabaseSingleItems(self, item, seqId):
        """
        build the projected database for the item
        :param item: item in the database
        :param seqId: sequence of the item in the database
        :return: return the projected database in the form of the list
        """
        proDataBase = []
        for i in seqId:
            k = self.Database[i]
            for j in range(len(k)):
                token = k[j]
                if token != -2:
                    if token == item:
                        if k[j + 1] != -2:
                            proData = self._PseudoSequence(i, j + 1)
                            proDataBase.append(proData)
                            continue
        return proDataBase

    def _buildProjectedDatabaseFirstTimeMultipleItems(self, item, seqId):
        """
        build the projected database for the item
        :param item: item in the database
        :param seqId: sequence of the item in the database
        :return: return the projected database in the form of the list
        """
        proDataBase = []
        for i in seqId:
            k = self.Database[i]
            for j in range(len(k)):
                token = k[j]
                if token != -2:
                    if token == item:
                        isEndOfSeq = (k[j + 1] == -1 and k[j + 2] == -2)
                        if not isEndOfSeq:
                            proData = self._PseudoSequence(i, j + 1)
                            proDataBase.append(proData)
                            continue
        return proDataBase

    def _findAllFrequentPairsSingleItems(self, dataBase):
        """
        calculates the all possible pattern pairs for the items
        :param dataBase: projected database
        :return: return the dictionary containing item and their values
        """
        mapItemPseudoSeq = _ab._dd(list)
        for i in range(len(dataBase)):
            seqId = dataBase[i][0]
            seq = self.Database[seqId]
            for j in range(dataBase[i][1], len(seq)):
                token = seq[j]
                if token >= 0:
                    listSeq = mapItemPseudoSeq[token]
                    ok = True
                    if len(listSeq) > 0:
                        ok = listSeq[len(listSeq) - 1][0] != seqId
                    if ok:
                        listSeq.append(self._PseudoSequence(seqId, j + 1))
        return mapItemPseudoSeq

    def _findAllFrequentPairs(self, dataBase, lastBufPos):
        """
        calculates the all possible pattern pairs for the items
        :param dataBase: projected database
        :param lastBufPos: pattern buffer position to save the pattern
        :return: return the dictionary containing item and their values
        """
        pseudoSeqMP = {}
        pseudoSeqMPIP = {}
        mapsPairs = _ab._dd(list)
        mapPairsInPostfix = _ab._dd(list)
        firstPosOfLasItemBuf = lastBufPos
        while lastBufPos > 0:
            firstPosOfLasItemBuf -= 1
            if firstPosOfLasItemBuf < 0 or self._patternBuffer[firstPosOfLasItemBuf] == -1:
                firstPosOfLasItemBuf += 1
                break
        posToMatch = firstPosOfLasItemBuf
        for i in range(len(dataBase)):
            seqId = dataBase[i][0]
            seq = self.Database[seqId]
            prevItem = seq[dataBase[i][1] - 1]
            curItemSetIsPostfix = (prevItem != -1)
            isFirstItemSet = True
            for j in range(dataBase[i][1], len(seq)):
                token = seq[j]
                if token >= 0:
                    pair = token
                    if curItemSetIsPostfix:
                        if pair not in list(pseudoSeqMPIP):
                            pseudoSeqMPIP[pair] = []
                    else:
                        if pair not in list(pseudoSeqMP):
                            pseudoSeqMP[pair] = []
                    if curItemSetIsPostfix:
                        oldPair = mapPairsInPostfix[pair]
                    else:
                        oldPair = mapsPairs[pair]
                    if not oldPair:
                        if curItemSetIsPostfix:
                            mapPairsInPostfix[pair] = pair
                        else:
                            mapsPairs[pair] = pair
                    else:
                        pair = oldPair
                    ok = True
                    if curItemSetIsPostfix:
                        if len(pseudoSeqMPIP[pair]) > 0:
                            ok = pseudoSeqMPIP[pair][len(pseudoSeqMPIP[pair]) - 1][0] != seqId
                        if ok:
                            pseudoSeqMPIP[pair].append(self._PseudoSequence(seqId, j + 1))
                    else:
                        if len(pseudoSeqMP[pair]) > 0:
                            ok = pseudoSeqMP[pair][len(pseudoSeqMP[pair]) - 1][0] != seqId
                        if ok:
                            pseudoSeqMP[pair].append(self._PseudoSequence(seqId, j + 1))
                    if curItemSetIsPostfix and not isFirstItemSet:
                        pair = token
                        oldPair = mapsPairs[pair]
                        if not oldPair:
                            mapsPairs[pair] = pair
                        else:
                            pair = oldPair
                        ok = True
                        if pair not in list(pseudoSeqMP):
                            pseudoSeqMP[pair] = []
                        if len(pseudoSeqMP[pair]) > 0:
                            ok = pseudoSeqMP[pair][len(pseudoSeqMP[pair]) - 1][0] != seqId
                        if ok:
                            pseudoSeqMP[pair].append(self._PseudoSequence(seqId, j + 1))
                    if curItemSetIsPostfix is False and self._patternBuffer[posToMatch] is token:
                        posToMatch += 1
                        if posToMatch > lastBufPos:
                            curItemSetIsPostfix = True
                elif token == -1:
                    isFirstItemSet = False
                    curItemSetIsPostfix = False
                    posToMatch = firstPosOfLasItemBuf
        return [mapsPairs, mapPairsInPostfix, pseudoSeqMP, pseudoSeqMPIP]

    def _recursionSingleItems(self, patternBuff, dataBase, lastBuffPo):
        """
        recursion of the items if the pattern is frequent and finds the other pattern from that sequence
        :param patternBuff: pattern buffer we have taken
        :param dataBase: projected database
        :param lastBuffPo: buffer position in the pattern buffer
        """
        itemPseudoSeq = self._findAllFrequentPairsSingleItems(dataBase)
        if itemPseudoSeq:
            for i in list(sorted(itemPseudoSeq)):
                if len(itemPseudoSeq[i]) >= self.minSup:
                    patternBuff[lastBuffPo + 1] = -1
                    patternBuff[lastBuffPo + 2] = i
                    self._save1(lastBuffPo + 2, itemPseudoSeq[i])
                    self._recursionSingleItems(patternBuff, itemPseudoSeq[i], lastBuffPo + 2)

    def _recursion(self, patternBuff, dataBase, lastBuffPo):
        """
        recursion of the items if the pattern is frequent and finds the other pattern from that sequence
        :param patternBuff: pattern buffer we have taken
        :param dataBase: projected database
        :param lastBuffPo: buffer position in the pattern buffer
        """
        mapsPairs = self._findAllFrequentPairs(dataBase, lastBuffPo)
        if mapsPairs:
            for i in list(sorted(mapsPairs[1])):
                if len(mapsPairs[3][i]) >= self._minSup:
                    newBufferPos = lastBuffPo
                    newBufferPos += 1
                    patternBuff[newBufferPos] = i
                    self._save1(newBufferPos, mapsPairs[3][i])
                    self._recursion(patternBuff, mapsPairs[3][i], newBufferPos)
            for i in list(sorted(mapsPairs[0])):
                if len(mapsPairs[2][i]) >= self._minSup:
                    newBufferPos = lastBuffPo
                    newBufferPos += 1
                    patternBuff[newBufferPos] = -1
                    newBufferPos += 1
                    patternBuff[newBufferPos] = i
                    self._save1(newBufferPos, mapsPairs[2][i])
                    self._recursion(patternBuff, mapsPairs[2][i], newBufferPos)

    def startMine(self):
        """
        mining of the patterns starts from here
        """
        self._startTime = _ab._time.time()
        dat = self._scanDataBase()
        seq = self._findSequenceContainItems(dat)
        self._minSupAbs(self._iFile, self._minSup)
        if self._containsItemSetsWithMultipleItems:
            self._prefixSpanWithMultipleItems(seq, dat)
        else:
            self._prefixSpanWithSingleItems(seq, dat)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def getMemoryUSS(self):
        """
        Total USS memory consumed by the program will be retrieved by this function
        :return:
        """
        return self._memoryUSS

    def getMemoryRSS(self):
        """
        Total RSS memory consumed by the program will be retrieved by this function
        :return: return the memory
        """
        return self._memoryRSS

    def getRuntime(self):
        """
        Total amount of runtime taken by the program will be retrieved from this function
        :return: returns the time
        """
        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """
        complete set of patterns will be stored in the dataframe by this function
        :return: returns the dataframe
        """
        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['SequentialPatterns', 'Support'])
        return dataFrame

    def getPatterns(self):
        """
        Complete set of frequent patterns will be retrieved by this function
        :return: returns the frequent patterns dictionary
        """
        return self._finalPatterns

    def save(self, oFile):
        """
        save the frequent patterns in the output file
        :param oFile: output file which we saved the patterns
        """
        writer = open(oFile, 'w+')
        for x, y in self._finalPatterns.items():
            st = x + " # SUP " + y
            writer.write("%s \n" % st)


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = prefixSpan(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = prefixSpan(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_Patterns))
        _ap.save(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        '''l = [600,700,800, 900, 1000]
        for i in l:
            ap = prefixSpan('/Users/Likhitha/Downloads/PrefixSpan/small.txt', 1, ' ')
            ap.startMine()
            Patterns = ap.getPatterns()
            print("Total number of Frequent Patterns:", len(Patterns))
            ap.save('/Users/Likhitha/Downloads/output')
            memUSS = ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = ap.getRuntime()
            print("Total ExecutionTime in ms:", run)'''
        print("Error! The number of input parameters do not match the total number of parameters provided")

