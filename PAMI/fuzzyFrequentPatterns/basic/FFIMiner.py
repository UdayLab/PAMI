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
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.


from PAMI.fuzzyFrequentPatterns.basic import abstract as _ab


class _FFList:
    """
     A class represent a Fuzzy List of an element

    Attributes :
    ----------
         item: int
             the item name
         sumIUtil: float
             the sum of utilities of an fuzzy item in database
         sumRUtil: float
             the sum of resting values of a fuzzy item in database
         elements: list
             a list of elements contain tid,Utility and resting values of element in each transaction
    Methods :
    -------
        addElement(element)
            Method to add an element to this fuzzy list and update the sums at the same time.

        printElement(e)
            Method to print elements

    """

    def __init__(self, itemName):
        self.item = itemName
        self.sumIUtil = 0.0
        self.sumRUtil = 0.0
        self.elements = []

    def addElement(self, element):
        """
            A Method that add a new element to FFList

            :param element: an element to be add to FFList
            :pram type: Element
        """
        self.sumIUtil += element.iUtils
        self.sumRUtil += element.rUtils
        self.elements.append(element)

    def printElement(self):
        """
            A method to print elements
        """
        for ele in self.elements:
            print(ele.tid, ele.iUtils, ele.rUtils)


class _Element:
    """
        A class represents an Element of a fuzzy list

    Attributes:
    ----------
            tid : int
                keep tact of transaction id
            iUtils: float
                the utility of an fuzzy item in the transaction
            rUtils : float
                the  resting value of an fuzzy item in the transaction
    """

    def __init__(self, tid, iUtil, rUtil):
        self.tid = tid
        self.iUtils = iUtil
        self.rUtils = rUtil


class _Regions:
    """
        A class calculate the regions

    Attributes:
    ----------
            low : int
                low region value
            middle: int
                middle region value
            high : int
                high region values
        """

    def __init__(self, quantity, regionsNumber):
        self.low = 0
        self.middle = 0
        self.high = 0
        if regionsNumber == 3:
            if 0 < quantity <= 1:
                self.low = 1
                self.high = 0
                self.middle = 0
            elif 1 < quantity <= 6:
                self.low = float((6 - quantity) / 5)
                self.middle = float((quantity - 1) / 5)
                self.high = 0
            elif 6 < quantity <= 11:
                self.low = 0
                self.middle = float((11 - quantity) / 5)
                self.high = float((quantity - 6) / 5)
            else:
                self.low = 0
                self.middle = 0
                self.high = 1


class _Pair:
    """
        A class to store item and it's quantity together
    """

    def __init__(self):
        self.item = 0
        self.quantity = 0


class FFIMiner(_ab._fuzzyFrequentPattenrs):
    """
        Fuzzy Frequent  Pattern-Miner is desired to find all  frequent fuzzy patterns which is on-trivial and challenging problem
        to its huge search space.we are using efficient pruning techniques to reduce the search space.
    Reference :
    ---------
        https://www.researchgate.net/publication/286510908_A_fast_Algorithm_for_mining_fuzzy_frequent_itemSets

    Attributes :
    ----------
        iFile : string
            Name of the input file to mine complete set of fuzzy spatial frequent patterns
        oFile : string
               Name of the oFile file to store complete set of fuzzy spatial frequent patterns
        minSup : float
            The user given minimum support
        memoryRSS : float
                To store the total amount of RSS memory consumed by the program
        startTime:float
               To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        itemsCnt: int
            To record the number of fuzzy spatial itemSets generated
        mapItemsLowSum: map
            To keep track of low region values of items
        mapItemsMidSum: map
            To keep track of middle region values of items
        mapItemsHighSum: map
            To keep track of high region values of items
        mapItemSum: map
            To keep track of sum of Fuzzy Values of items
        mapItemRegions: map
            To Keep track of fuzzy regions of item
        jointCnt: int
            To keep track of the number of ffi-list that was constructed
        BufferSize: int
            represent the size of Buffer
        itemBuffer list
            to keep track of items in buffer
    Methods :
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        convert(value):
            To convert the given user specified value
        compareItems(o1, o2)
            A Function that sort all ffi-list in ascending order of Support
        FSFIMining(prefix, prefixLen, FSFIM, minSup)
            Method generate ffi from prefix
        construct(px, py)
            A function to construct Fuzzy itemSet from 2 fuzzy itemSets
        findElementWithTID(uList, tid)
            To find element with same tid as given
        WriteOut(prefix, prefixLen, item, sumIUtil)
            To Store the patten

    Executing the code on terminal :
    -------
        Format:
            python3 FFIMinerMiner.py <inputFile> <outputFile> <minSup> <separator>
        Examples:
            python3  FFIMinerMiner.py sampleTDB.txt output.txt 6  (minSup will be considered in support count or frequency)

            python3  FFIMinerMiner.py sampleTDB.txt output.txt 0.3 (minSup and maxPer will be considered in percentage of database)
                                                      (it will consider '\t' as a separator)

            python3  FFIMinerMiner.py sampleTDB.txt output.txt 6 , (it consider ',' as a separator)

    Sample run of importing the code:
    -------------------------------

        from PAMI.fuzzyFrequentPatterns import FFIMiner as alg

        obj = alg.FFIMiner("input.txt", 2)

        obj.startMine()

        fuzzyFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Frequent Patterns:", len(fuzzyFrequentPatterns))

        obj.savePatterns("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


    Credits:
    -------
        The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.
    """

    _startTime = float()
    _endTime = float()
    _minSup = str()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _sep = "\t"

    def __init__(self, iFile, minSup, sep="\t"):
        super().__init__(iFile, minSup, sep)
        self._startTime = 0
        self._endTime = 0
        self._itemsCnt = 0
        self._mapItemsLowSum = {}
        self._mapItemsMidSum = {}
        self._mapItemsHighSum = {}
        self._mapItemSum = {}
        self._mapItemRegions = {}
        self._regionReferenceMap = {}
        self._joinsCnt = 0
        self._BufferSize = 200
        self._itemSetBuffer = []
        self._transactionsDB = []
        self._fuzzyValueDB = []
        self._finalPatterns = {}
        self._dbLen = 0

    def _compareItems(self, o1, o2):
        """
            A Function that sort all ffi-list in ascending order of Support
        """
        compare = self._mapItemSum[o1.item] - self._mapItemSum[o2.item]
        if compare == 0:
            if o1.item < o2.item:
                return -1
            elif o1.item > o2.item:
                return 1
            else:
                return 0
        else:
            return compare

    def _convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._dbLen * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._dbLen * value)
            else:
                value = int(value)
        return value

    def _creatingItemsets(self):
        self._transactionsDB, self._fuzzyValueDB, self._Database = [], [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._transactionsDB = self._iFile['Transactions'].tolist()
            if 'fuzzyValues' in i:
                self._fuzzyValueDB = self._iFile['fuzzyValues'].tolist()
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = line.split(":")
                    parts[0] = parts[0].strip()
                    parts[2] = parts[2].strip()
                    items = parts[0].split(self._sep)
                    quantities = parts[2].split(self._sep)
                    self._transactionsDB.append([x for x in items])
                    self._fuzzyValueDB.append([x for x in quantities])
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.split("\n")[0]
                            parts = line.split(":")
                            parts[0] = parts[0].strip()
                            parts[2] = parts[2].strip()
                            items = parts[0].split(self._sep)
                            quantities = parts[2].split(self._sep)
                            self._transactionsDB.append([x for x in items])
                            self._fuzzyValueDB.append([x for x in quantities])
                except IOError:
                    print("File Not Found")
                    quit()

    def startMine(self):
        """
         fuzzy-Frequent pattern mining process will start from here
        """
        low = 0
        mid = 1
        high = 2

        self._startTime = _ab._time.time()
        self._creatingItemsets()

        for j in range(len(self._transactionsDB)):
            item_list = self._transactionsDB[j]
            value_list = self._fuzzyValueDB[j]
            self._dbLen += 1

            for i in range(0, len(item_list)):
                item = item_list[i]
                fuzzy_val = value_list[i]
                if fuzzy_val not in self._regionReferenceMap:
                    regions = _Regions(float(fuzzy_val), 3)
                    self._regionReferenceMap[fuzzy_val] = [regions.low, regions.middle, regions.high]
                else:
                    if item in self._mapItemsLowSum.keys():
                        item_total_low = self._mapItemsLowSum[item]
                        item_total_low += self._regionReferenceMap[fuzzy_val][low]
                        self._mapItemsLowSum[item] = item_total_low
                    else:
                        self._mapItemsLowSum[item] = self._regionReferenceMap[fuzzy_val][low]
                    if item in self._mapItemsMidSum.keys():
                        item_total_mid = self._mapItemsMidSum[item]
                        item_total_mid += self._regionReferenceMap[fuzzy_val][mid]
                        self._mapItemsMidSum[item] = item_total_mid
                    else:
                        self._mapItemsMidSum[item] = self._regionReferenceMap[fuzzy_val][mid]
                    if item in self._mapItemsHighSum.keys():
                        item_total_high = self._mapItemsHighSum[item]
                        item_total_high += self._regionReferenceMap[fuzzy_val][high]
                        self._mapItemsHighSum[item] = item_total_high
                    else:
                        self._mapItemsHighSum[item] = self._regionReferenceMap[fuzzy_val][high]
        listOfffilist = []
        mapItemsToFFLIST = {}
        self._minSup = self._convert(self._minSup)
        keyList = self._mapItemsLowSum.keys()
        for item1 in keyList:
            item = item1
            item_total_low = self._mapItemsLowSum[item]
            item_total_mid = self._mapItemsMidSum[item]
            item_total_high = self._mapItemsHighSum[item]
            if item_total_low >= item_total_mid and item_total_low >= item_total_high:
                self._mapItemSum[item] = item_total_low
                self._mapItemRegions[item] = "L"
            elif item_total_mid >= item_total_low and item_total_mid >= item_total_high:
                self._mapItemSum[item] = item_total_mid
                self._mapItemRegions[item] = "M"
            elif item_total_high >= item_total_low and item_total_high >= item_total_mid:
                self._mapItemRegions[item] = "H"
                self._mapItemSum[item] = item_total_high
            if self._mapItemSum[item] >= self._minSup:
                fuList = _FFList(item)
                mapItemsToFFLIST[item] = fuList
                listOfffilist.append(fuList)
            else:
                del self._mapItemSum[item]

        listOfffilist.sort(key=_ab._functools.cmp_to_key(self._compareItems))
        tid = 0
        for j in range(len(self._transactionsDB)):

            item_list = list(set(self._transactionsDB[j]).intersection(set(self._mapItemSum.keys())))
            revisedTransaction = []
            for i in range(0, len(item_list)):
                pair = _Pair()
                pair.item = item_list[i]
                fuzzy_val = self._fuzzyValueDB[j][self._transactionsDB[j].index(pair.item)]

                if self._mapItemRegions[pair.item] == "L":
                    pair.quantity = self._regionReferenceMap[str(fuzzy_val)][low]
                elif self._mapItemRegions[pair.item] == "M":
                    pair.quantity = self._regionReferenceMap[str(fuzzy_val)][mid]
                elif self._mapItemRegions[pair.item] == "H":
                    pair.quantity = self._regionReferenceMap[str(fuzzy_val)][high]
                if pair.quantity > 0:
                    revisedTransaction.append(pair)
            revisedTransaction.sort(key=_ab._functools.cmp_to_key(self._compareItems))
            for i in range(len(revisedTransaction) - 1, -1, -1):
                pair = revisedTransaction[i]
                remainUtil = 0
                for j in range(len(revisedTransaction) - 1, i, -1):
                    remainUtil += revisedTransaction[j].quantity
                    """ print("len:", len(revisedTransaction), "item j", revisedTransaction[j].item, " value: ", revisedTransaction[j].quantity, "total: ", remainUtil)
                    if tid == 3:
                        exit(2)"""
                remainingUtility = remainUtil
                if mapItemsToFFLIST.get(pair.item) is not None:
                    FFListOfItem = mapItemsToFFLIST[pair.item]
                    element = _Element(tid, pair.quantity, remainingUtility)
                    FFListOfItem.addElement(element)
            tid += 1


        self._FSFIMining(self._itemSetBuffer, 0, listOfffilist, self._minSup)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def _FSFIMining(self, prefix, prefixLen, FSFIM, minSup):
        """
         Generates ffi from prefix
        :param prefix: the prefix patterns of ffi
        :type prefix: len
        :param prefixLen: the length of prefix
        :type prefixLen: int
        :param FSFIM: the Fuzzy list of prefix itemSets
        :type FSFIM: list this list contains objects of class _FFlist
        :param minSup: the minimum support of
        :type minSup:int
        """
        for i in range(0, len(FSFIM)):
            _FFListObject1 = FSFIM[i]
            if _FFListObject1.sumIUtil >= minSup:
                self._WriteOut(prefix, prefixLen, _FFListObject1.item, _FFListObject1.sumIUtil)
            if _FFListObject1.sumRUtil >= minSup:
                exULs = []
                for j in range(i + 1, len(FSFIM)):
                    _FFListObject2 = FSFIM[j]
                    exULs.append(self._construct(_FFListObject1, _FFListObject2))
                    self._joinsCnt += 1
                self._itemSetBuffer.insert(prefixLen, _FFListObject1.item)
                self._FSFIMining(self._itemSetBuffer, prefixLen + 1, exULs, minSup)

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

    def _construct(self, _FFListObject1, _FFListObject2):
        """
            A function to construct a new Fuzzy itemSet from 2 fuzzy itemSets

            :param _FFListObject1:the itemSet px
            :type _FFListObject1:ffi-List
            :param _FFListObject2:itemSet py
            :type _FFListObject2:ffi-List
            :return :the itemSet of pxy(px and py)
            :rtype :ffi-List
        """
        _newObject = _FFList(_FFListObject2.item)
        for Ob1Element in _FFListObject1.elements:
            Ob2Element = self._findElementWithTID(_FFListObject2, Ob1Element.tid)
            if Ob2Element is None:
                continue
            newElement = _Element(Ob1Element.tid, min([Ob1Element.iUtils, Ob2Element.iUtils], key=lambda x: float(x)),
                                  Ob2Element.rUtils)
            _newObject.addElement(newElement)
        return _newObject

    def _findElementWithTID(self, uList, tid):
        """
            To find element with same tid as given
            :param uList: fuzzyList
            :type uList: ffi-List
            :param tid: transaction id
            :type tid: int
            :return: element  tid as given
            :rtype: element if exit or None
        """
        List = uList.elements
        first = 0
        last = len(List) - 1
        while first <= last:
            mid = (first + last) >> 1
            if List[mid].tid < tid:
                first = mid + 1
            elif List[mid].tid > tid:
                last = mid - 1
            else:
                return List[mid]
        return None

    def _WriteOut(self, prefix, prefixLen, item, sumIUtil):
        """
            To Store the patten

            :param prefix: prefix of itemSet
            :type prefix: list
            :param prefixLen: length of prefix
            :type prefixLen: int
            :param item: the last item
            :type item: int
            :param sumIUtil: sum of utility of itemSet
            :type sumIUtil: float

        """
        self._itemsCnt += 1
        res = ""
        for i in range(0, prefixLen):
            res += str(prefix[i]) + "." + str(self._mapItemRegions[prefix[i]]) + " "
        res += str(item) + "." + str(self._mapItemRegions.get(item))
        res1 = str(sumIUtil) + "\n"
        self._finalPatterns[res] = res1

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

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s" % patternsAndSupport)


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = FFIMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = FFIMiner(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        _fuzzyFrequentPatterns = _ap.getPatterns()
        print("Total number of Fuzzy-Frequent Patterns:", len(_fuzzyFrequentPatterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in seconds:", _run)
    elif len(_ab._sys.argv) == 1:
        # l = [1800, 1900, 2000, 2200, 2400]
        l = [5000]
        for i in range(1):
            _ap = FFIMiner('mushroom_utility_spmf.txt', 3900, ' ')
            _ap.startMine()
            fuzzycorrelatedFrequentPatterns = _ap.getPatterns()
            print("Total number of Fuzzy-Frequent Patterns:", len(fuzzycorrelatedFrequentPatterns))
            _ap.savePatterns('pattern.txt')
            memUSS = _ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = _ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = _ap.getRuntime()
            print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
