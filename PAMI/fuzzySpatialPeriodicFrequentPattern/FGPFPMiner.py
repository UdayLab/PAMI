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
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.
import pandas as pd
import plotly.express as px
import abstract as _ab


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
            A Method to Print elements in the FFList
        """
        for ele in self.elements:
            print(ele.tid, ele.iUtils, ele.rUtils)


class _Element:
    """
        A class represents an Element of a fuzzy list

    Attributes :
    ----------
        tid : int
            keep tact of transaction id
        iUtils: float
            the utility of an fuzzy item in the transaction
        rUtils : float
            the neighbourhood resting value of an fuzzy item in the transaction
    """

    def __init__(self, tid, iUtil, rUtil):
        self.tid = tid
        self.iUtils = iUtil
        self.rUtils = rUtil


class _Regions:
    """
            A class calculate the regions

    Attributes :
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


class FGPFPMiner(_ab._fuzzySpatialFrequentPatterns):
    """
        Fuzzy Frequent Spatial Pattern-Miner is desired to find all Spatially frequent fuzzy patterns
        which is on-trivial and challenging problem to its huge search space.we are using efficient pruning
         techniques to reduce the search space.

    Attributes :
    ----------
        iFile : file
            Name of the input file to mine complete set of fuzzy spatial frequent patterns
        oFile : file
               Name of the oFile file to store complete set of fuzzy spatial frequent patterns
        minSup : float
            The user given minimum support
        neighbors: map
            keep track of neighbours of elements
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
            To keep track of the number of FFI-list that was constructed
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
        FSFIMining( prefix, prefixLen, fsFim, minSup)
            Method generate FFI from prefix
        construct(px, py)
            A function to construct Fuzzy itemSet from 2 fuzzy itemSets
        Intersection(neighbourX,neighbourY)
            Return common neighbours of 2 itemSet Neighbours
        findElementWithTID(uList, tid)
            To find element with same tid as given
        WriteOut(prefix, prefixLen, item, sumIUtil,period)
            To Store the patten

    Executing the code on terminal :
    -------
        Format:
            python3 FFSPMiner.py <inputFile> <outputFile> <neighbours> <minSup> <sep>
        Examples:
            python3  FFSPMiner.py sampleTDB.txt output.txt sampleN.txt 3  (minSup will be considered in support count or frequency)

            python3  FFSPMiner.py sampleTDB.txt output.txt sampleN.txt 0.3 (minSup and maxPer will be considered in percentage of database)
                                                            (will consider "\t" as separator in both input and neighbourhood files)

            python3  FFSPMiner.py sampleTDB.txt output.txt sampleN.txt 3 ,
                                                              (will consider "," as separator in both input and neighbourhood files)
    Sample run of importing the code:
    -------------------------------

        from PAMI.fuzzyFrequentSpatialPattern import FFSPMiner as alg

        obj = alg.FFSPMiner("input.txt", "neighbours.txt", 2)

        obj.startMine()

        fuzzySpatialFrequentPatterns = obj.getPatterns()

        print("Total number of fuzzy frequent spatial patterns:", len(fuzzySpatialFrequentPatterns))

        obj.savePatterns("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
            The complete program was written by B.Sai Chitra and modified by Kundai Kwangwari
            under the supervision of Professor Rage Uday Kiran.
    """

    _startTime = float()
    _endTime = float()
    _minSup = str()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _nFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _sep = "\t"
    _transactionsDB = []
    _fuzzyValuesDB = []

    def __init__(self, iFile, nFile, minSup, maxPer, sep):
        super().__init__(iFile, nFile, minSup, maxPer, sep)
        self._mapItemNeighbours = {}
        self._startTime = 0
        self._endTime = 0
        self._itemsCnt = 0
        self._mapItemsLowSum = {}
        self._mapItemsMidSum = {}
        self._mapItemsHighSum = {}
        self._mapItemSum = {}
        self._mapItemRegions = {}
        self._fuzzyRegionReferenceMap = {}
        self._joinsCnt = 0
        self._BufferSize = 200
        self._itemSetBuffer = []
        self._finalPatterns = {}
        self._finalPeriodicPatterns = {}
        self._dbLen = 0

    def _compareItems(self, o1, o2):
        """
            A Function that sort all FFI-list in ascending order of Support
        """
        compare = self._mapItemSum[o1.item] - self._mapItemSum[o2.item]
        if compare == 0:
            return int(o1.item) - int(o2.item)
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
            else:
                value = int(value)
        return value

    def _creatingItemSets(self):
        self._transactionsDB, self._fuzzyValuesDB = [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._transactionsDB = self._iFile['Transactions'].tolist()
            if 'fuzzyValues' in i:
                self._fuzzyValuesDB = self._iFile['Utilities'].tolist()

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = line.split(":")
                    items = parts[0].split(self._sep)
                    quantities = parts[2].split(self._sep)
                    self._transactionsDB.append([x for x in items])
                    self._fuzzyValuesDB.append([x for x in quantities])
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
                            self._fuzzyValuesDB.append([x for x in quantities])
                except IOError:
                    print("File Not Found")
                    quit()

    def _mapNeighbours(self):
        self._mapItemNeighbours = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data, items = [], []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'items' in i:
                items = self._nFile['items'].tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for k in range(len(items)):
                self._mapItemNeighbours[items[k]] = data[k]

        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = [i.rstrip() for i in line.split(self._sep)]
                    parts = [x for x in parts]
                    item = parts[0]
                    neigh1 = []
                    for i in range(1, len(parts)):
                        neigh1.append(parts[i])
                    self._mapItemNeighbours[item] = neigh1
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.split("\n")[0]
                            parts = [i.rstrip() for i in line.split("\t")]
                            parts = [x for x in parts]
                            item = parts[0]
                            neigh1 = []
                            for i in range(1, len(parts)):
                                neigh1.append(parts[i])
                            self._mapItemNeighbours[item] = neigh1
                except IOError:
                    print(self._nFile)
                    print("File Not Found")
                    quit()

    def startMine(self):
        """ Frequent pattern mining process will start from here
        """
        self._startTime = _ab._time.time()
        self._mapNeighbours()
        self._creatingItemSets()
        self._finalPatterns = {}
        low = 0
        mid = 1
        high = 2
        for line in range(len(self._transactionsDB)):
            item_list = self._transactionsDB[line]
            fuzzyValues_list = self._fuzzyValuesDB[line]
            self._dbLen += 1
            for i in range(0, len(item_list)):
                item = item_list[i]
                fuzzy_ref = fuzzyValues_list[i]
                if item in self._mapItemNeighbours:
                    if fuzzy_ref not in self._fuzzyRegionReferenceMap:
                        regions = _Regions(int(fuzzy_ref), 3)
                        self._fuzzyRegionReferenceMap[fuzzy_ref] = [regions.low, regions.middle, regions.high]
                    else:
                        if item in self._mapItemsLowSum.keys():
                            item_total_low = self._mapItemsLowSum[item]
                            item_total_low += self._fuzzyRegionReferenceMap[fuzzy_ref][low]
                            self._mapItemsLowSum[item] = item_total_low
                        else:
                            self._mapItemsLowSum[item] = self._fuzzyRegionReferenceMap[fuzzy_ref][low]
                        if item in self._mapItemsMidSum.keys():
                            item_total_mid = self._mapItemsMidSum[item]
                            item_total_mid += self._fuzzyRegionReferenceMap[fuzzy_ref][mid]
                            self._mapItemsMidSum[item] = item_total_mid
                        else:
                            self._mapItemsMidSum[item] = self._fuzzyRegionReferenceMap[fuzzy_ref][mid]
                        if item in self._mapItemsHighSum.keys():
                            item_total_high = self._mapItemsHighSum[item]
                            item_total_high += self._fuzzyRegionReferenceMap[fuzzy_ref][high]
                            self._mapItemsHighSum[item] = item_total_high
                        else:
                            self._mapItemsHighSum[item] = self._fuzzyRegionReferenceMap[fuzzy_ref][high]

        listOfFFList = []
        mapItemsToFFLIST = {}
        self._minSup = self._convert(self._minSup)
        for item in self._mapItemsLowSum.keys():
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
                listOfFFList.append(fuList)
            else: del self._mapItemSum[item]

        listOfFFList.sort(key=_ab._functools.cmp_to_key(self._compareItems))
        tid = 0
        for j in range(len(self._transactionsDB)):
            item_list = list(set(self._transactionsDB[j]).intersection(set(self._mapItemSum.keys())))
            revisedTransaction = []
            for i in range(0, len(item_list)):
                pair = _Pair()
                pair.item = item_list[i]
                fuzzy_ref = str(self._fuzzyValuesDB[j][self._transactionsDB[j].index(pair.item)])
                if self._mapItemRegions[pair.item] == "L":
                    pair.quantity = self._fuzzyRegionReferenceMap[fuzzy_ref][low]
                elif self._mapItemRegions[pair.item] == "M":
                    pair.quantity = self._fuzzyRegionReferenceMap[fuzzy_ref][mid]
                elif self._mapItemRegions[pair.item] == "H":
                    pair.quantity = self._fuzzyRegionReferenceMap[fuzzy_ref][high]
                if pair.quantity > 0:
                    revisedTransaction.append(pair)
            revisedTransaction.sort(key=_ab._functools.cmp_to_key(self._compareItems))
            qaunt = {}
            for i in range(len(revisedTransaction) -1, -1, -1):
                pair = revisedTransaction[i]
                qaunt[pair.item] = pair.quantity
                remainUtil = 0
                temp = list(set(self._mapItemNeighbours[pair.item]).intersection(set(qaunt.keys())))
                for j in temp:
                    remainUtil += float(qaunt[j])
                del temp
                remainingUtility = remainUtil
                FFListObject = mapItemsToFFLIST[pair.item]
                element = _Element(tid, pair.quantity, remainingUtility)
                FFListObject.addElement(element)
            del qaunt
            tid += 1
        itemNeighbours = list(self._mapItemNeighbours.keys())
        self._FSFIMining(self._itemSetBuffer, 0, listOfFFList, self._minSup, itemNeighbours)
        self._generatePeriodicPatterns()
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss


    def _FSFIMining(self, prefix, prefixLen, FSFIM, minSup, itemNeighbours):
        """Generates FFSPMiner from prefix

        :param prefix: the prefix patterns of FFSPMiner
        :type prefix: len
        :param prefixLen: the length of prefix
        :type prefixLen: int
           :param FSFIM: the Fuzzy list of prefix itemSets
           :type FSFIM: list
           :param minSup: the minimum support of
           :type minSup:int
           :param itemNeighbours: the set of common neighbours of prefix
           :type itemNeighbours: list or set
        """
        for i in range(0, len(FSFIM)):
            _FFListObject1 = FSFIM[i]
            if _FFListObject1.sumIUtil >= minSup:
                self._WriteOut(prefix, prefixLen, _FFListObject1.item, _FFListObject1.sumIUtil)
            newNeighbourList = self._Intersection(self._mapItemNeighbours.get(_FFListObject1.item), itemNeighbours)
            if _FFListObject1.sumRUtil >= minSup:
                exULs = []
                for j in range(i + 1, len(FSFIM)):
                    _FFListObject2 = FSFIM[j]
                    if _FFListObject2.item in newNeighbourList:
                        exULs.append(self._construct(_FFListObject1, _FFListObject2))
                        self._joinsCnt += 1
                self._itemSetBuffer.insert(prefixLen, _FFListObject1.item)
                self._FSFIMining(self._itemSetBuffer, prefixLen + 1, exULs, minSup, newNeighbourList)

    def _Intersection(self, neighbourX, neighbourY):
        """
            A function to get common neighbours from 2 itemSets
            :param neighbourX: the set of neighbours of itemSet 1
            :type neighbourX: set or list
            :param neighbourY: the set of neighbours of itemSet 2
            :type neighbourY: set or list
            :return : set of common neighbours of 2 itemSets
            :rtype :set
        """
        result = []
        if neighbourX is None or neighbourY is None:
            return result
        for i in range(0, len(neighbourX)):
            if neighbourX[i] in neighbourY:
                result.append(neighbourX[i])
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

    def _construct(self, _FFListObject1, _FFListObject2):
        """
            A function to construct a new Fuzzy itemSet from 2 fuzzy itemSets

            :param _FFListObject1:the itemSet px
            :type _FFListObject1:FFI-List
            :param _FFListObject2:itemSet py
            :type _FFListObject2:FFI-List
            :return :the itemSet of pxy(px and py)
            :rtype :FFI-List
        """
        _newFFListObject = _FFList(_FFListObject2.item)
        for Ob1Element in _FFListObject1.elements:
            Ob2Element = self._findElementWithTID(_FFListObject2, Ob1Element.tid)
            if Ob2Element is None:
                continue
            newElement = _Element(Ob1Element.tid, min([Ob1Element.iUtils, Ob2Element.iUtils], key=lambda x: float(x)), Ob2Element.rUtils)
            _newFFListObject.addElement(newElement)
        return _newFFListObject

    def _findElementWithTID(self, uList, tid):
        """
            To find element with same tid as given
            :param uList:fuzzyList
            :type uList:FFI-List
            :param tid:transaction id
            :type tid:int
            :return:element tid as given
            :rtype: element if exist or None
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
        res1 = str(sumIUtil)
        self._finalPatterns[res] = res1

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPeriodicPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPeriodicPatterns

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPeriodicPatternsAsDataframe(self):

        """
        :return: returning periodic frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        data = []
        dataFrame = _ab._pd.DataFrame()
        for a, b in self._finalPeriodicPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def _generatePeriodicPatterns(self):

        """finding periodicity in the patterns

        :return: returning periodic frequent patterns in a dict
        :rtype: dict
        """
        dataFrame = {}
        data = []
        pattern_list = list(self._finalPatterns.keys())
        list_len = len(pattern_list)
        recent_occur, first_occur, tid = 0, 0, 0
        self._maxPer = 0.95 * list_len

        for i in range(0, list_len):
            pattern1 = pattern_list[i].split(" ")
            periodlist = []
            maxperiod = 0
            for j in range(0, list_len):
                pattern2 = pattern_list[j].split(" ")
                tid = j + 1
                if set(pattern1).issubset(set(pattern2)):
                    if len(periodlist) == 0:
                        periodlist.append(abs(first_occur - tid))
                        recent_occur = tid
                    else:
                        periodlist.append(tid - recent_occur)
                        recent_occur = tid
            periodlist.append(list_len - tid)

            maxperiod = max(periodlist)
            if int(self._maxPer) >= maxperiod:
                self._finalPeriodicPatterns[pattern_list[i]] = self._finalPatterns[pattern_list[i]]

if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            print(_ab._sys.argv[1], _ab._sys.argv[2], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
            _ap = FGPFPMiner(_ab._sys.argv[1], _ab._sys.argv[2], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = FGPFPMiner(_ab._sys.argv[1], _ab._sys.argv[2], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_Patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        print(_ap.getPatternsAsDataFrame())
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print('Error! The number of input parameters do not match the total number of parameters provided')






