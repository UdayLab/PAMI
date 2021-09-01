import sys
import functools
import pandas as pd
from PAMI.fuzzyPeriodicFrequentPattern.basic.abstract import *


class FFList:
    """
        A class represent a Fuzzy List of an element

    Attributes:
    ----------
        item: int
            the item name
        sumLUtil: float
            the sum of utilities of an fuzzy item in database
        sumRUtil: float
            the sum of resting values of a fuzzy item in database
        elements: list
            list of elements contain tid,Utility and resting values of element in each transaction
        maxPeriod: int
            it represent the max period of a item

    Methods:
    -------
        addElement(element)
            Method to add an element to this fuzzy list and update the sums at the same time.

        printElement(e)
            Method to print elements

    """

    def __init__(self, itemName):
        self.item = itemName
        self.sumLUtil = 0.0
        self.sumRUtil = 0.0
        self.elements = []
        self.maxPeriod = 0

    def addElement(self, element):
        """
            A Method that add a new element to FFList

            :param element: an element to be add to FFList
            :pram type: Element
        """
        self.sumLUtil += element.lUtils
        self.sumRUtil += element.rUtils
        self.elements.append(element)
        self.maxPeriod = max(self.maxPeriod, element.period)

    def printElement(self):
        """
            A Method to Print elements in the FFList
        """
        for ele in self.elements:
            print(ele.tid, ele.lUtils, ele.rUtils, ele.period)


class Element:
    """
        A class represents an Element of a fuzzy list

        Attributes:
        ----------
        tid : int
            keep tact of transaction id
        lUtils: float
            the utility of an fuzzy item in the transaction
        rUtils : float
            the resting value of an fuzzy item in the transaction
        period: int
            represent the period of the element
    """

    def __init__(self, tid, iUtil, rUtil, period):
        self.tid = tid
        self.lUtils = iUtil
        self.rUtils = rUtil
        self.period = period


class Reagions:
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
        if regionsNumber == 3:  # if we have 3 regions
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


class Pair:
    """
        A class to store item name and quantity together.
    """

    def __init__(self):
        self.item = 0
        self.quantity = 0


class fpfpMiner(fuzzyPeriodicFrequentPatterns):
    """
        Fuzzy Periodic Frequent Pattern Miner is desired to find all fuzzy periodic frequent patterns which is
        on-trivial and challenging problem to its huge search space.we are using efficient pruning
        techniques to reduce the search space.


    Attributes:
    ----------
        iFile : file
            Name of the input file to mine complete set of fuzzy spatial frequent patterns
        oFile : file
               Name of the oFile file to store complete set of fuzzy spatial frequent patterns
        minSup : float
            The user given support
        period: int
            periodicity of an element
        memoryRSS : float
                To store the total amount of RSS memory consumed by the program
        startTime:float
               To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        itemsCnt: int
            To record the number of fuzzy spatial itemsets generated
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
        maxTID: int
            represent the maximum tid of the database
        lastTIDs: map
            represent the last tid of fuzzy items
        itemsToRegion: map
            represent items with respective regions
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
        convert(value):
            To convert the given user specified value
        FSFIMining( prefix, prefixLen, fsFim, minSup)
            Method generate FFI from prefix
        construct(px, py)
            A function to construct Fuzzy itemset from 2 fuzzy itemsets
        findElementWithTID(ulist, tid)
            To find element with same tid as given
        WriteOut(prefix, prefixLen, item, sumIutil,period)
            To Store the patten
    
    Executing the code on terminal :
    -------
        Format:
        ------
        python3 fpfpMiner.py <inputFile> <outputFile> <minSup> <maxPer> <sep>

        Examples:
        ------
        python3  fpfpMiner.py sampleTDB.txt output.txt 2 3 (minSup and maxPer will be considered in support count or frequency)
        python3  fpfpMiner.py sampleTDB.txt output.txt 0.25 3 (minSup and maxPer will be considered in percentage of database)
                                        (will consider "\t" as separator)
        python3  fpfpMiner.py sampleTDB.txt output.txt 2 3  ,(will conseider ',' as separato)
        
    Sample run of importing the code:
    -------------------------------

        import fpfpMiner as alg

        obj =alg.fpfpMiner("input.txt",2,3)

        obj.startMine()

        periodicFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))

        obj.storePatternsInFile("output.txt")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
    Credits:
    -------
            The complete program was written by Sai Chitra.B under the supervision of Professor Rage Uday Kiran.
            The complete verification and documentation is done by Penugonda Ravikumar.

    """
    startTime = float()
    endTime = float()
    minSup = str()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    memoryUSS = float()
    memoryRSS = float()
    sep = ""

    def __init__(self, iFile, minSup, period, sep="\t"):
        super().__init__(iFile, minSup, period, sep)
        self.oFile = ""
        self.BufferSize = 200
        self.itemsetBuffer = []
        self.mapItemRegions = {}
        self.mapItemSum = {}
        self.mapItemsHighSum = {}
        self.finalPatterns = {}
        self.joinsCnt = 0
        self.itemsCnt = 0
        self.mapItemsMidSum = {}
        self.startTime = float()
        self.endTime = float()
        self.mapItemsLowSum = {}
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.dbLen = 0

    def compareItems(self, o1, o2):
        """
            A Function that sort all FFI-list in ascending order of Support
        """
        compare = self.mapItemSum[o1.item] - self.mapItemSum[o2.item]
        if compare == 0:
            return o1.item - o2.item
        else:
            return compare

    def convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.dbLen * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.dbLen * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        	Fuzzy periodic Frequent pattern mining process will start from here
        """
        maxTID = 0
        lastTIDs = {}
        self.startTime = time.time()
        with open(self.iFile, 'r') as file:
            for line in file:
                parts = line.split(":")
                parts[1]=parts[1].strip()
                parts[3]=parts[3].strip()
                tid = int(parts[0])
                self.dbLen += 1
                items = parts[1].split(self.sep)
                quantities = parts[3].split(self.sep)
                if tid < maxTID:
                    maxTID = tid
                for i in range(0, len(items)):
                    regions = Reagions(int(quantities[i]), 3)
                    item = items[i]
                    if item in self.mapItemsLowSum.keys():
                        low = self.mapItemsLowSum[item]
                        low += regions.low
                        self.mapItemsLowSum[item] = low
                    else:
                        self.mapItemsLowSum[item] = regions.low
                    if item in self.mapItemsMidSum.keys():
                        mid = self.mapItemsMidSum[item]
                        mid += regions.middle
                        self.mapItemsMidSum[item] = mid
                    else:
                        self.mapItemsMidSum[item] = regions.middle
                    if item in self.mapItemsHighSum.keys():
                        high = self.mapItemsHighSum[item]
                        high += regions.high
                        self.mapItemsHighSum[item] = high
                    else:
                        self.mapItemsHighSum[item] = regions.high
            listOfFFIList = []
            mapItemsToFFLIST = {}
            itemsToRegion = {}
            self.minSup = self.convert(self.minSup)
            self.maxPer = self.convert(self.maxPer)
            print(self.minSup, self.maxPer)
            for item1 in self.mapItemsLowSum.keys():
                item = item1
                low = self.mapItemsLowSum[item]
                mid = self.mapItemsMidSum[item]
                high = self.mapItemsHighSum[item]
                if low >= mid and low >= high:
                    self.mapItemSum[item] = low
                    self.mapItemRegions[item] = "L"
                    itemsToRegion[item] = "L"
                elif mid >= low and mid >= high:
                    self.mapItemSum[item] = mid
                    self.mapItemRegions[item] = "M"
                    itemsToRegion[item] = "M"
                elif high >= low and high >= mid:
                    self.mapItemRegions[item] = "H"
                    self.mapItemSum[item] = high
                    itemsToRegion[item] = "H"
                if self.mapItemSum[item] >= self.minSup:
                    fuList = FFList(item)
                    k = tuple([item, itemsToRegion.get(item)])
                    mapItemsToFFLIST[k] = fuList
                    listOfFFIList.append(fuList)
                    lastTIDs[item] = tid
            listOfFFIList.sort(key=functools.cmp_to_key(self.compareItems))
        with open(self.iFile, 'r') as file:
            for line in file:
                parts = line.split(":")
                tid = int(parts[0])
                parts[1]=parts[1].strip()
                parts[3]=parts[3].strip()
                items = parts[1].split(self.sep)
                quantities = parts[3].split(self.sep)
                revisedTransaction = []
                for i in range(0, len(items)):
                    pair = Pair()
                    pair.item = items[i]
                    regions = Reagions(int(quantities[i]), 3)
                    item = pair.item
                    if self.mapItemSum[item] >= self.minSup:
                        if self.mapItemRegions[pair.item] == "L":
                            pair.quantity = regions.low
                        elif self.mapItemRegions[pair.item] == "M":
                            pair.quantity = regions.middle
                        elif self.mapItemRegions[pair.item] == "H":
                            pair.quantity = regions.high
                        if pair.quantity > 0:
                            revisedTransaction.append(pair)
                revisedTransaction.sort(key=functools.cmp_to_key(self.compareItems))
                for i in range(len(revisedTransaction) - 1, -1, -1):
                    pair = revisedTransaction[i]
                    remainUtil = 0
                    for j in range(len(revisedTransaction) - 1, i - 1, -1):
                        remainUtil += revisedTransaction[j].quantity
                    if pair.quantity > remainUtil:
                        remainingUtility = pair.quantity
                    else:
                        remainingUtility = remainUtil
                    if mapItemsToFFLIST.get(tuple([pair.item, itemsToRegion[pair.item]])) is not None:
                        FFListOfItem = mapItemsToFFLIST[tuple([pair.item, itemsToRegion[pair.item]])]
                        if len(FFListOfItem.elements) == 0:
                            element = Element(tid, pair.quantity, remainingUtility, 0)
                        else:
                            if lastTIDs[pair.item] == tid:
                                element = Element(tid, pair.quantity, remainingUtility, maxTID - tid)
                            else:
                                lastTid = FFListOfItem.elements[-1].tid
                                curPer = tid - lastTid
                                element = Element(tid, pair.quantity, remainingUtility, curPer)
                        FFListOfItem.addElement(element)
        self.FSFIMining(self.itemsetBuffer, 0, listOfFFIList, self.minSup)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

    def FSFIMining(self, prefix, prefixLen, fsFim, minSup):

        """Generates FPFP from prefix

        :param prefix: the prefix patterns of FPFP
        :type prefix: len
        :param prefixLen: the length of prefix
        :type prefixLen: int
        :param fsFim: the Fuzzy list of prefix itemsets
        :type fsFim: list
        :param minSup: the minimum support of 
        :type minSup:int
        """
        for i in range(0, len(fsFim)):
            X = fsFim[i]
            if X.sumLUtil >= minSup and X.maxPeriod <= self.maxPer:
                self.WriteOut(prefix, prefixLen, X.item, X.sumLUtil, X.maxPeriod)
            if X.sumRUtil >= minSup:
                exULs = []
                for j in range(i + 1, len(fsFim)):
                    Y = fsFim[j]
                    exULs.append(self.construct(X, Y))
                    self.joinsCnt += 1
                self.itemsetBuffer.insert(prefixLen, X.item)
                self.FSFIMining(self.itemsetBuffer, prefixLen + 1, exULs, minSup, )

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self.memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
       """
        return self.memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process
        :rtype: float
       """
        return self.endTime - self.startTime

    def construct(self, px, py):
        """
            A function to construct a new Fuzzy item set from 2 fuzzy itemsets

            :param px:the item set px
            :type px:FFI-List
            :param py:item set py
            :type py:FFI-List
            :return :the item set of pxy(px and py)
            :rtype :FFI-List
        """
        pxyUL = FFList(py.item)
        prev = 0
        for ex in px.elements:
            ey = self.findElementWithTID(py, ex.tid)
            if ey is None:
                continue
            eXY = Element(ex.tid, min([ex.lUtils, ey.lUtils], key=lambda x: float(x)), ey.rUtils, ex.tid - prev)
            pxyUL.addElement(eXY)
            prev = ex.tid
        return pxyUL

    def findElementWithTID(self, uList, tid):
        """
            To find element with same tid as given
            :param uList: fuzzy list
            :type uList:FFI-List
            :param tid:transaction id
            :type tid:int
            :return:element eith tid as given
            :rtype: element if exizt or None
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

    def WriteOut(self, prefix, prefixLen, item, sumLUtil, period):
        """
            To Store the patten
            :param prefix: prefix of itemSet
            :type prefix: list
            :param prefixLen: length of prefix
            :type prefixLen: int
            :param item: the last item
            :type item: int
            :param sumLUtil: sum of utility of itemSet
            :type sumLUtil: float
            :param period: represent the period of itemSet
            :type period: int
        """
        self.itemsCnt += 1
        res = ""
        for i in range(0, prefixLen):
            res += str(prefix[i]) + "." + str(self.mapItemRegions[prefix[i]]) + " "
        res += str(item) + "." + str(self.mapItemRegions.get(item))
        res1 = str(sumLUtil) + " : " + str(period) + "\n"
        self.finalPatterns[res] = res1

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns

    def storePatternsInFile(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s \n" % patternsAndSupport)


if __name__ == "__main__":
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:  # to  include a user specifed separator
            ap = fpfpMiner(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:  # to consider "\t" as a separator
            ap = fpfpMiner(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        periodicFrequentPatterns = ap.getPatterns()
        print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
