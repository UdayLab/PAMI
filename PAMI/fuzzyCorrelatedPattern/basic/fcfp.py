import sys
import functools
import pandas as pd
from PAMI.fuzzyCorrelatedPattern.basic.abstract import *

class FFList:
    """
     A class represent a Fuzzy List of an element

    Attributes :
    ----------
         item: int
             the item name
         sumIutil: float
             the sum of utilities of an fuzzy item in database
         sumRutil: float
             the sum of resting values of a fuzzy item in database
         elements: list
             a list of elements contain tid,Utility and resting values of element in each transaction
    Methods :
    -------
        addElement(element)
            Method to add an element to this fuzzy list and update the sums at the same time.

        printelement(e)
            Method to print elements            

    """

    def __init__(self, itemName, region):
        self.item = itemName
        self.region = region
        self.sumIutil = 0.0
        self.sumRutil = 0.0
        self.elements = []

    def addElement(self, element):
        """
            A Method that add a new element to FFList

            :param element: an element to be add to FFList
            :pram type: Element
        """
        self.sumIutil += element.iutils
        self.sumRutil += element.rutils
        self.elements.append(element)


class Element:
    """
        A class represents an Element of a fuzzy list

    Attributes :
    ----------
        tid : int
            keep tact of transaction id
        iutils: float
            the utility of an fuzzy item in the transaction
        rutil : float
            the neighbourhood resting value of an fuzzy item in the transaction
    """

    def __init__(self, tid, iutil, rutil):
        self.tid = tid
        self.iutils = iutil
        self.rutils = rutil


class Reagions:
    """
            A class calculate the region value

    Attributes:
    ----------
            low : int
                low region value
            middle: int 
                middle region value
            high : int
                high region values
        """

    def __init__(self, item, quantity, regionsNumber, mapOfregios):
        self.low = 0
        self.middle = 0
        self.high = 0
        if regionsNumber == 3:
            if 0 < quantity <= 1:
                self.low = 1
                self.high = 0
                self.middle = 0
                t1 = (item, 'L')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios[t1] = temp + 1
            elif 1 <= quantity < 6:
                self.low = float((-0.2 * quantity) + 1.2)
                self.middle = float((0.2 * quantity) - 0.2)
                self.high = 0
                t1 = (item, 'L')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios[t1] = temp + 1
                t1 = (item, 'M')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios[t1] = temp + 1
            elif 6 <= quantity <= 11:
                self.low = 0
                self.middle = float((-0.2 * quantity) + 2.2)
                self.high = float((0.2 * quantity) - 1.2)
                t1 = (item, 'M')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios[t1] = temp + 1
                t1 = (item, 'H')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios[t1] = temp + 1

            else:
                self.low = 0
                self.middle = 0
                self.high = 1
                t1 = (item, 'H')
                if t1 not in mapOfregios.keys():
                    mapOfregios[t1] = 1
                else:
                    temp = mapOfregios[t1]
                    mapOfregios = temp + 1


class Pair:
    def __init__(self):
        """
            A Class to Store item and its quantity together
        """
        self.item = 0
        self.quantity = 0
        self.region = 'N'


class fcfp(corelatedFuzzyFrequentPatterns):
    """
        fcfp is the algorithm to discover Corelated Fuzzy-frequent patterns in a transactional database.
        it is based on traditianl fuzzy frequent pattern mining.

    Attributes :
    ----------
        iFile : file
            Name of the input file to mine complete set of fuzzy spatial frequent patterns
        oFile : file
               Name of the oFile file to store complete set of fuzzy spatial frequent patterns
        minSup : int
                The user given support
        minAllConf: float
             user Specified minAllConf( should be in range 0 and 1)
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
            To keep track of sum of Fuzzy Vlues of items
        mapItemRegions: map
            To Kepp track of fuzzy regions of item
        jointCnt: int
            To keep track of the number of FFI-list that was constructed
        BufferSize: int
            represent the size of Buffer
        itemBuffer list
            to kepp track of items in buffer
    Methods :
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
        getRatio(self, prefix, prefixLen, item)
            Method to calculate the ration of itemset
        convert(value):
            To convert the given user specified value  
        FSFIMining( prefix, prefixLen, fsFim, minSup)
            Method generate FFI from prefix
        construct(px, py)
            A function to construct Fuzzy itemset from 2 fuzzy itemsets
        findElementWithTID(ulist, tid)
            To find element with same tid as given
        WriteOut(prefix, prefixLen, item, sumIutil,ratio)
            To Store the patten      

    Executing the code on terminal :
    -------
            Format: python3 fcfp.py <inputFile> <outputFile> <minSup> <minAllConf> <sep>
            Examples: python3 fcfp.py sampleTDB.txt output.txt 2 0.2 (minSup will be considered in support count or frequency)
                      python3 fcfp.py sampleTDB.txt output.txt 0.25 0.2 (minSup and maxPer will be considered in percentage of database)
                                                                     (it will consider separator as "\t")
                      python3 fcfp.py sampleTDB.txt output.txt 2 0.2 ,                    
                                                                      (it will consider separator as ',')
    Sample run of importing the code:
    -------------------------------
        
        import fcfp as alg

        obj = alg.fcfp("input.txt",2,0.4)

        obj.startMine()

        corelatedFuzzyFrequentPatterns = obj.getPatterns()

        print("Total number of Corelated Fuzzy Frequent Patterns:", len(corelatedFuzzyFrequentPatterns))

        obj.storePatternsInFile("outp")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
            The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.

    """
    startTime = float()
    endTime = float()
    minSup = str()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    minAllConf=0.0
    sep="\t"
    memoryUSS = float()
    memoryRSS = float()

    def __init__(self, iFile, minSup, ratio,sep="\t"):
        super().__init__(iFile, minSup,ratio,sep)
        self.temp = {}
        self.mapItemRegionSum = {}
        self.start = 0
        self.end = 0
        self.itemsCnt = 0
        self.mapItemsLowSum = {}
        self.mapItemsMidSum = {}
        self.mapItemsHighSum = {}
        self.mapItemSum = {}
        self.mapItemRegions = {}
        self.joinsCnt = 0
        self.BufferSize = 200
        self.itemsetBuffer = []
        self.finalPatterns = {}
        self.dbLen = 0

    def compareItems(self, o1, o2):
        """
            A Function that sort all FFI-list in asendng order of Support
        """
        compare = self.mapItemSum[o1.item] - self.mapItemSum[o2.item]
        if compare == 0:
            return o1.item - o2.item
        else:
            return compare

    def findElementWithTID(self, ulist, tid):
        """
        To find element with same tid as given
        :param ulist: fuzzylist
        :type ulist: FFI-List
        :param tid: transaction id
        :type tid: int
        :return: element eith tid as given
        :rtype: element if exists or None
        """
        List = ulist.elements
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
            Frequent pattern mining process will start from here
        """
        self.start = time.time()
        self.minSup = self.convert(self.minSup)
        minSup = self.minSup
        with open(self.iFile, 'r') as file:
            for line in file:
                self.dbLen += 1
                parts = line.split(":")
                items = parts[0].split(self.sep)
                quanaities = parts[2].split(self.sep)
                for i in range(0, len(items)):
                    item = items[i]
                    regions = Reagions(item, int(quanaities[i]), 3, self.mapItemRegionSum)
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
            listOfFFIlist = []
            mapItemsToFFLIST = {}
            for item1 in self.mapItemsLowSum.keys():
                item = item1
                region = 'N'
                low = self.mapItemsLowSum[item]
                mid = self.mapItemsMidSum[item]
                high = self.mapItemsHighSum[item]
                if low >= mid and low >= high:
                    self.mapItemSum[item] = low
                    self.mapItemRegions[item] = "L"
                    region = 'L'
                elif mid >= low and mid >= high:
                    self.mapItemSum[item] = mid
                    self.mapItemRegions[item] = "M"
                    region = 'M'
                elif high >= low and high >= mid:
                    self.mapItemRegions[item] = "H"
                    region = 'H'
                    self.mapItemSum[item] = high
                if self.mapItemSum[item] >= self.minSup:
                    fuList = FFList(item, region)
                    mapItemsToFFLIST[item] = fuList
                    listOfFFIlist.append(fuList)
            listOfFFIlist.sort(key=functools.cmp_to_key(self.compareItems))
        tid = 0
        with open(self.iFile, 'r') as file:
            for line in file:
                parts = line.split(":")
                items = parts[0].split(self.sep)
                quanaities = parts[2].split(self.sep)
                revisedTransaction = []
                for i in range(0, len(items)):
                    pair = Pair()
                    pair.item = items[i]
                    regions = Reagions(pair.item, int(quanaities[i]), 3, self.temp)
                    item = pair.item
                    if self.mapItemSum[item] >= minSup:
                        if self.mapItemRegions[pair.item] == "L":
                            pair.quantity = regions.low
                            pair.region = 'L'
                        elif self.mapItemRegions[pair.item] == "M":
                            pair.region = 'M'
                            pair.quantity = regions.middle
                        elif self.mapItemRegions[pair.item] == "H":
                            pair.quantity = regions.high
                            pair.region = 'H'
                        if pair.quantity > 0:
                            revisedTransaction.append(pair)
                revisedTransaction.sort(key=functools.cmp_to_key(self.compareItems))
                for i in range(len(revisedTransaction) - 1, -1, -1):
                    pair = revisedTransaction[i]
                    remainUtil = 0
                    for j in range(len(revisedTransaction) - 1, i - 1, -1):
                        remainUtil += revisedTransaction[j].quantity
                    if pair.quantity > remainUtil:
                        remaingUtility = pair.quantity
                    else:
                        remaingUtility = remainUtil
                    if mapItemsToFFLIST.get(pair.item) is not None:
                        FFListOfItem = mapItemsToFFLIST[pair.item]
                        element = Element(tid, pair.quantity, remaingUtility)
                        FFListOfItem.addElement(element)
                tid += 1
        self.FSFIMining(self.itemsetBuffer, 0, listOfFFIlist, self.minSup)
        self.endtime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("No.Of Itemssets ", self.itemsCnt)
        print("joinsCnt: ", self.joinsCnt)

    def FSFIMining(self, prefix, prefixLen, FSFIM, minsup):
        """
            Generates FFSI from prefix

        :param prefix: the prefix patterns of FFSI
        :type prefix: len
        :param prefixLen: the length of prefix
        :type prefixLen: int
           :param FSFIM: the Fuzzy list of prefix itemsets
           :type FSFIM: list
           :param minsup: the minimum support of 
           :type minsup:int
           :param itemNighbours : the set of common neighbours of prefix
           :type itemNighbours : set
        """
        for i in range(0, len(FSFIM)):
            X = FSFIM[i]
            if X.sumIutil >= minsup:
                ratio = self.getRatio(prefix, prefixLen, X)
                if ratio >= self.minAllConf:
                    self.WriteOut(prefix, prefixLen, X, ratio)
            if X.sumRutil >= minsup:
                exULs = []
                for j in range(i + 1, len(FSFIM)):
                    Y = FSFIM[j]
                    exULs.append(self.construct(X, Y))
                    self.joinsCnt += 1
                self.itemsetBuffer.insert(prefixLen, X)
                self.FSFIMining(self.itemsetBuffer, prefixLen + 1, exULs, minsup)

    def construct(self, px, py):
        """
            A function to construct a new Fuzzy itemset from 2 fuzzy itemsets

            :param px:the itemset px
            :type px:FFI-List
            :param py:ithemset py
            :type py:FFI-List
            :return :the itemset of pxy(px and py)
            :rtype :FFI-List
        """
        pxyUL = FFList(py.item, py.region)
        for ex in px.elements:
            ey = self.findElementWithTID(py, ex.tid)
            if ey is None:
                continue
            eXY = Element(ex.tid, min([ex.iutils, ey.iutils], key=lambda x: float(x)), ey.rutils)
            pxyUL.addElement(eXY)
        return pxyUL

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
        return self.endtime - self.start

    def getRatio(self, prefix, prefixLen, item):
        """Method to calculate the ration of itemset
            :param prefix: prefix of itemset
            :type prefix: list
            :param prefixLen: length of prefix
            :type prefixLen: int
            :param item: the last item
            :type item: int
            :return : corelated ratio
            :rtype: float
        """
        res = 1.0
        n = prefixLen
        for i in prefix:
            if self.mapItemRegionSum.get((i.item, i.region)) is not None and res < self.mapItemRegionSum[
                (i.item, i.region)]:
                res = self.mapItemRegionSum[(i.item, i.region)]
        if self.mapItemRegionSum.get((item.item, item.region)) is not None and res < self.mapItemRegionSum[
            (item.item, item.region)]:
            res = self.mapItemRegionSum[(item.item, item.region)]
        return item.sumIutil / res

    def WriteOut(self, prefix, prefixLen, item, ratio):
        """
            To Store the patten
            :param prefix: prefix of itemset
            :type prefix: list
            :param prefixLen: length of prefix
            :type prefixLen: int
            :param item: the last item
            :type item: int
            :param sumIutil: sum of utility of itemset
            :type sumIutil: float
            :param ratio: the ratio of itemset
            :type ratio: float
        """
        self.itemsCnt += 1
        res = ""
        for i in range(0, prefixLen):
            res += str(prefix[i].item) + "." + str(prefix[i].region) + ' '
        res += str(item.item) + "." + str(item.region)
        res1 = str(item.sumIutil) + " : " + str(ratio) + "\n"
        # self.bwriter.write(res)
        self.finalPatterns[res] = res1

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

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

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6: # to includes separator
            ap = fcfp(sys.argv[1], sys.argv[3],float(sys.argv[4]),sys.argv[5])
        if len(sys.argv) == 5: # to consider '\t' as separator
           ap = fcfp(sys.argv[1], sys.argv[3],float(sys.argv[4]))
        ap.startMine()
        fuzzycorelatedFrequentPattenrs = ap.getPatterns()
        print("Total number of Fuzzy-Frequent Patterns:", len(fuzzycorelatedFrequentPattenrs))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")