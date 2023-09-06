# F3PMiner algorithm discovers the fuzzy partial periodic patterns in quantitative Irregulat multiple timeseries databases.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#     import PAMI.fuzzyPartialPeriodicPattern.basic.F3PMiner as alg
#
#     obj = alg.F3PMiner(iFile, minSup, sep)
#
#     obj.startMine()
#
#     fuzzyPartialPeriodicPatterns = obj.getPatterns()
#
#     print("Total number of Fuzzy Partial Periodic Patterns:", len(fuzzyPartialPeriodicPatterns))
#
#     obj.save(oFile)
#
#     Df = obj.getPatternInDataFrame()
#
#     memUSS = obj.getMemoryUSS()
#
#     print("Total Memory in USS:", memUSS)
#
#     memRSS = obj.getMemoryRSS()
#
#     print("Total Memory in RSS", memRSS)
#
#     run = obj.getRuntime()
#
#     print("Total ExecutionTime in seconds:", run)



__copyright__ = """

 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from PAMI.fuzzyPartialPeriodicPatterns.basic import abstract as _ab


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
        self.elements = []

    def addElement(self, element):
        """
            A Method that add a new element to FFList

            :param element: an element to be added to FFList
            :param type: Element
        """
        self.sumIUtil += element.iUtils
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

    def __init__(self, tid, iUtil):
        self.tid = tid
        self.iUtils = iUtil


class _Pair:
    """
        A class to store item and it's quantity together
    """

    def __init__(self):
        self.item = 0
        self.quantity = 0


class F3PMiner(_ab._fuzzyPartialPeriodicPatterns):
    """
    Description:
    ----------
        F3PMiner algorithm discovers the fuzzy partial periodic patterns in quantitative Irregulat multiple timeseries databases.
    
    Reference :
    ---------
        
        
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
        mapItemsGSum: map
            To keep track of G region values of items
        mapItemsMidSum: map
            To keep track of M region values of items
        mapItemsHSum: map
            To keep track of H region values of items
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
        convert(value):
            To convert the given user specified value
        compareItems(o1, o2)
            A Function that sort all ffi-list in ascending order of Support
        F3PMining(prefix, prefixLen, FSFIM, minSup)
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
            python3 F3PMiner.py <inputFile> <outputFile> <minSup> <separator>
        Examples:
            python3  F3PMiner.py sampleTDB.txt output.txt 6  (minSup will be considered in support count or frequency)

            python3  F3PMiner.py sampleTDB.txt output.txt 0.3 (minSup and maxPer will be considered in percentage of database)
                                                      (it will consider '\t' as a separator)

            python3  F3PMiner.py sampleTDB.txt output.txt 6 , (it consider ',' as a separator)

    Sample run of importing the code:
    -------------------------------

        from PAMI.fuzzyPartialPeriodicPatterns import F3PMiner as alg

        obj = alg.F3PMiner("input.txt", 2)

        obj.startMine()

        fuzzyPartialPeriodicPatterns = obj.getPatterns()

        print("Total number of Fuzzy Frequent Patterns:", len(fuzzyPartialPeriodicPatterns))

        obj.save("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


    Credits:
    -------
        The complete program was written by PALLA Likhitha under the supervision of Professor Rage Uday Kiran.
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
        self._mapItemSum = {}
        self._mapItemRegions = {}
        self._joinsCnt = 0
        self._BufferSize = 200
        self._itemSetBuffer = []
        self._transactions = []
        self._fuzzyValues = []
        self._ts = []
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
        """
           Storing the complete transactions of the database/input file in a database variable

        """
        self._transactions, self._fuzzyValues, self._Database, self._ts = [], [], [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._transactions = self._iFile['Transactions'].tolist()
            if 'fuzzyValues' in i:
                self._fuzzyValues = self._iFile['fuzzyValues'].tolist()
            # print(self.Database)
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
                    self._transactions.append([x for x in items])
                    self._fuzzyValues.append([x for x in quantities])
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            parts = line.split(":")
                            parts[0] = parts[0].strip()
                            parts[1] = parts[1].strip()
                            parts[2] = parts[2].strip()
                            times = parts[0].split(self._sep)
                            items = parts[1].split(self._sep)
                            quantities = parts[2].split(self._sep)
                            #print(times, items, quantities)
                            _time = [x for x in times if x]
                            items = [x for x in items if x]
                            quantities = [float(x) for x in quantities if x]
                            tempList = []
                            for k in range(len(_time)):
                                ite = "(" + _time[k] + "," + items[k] + ")"
                                tempList.append(ite)
                            self._ts.append([x for x in times])
                            self._transactions.append([x for x in tempList])
                            self._fuzzyValues.append([x for x in quantities])
                except IOError:
                    print("File Not Found")
                    quit()

    def startMine(self):
        """
          fuzzy-Frequent pattern mining process will start from here
        """
        self._startTime = _ab._time.time()
        self._creatingItemsets()
        for line in range(len(self._transactions)):
            times = self._ts[line]
            items = self._transactions[line]
            quantities = self._fuzzyValues[line]
            self._dbLen += 1
            for i in range(0, len(items)):
                item = items[i]
                if item in self._mapItemSum:
                    self._mapItemSum[item] += quantities[i]
                else:
                    self._mapItemSum[item] = quantities[i]
        listOfffilist = []
        mapItemsToFFLIST = {}
        #self._minSup = float(self._minSup)
        self._minSup = self._convert(self._minSup)
        minSup = self._minSup
        for item1 in self._mapItemSum.keys():
            item = item1
            # print(type(self._mapItemSum[item]))
            if self._mapItemSum[item] >= self._minSup:
                fuList = _FFList(item)
                mapItemsToFFLIST[item] = fuList
                listOfffilist.append(fuList)
        listOfffilist.sort(key=_ab._functools.cmp_to_key(self._compareItems))
        tid = 0
        for line in range(len(self._transactions)):
            items = self._transactions[line]
            quantities = self._fuzzyValues[line]
            revisedTransaction = []
            for i in range(0, len(items)):
                pair = _Pair()
                pair.item = items[i]
                pair.quantity = quantities[i]
                item = pair.item
                if self._mapItemSum[item] >= self._minSup:
                    if pair.quantity > 0:
                        revisedTransaction.append(pair)
            revisedTransaction.sort(key=_ab._functools.cmp_to_key(self._compareItems))
            for i in range(len(revisedTransaction) - 1, -1, -1):
                pair = revisedTransaction[i]
                if mapItemsToFFLIST.get(pair.item) is not None:
                    FFListOfItem = mapItemsToFFLIST[pair.item]
                    element = _Element(tid, pair.quantity)
                    FFListOfItem.addElement(element)
            tid += 1
        self._F3PMining(self._itemSetBuffer, 0, listOfffilist, self._minSup)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def _F3PMining(self, prefix, prefixLen, FSFIM, minSup):
        """Generates ffi from prefix

        :param prefix: the prefix patterns of ffi
        :type prefix: len
        :param prefixLen: the length of prefix
        :type prefixLen: int
        :param FSFIM: the Fuzzy list of prefix itemSets
        :type FSFIM: list
        :param minSup: the minimum support of
        :type minSup:int
        """
        for i in range(0, len(FSFIM)):
            X = FSFIM[i]
            exULs = []
            if X.sumIUtil >= minSup:
                self._WriteOut(prefix, prefixLen, X.item, X.sumIUtil)
                for j in range(i + 1, len(FSFIM)):
                    Y = FSFIM[j]
                    exULs.append(self._construct(X, Y))
                    self._joinsCnt += 1
                self._itemSetBuffer.insert(prefixLen, X.item)
                self._F3PMining(self._itemSetBuffer, prefixLen + 1, exULs, minSup)

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

    def _construct(self, px, py):
        """
            A function to construct a new Fuzzy itemSet from 2 fuzzy itemSets

            :param px:the itemSet px
            :type px:ffi-List
            :param py:itemSet py
            :type py:ffi-List
            :return :the itemSet of pxy(px and py)
            :rtype :ffi-List
        """
        pxyUL = _FFList(py.item)
        for ex in px.elements:
            ey = self._findElementWithTID(py, ex.tid)
            if ey is None:
                continue
            eXY = _Element(ex.tid, min([ex.iUtils, ey.iUtils], key=lambda x: float(x)))
            pxyUL.addElement(eXY)
        return pxyUL

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
            res += str(prefix[i]) + "\t"
        res += str(item)
        res1 = str(sumIUtil)
        self._finalPatterns[res] = res1

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

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

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

    def printResults(self):
        """ This function is used to print the results
        """
        print("Total number of Fuzzy Partial Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = F3PMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = F3PMiner(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        _ap.save(_ab._sys.argv[2])
        _ap.printResults()
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

