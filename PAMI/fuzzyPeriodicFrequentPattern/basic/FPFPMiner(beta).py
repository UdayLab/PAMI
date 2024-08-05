# Fuzzy Periodic Frequent Pattern Miner is desired to find all fuzzy periodic frequent patterns which is
# on-trivial and challenging problem to its huge search space.we are using efficient pruning
# techniques to reduce the search space.
#
# Sample run of importing the code:
# ----------------------------------------
#
#             from PAMI.fuzzyPeriodicFrequentPattern.basic import FPFPMiner as alg
#
#             obj =alg.FPFPMiner("input.txt",2,3)
#
#             obj.mine()
#
#             periodicFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))
#
#             obj.save("output.txt")
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
#




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


from PAMI.fuzzyPeriodicFrequentPattern.basic import abstract as _ab
from typing import List, Dict, Tuple, Set, Union, Any, Generator
from deprecated import deprecated
import numpy as np


class FPFPMiner(_ab._fuzzyPeriodicFrequentPatterns):
    """
    :Description:   Fuzzy Periodic Frequent Pattern Miner is desired to find all fuzzy periodic frequent patterns which is
                    on-trivial and challenging problem to its huge search space.we are using efficient pruning
                    techniques to reduce the search space.

    :Reference:   R. U. Kiran et al., "Discovering Fuzzy Periodic-Frequent Patterns in Quantitative Temporal Databases,"
                  2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), Glasgow, UK, 2020, pp.
                  1-8, doi: 10.1109/FUZZ48607.2020.9177579.

    :param  iFile: str :
                   Name of the Input file to mine complete set of frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns
    :param  minSup: int or float or str :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
    :param maxPer: float :
                   The user can specify maxPer in count or proportion of database size. If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    :Attributes:

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
        maxTID: int
            represent the maximum tid of the database
        lastTIDs: map
            represent the last tid of fuzzy items
        itemsToRegion: map
            represent items with respective regions

    :Methods:

        mine()
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
        FSFIMining( prefix, prefixLen, fsFim, minSup)
            Method generate FFI from prefix
        construct(px, py)
            A function to construct Fuzzy itemSet from 2 fuzzy itemSets
        findElementWithTID(UList, tid)
            To find element with same tid as given
        WriteOut(prefix, prefixLen, item, sumIUtil,period)
            To Store the patten

    **Executing the code on terminal :**
    ----------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 FPFPMiner.py <inputFile> <outputFile> <minSup> <maxPer> <sep>

      Example Usage:

      (.venv) $ python3  FPFPMiner.py sampleTDB.txt output.txt 2 3

    .. note:: minSup will be considered in percentage of database transactions


    **Sample run of importing the code:**
    --------------------------------------

        from PAMI.fuzzyPeriodicFrequentPattern.basic import FPFPMiner as alg

        obj =alg.FPFPMiner("input.txt",2,3)

        obj.mine()

        periodicFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))

        obj.save("output.txt")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    **Credits:**
    --------------
            The complete program was written by Sai Chitra.B under the supervision of Professor Rage Uday Kiran.

    """
    _startTime = float()
    _endTime = float()
    _minSup = float()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _sep = " "
    _Database = []
    _transactions = []
    _fuzzyValues = []
    _ts = []

    def __init__(self, iFile: Union[str, _ab._pd.DataFrame], minSup: Union[int, float], period: Union[int, float], sep: str="\t") -> None:
        super().__init__(iFile, minSup, period, sep)
        self._oFile = ""
        self._BufferSize = 200
        self._itemSetBuffer = []
        self._mapItemSum = {}
        self._finalPatterns = {}
        self._joinsCnt = 0
        self._itemsCnt = 0
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._dbLen = 0

    def _convert(self, value) -> float:
        """
        To convert the given user specified value

        :param value: user specified value

        :type value: int or float or str

        :return: converted value

        :rtype: float
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._dbLen * value)
        if type(value) is str:
            if '.' in value:
                value = (self._dbLen * value)
            else:
                value = int(value)
        return value

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable

        :return: None
        """
        data, self._transactions, self._fuzzyValues, ts = [], [], [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                self._ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                self._transactions = self._iFile['Transactions'].tolist()
            if 'fuzzyValues' in i:
                self._fuzzyValues = self._iFile['fuzzyValues'].tolist()
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                count = 0
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = line.split(":")
                    parts[0] = parts[0].strip()
                    parts[1] = parts[1].strip()
                    items = parts[0].split(self._sep)
                    quantities = parts[1].split(self._sep)
                    self._ts.append(int(items[0]))
                    self._transactions.append([x for x in items[1:]])
                    self._fuzzyValues.append([float(x) for x in quantities])
                    count += 1
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        count = 0
                        for line in f:
                            line = line.split("\n")[0]
                            parts = line.split(":")
                            parts[0] = parts[0].strip()
                            parts[1] = parts[1].strip()
                            items = parts[0].split(self._sep)
                            quantities = parts[1].split(self._sep)
                            self._ts.append(int(items[0]))
                            self._transactions.append([x for x in items[1:]])
                            self._fuzzyValues.append([float(x) for x in quantities])
                            count += 1
                except IOError:
                    print("File Not Found")
                    quit()

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        Fuzzy periodic Frequent pattern mining process will start from here
        """
        self.mine()
    
    def _getMaxPer(self, arr):
        arr = np.append(list(arr), [0, self._dbLen])
        arr = np.sort(arr)
        arr = np.diff(arr)

        return np.max(arr)
    
    def dfs(self, cands):
        for i in range(len(cands)):
            newCands = []
            for j in range(i + 1, len(cands)):
                newCand = tuple(cands[i] + tuple([cands[j][-1]]))
                # print(items[cands[i]], items[cands[j]])
                newCandItems = {}
                keys = self._Database[cands[i]].keys() & self._Database[cands[j]].keys()
                for k in keys:
                    newCandItems[k] = min(self._Database[cands[i]][k], self._Database[cands[j]][k])
                count = sum(newCandItems.values())
                maxPer = self._getMaxPer(list(newCandItems.keys()))
                if count >= self._minSup and maxPer <= self._maxPer:
                    newCands.append(newCand)
                    self._finalPatterns[newCand] = count
                    self._Database[newCand] = newCandItems
            if len(newCands) > 1:
                self.dfs(newCands)

    def mine(self) -> None:
        """
        Fuzzy periodic Frequent pattern mining process will start from here
        """
        maxTID = 0
        lastTIDs = {}
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._finalPatterns = {}

        items = {}

        for ts, transactions, fuzzyValues in zip(self._ts, self._transactions, self._fuzzyValues):
            for item, fuzzyValue in zip(transactions, fuzzyValues):
                item = tuple([item])
                if item not in items:
                    items[item] = {}
                items[item][ts] = fuzzyValue
            maxTID = max(maxTID, ts)

        self._dbLen = maxTID
        self._minSup = self._convert(self._minSup)

        self._Database = items.copy()


        supports = {k:[sum(v.values()),self._getMaxPer(list(v.keys()))] for k,v in items.items()}
    

        supports = {k:v for k,v in supports.items() if v[0] >= self._minSup and v[1] <= self._maxPer}
        self._Database = {k:v for k,v in items.items() if k in supports}
        self._Database = {k:v for k,v in sorted(self._Database.items(), key=lambda x: supports[x[0]][0], reverse=True)}

        self._finalPatterns = {k:v for k,v in supports.items() if k in self._Database}

        cands = list(self._Database.keys())
        self.dfs(cands)


        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss



    def getMemoryUSS(self) -> float:
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryRSS

    def getRuntime(self) -> float:
        """
        Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """
        return self._endTime - self._startTime

    def _WriteOut(self, prefix: List[int], prefixLen: int, item: int, sumLUtil: float, period: int) -> None:
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
        :return: None
        """
        self._itemsCnt += 1
        res = ""
        for i in range(0, prefixLen):
            res += str(prefix[i]) +  "\t"
        res += str(item)
        #res1 = str(sumLUtil) + " : " + str(period)
        self._finalPatterns[res] = [sumLUtil, period]

    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """
        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b[0], b[1]])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataFrame

    def getPatterns(self) -> Dict[str, str]:
        """
        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def save(self, outFile: str) -> None:
        """
        Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file
        :type outFile: csv file
        :return: None
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x.strip() + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % patternsAndSupport)

    def printResults(self) -> None:
        """
        This function is used to print the results
        """
        print("Total number of Fuzzy Periodic-Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:  # to  include a user specified separator
            _ap = FPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:  # to consider "\t" as a separator
            _ap = FPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _ap.mine()
        print("Total number of Fuzzy Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        _ap = FPFPMiner('/Users/tarunsreepada/Downloads/temporal_Fuzzy_T10I4D100K.csv', 500, 100000, '\t')
        _ap.mine()
        print("Total number of Fuzzy Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        # _ap.save('output.txt')
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
        print("Error! The number of input parameters do not match the total number of parameters provided")

