# FTApriori is one of the fundamental algorithm to discover fault-tolerant frequent patterns in a transactional database.
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------------------
#
#
#             from PAMI.faultTolerantFrequentPattern.basic import FTApriori as alg
#
#             obj = alg.FTApriori(inputFile,minSup,itemSup,minLength,faultTolerance)
#
#             obj.mine()
#
#             patterns = obj.getPatterns()
#
#             print("Total number of fault-tolerant frequent patterns:", len(patterns))
#
#             obj.save("outputFile")
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime
#
#             print("Total ExecutionTime in seconds:", run)
#




__copyright__ = """
Copyright (C)  2026 Rage Uday Kiran

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
     Copyright (C)  2026 Rage Uday Kiran
     
"""

from PAMI.faultTolerantFrequentPattern.basic import abstract as _ab
import pandas as pd
import numpy as _np
from typing import List, Dict, Tuple, Union
from deprecated import deprecated


class FTApriori(_ab._faultTolerantFrequentPatterns):
    """

    :Description:   FT-Apriori is one of the fundamental algorithm to discover fault-tolerant frequent patterns in a transactional database.
                    This implementation employs the downward-closure property of the fault-tolerant support (which is
                    anti-monotone for a fixed ``faultTolerance``) to prune the search space with level-wise apriori
                    candidate generation, instead of enumerating the whole power set.

    :Reference:    Pei, Jian & Tung, Anthony & Han, Jiawei. (2001). Fault-Tolerant Frequent Pattern Mining: Problems and Challenges.

    :param  iFile: str :
                   Name of the Input file to mine complete set of fault Tolerant frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of falut Tolerant frequent patterns
    :param  minSup: float or int or str :
                    The user can specify minSup either in count or proportion of database size.
                    If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                    Otherwise, it will be treated as float.
                    Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
    :param  itemSup: int or float :
                    Minimum support of an item to be considered as a candidate item of a fault-tolerant pattern
    :param minLength: int :
                     minimum length of a pattern
    :param faultTolerance: int :
                     maximum number of items of a pattern that a transaction is allowed to miss

    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    :Attributes:

        startTime : float
          To record the start time of the mining process

        endTime : float
          To record the completion time of the mining process

        finalPatterns : dict
          Storing the complete set of patterns in a dictionary variable

        memoryUSS : float
          To store the total amount of USS memory consumed by the program

        memoryRSS : float
          To store the total amount of RSS memory consumed by the program

        Database : list
          To store the transactions of a database in list


    **Methods to execute code on terminal**
    ------------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 FTApriori.py <inputFile> <outputFile> <minSup> <itemSup> <minLength> <faultTolerance>

      Example Usage:

      (.venv) $ python3 FTApriori.py sampleDB.txt patterns.txt 10.0 3.0 3 1


    .. note:: minSup will be considered in times of minSup and count of database transactions

    **Importing this algorithm into a python program**
    ----------------------------------------------------------------
    .. code-block:: python

            from PAMI.faultTolerantFrequentPattern.basic import FTApriori as alg

            obj = alg.FTApriori(inputFile,minSup,itemSup,minLength,faultTolerance)

            obj.mine()

            patterns = obj.getPatterns()

            print("Total number of fault-tolerant frequent patterns:",  len(patterns))

            obj.save("outputFile")

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:",  memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS",  memRSS)

            run = obj.getRuntime

            print("Total ExecutionTime in seconds:",  run)

    **Credits:**
    ----------------
             The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.

    """

    _minSup = float()
    _itemSup = float()
    _minLength = int()
    _faultTolerance = int()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _mapSupport = {}

    def _creatingItemSets(self) -> None:
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
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(set(temp))
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(set(temp))
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value) -> float:
        """
        To convert the user specified minSup value

        :param value: user specified minSup value

        :type value: int or float

        :return: converted type

        :rtype: float
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

    def _oneLengthFrequentItems(self) -> None:
        """
        Restricts the candidate items to those whose global support is at least ``itemSup``.
        """
        self._mapSupport = {}
        for li in self._Database:
            for i in li:
                if i not in self._mapSupport:
                    self._mapSupport[i] = 1
                else:
                    self._mapSupport[i] += 1
        self._mapSupport = {k: v for k, v in self._mapSupport.items() if v >= self._itemSup}

    def _buildMatrix(self) -> Tuple[List[str], _np.ndarray]:
        """
        Builds the boolean transaction x candidate-item matrix used to count fault-tolerant support.

        :return: the sorted list of candidate item names and the ``|D| x nItems`` 0/1 matrix.
        :rtype: tuple(list, numpy.ndarray)
        """
        items = sorted(self._mapSupport.keys())
        colOf = {item: j for j, item in enumerate(items)}
        matrix = _np.zeros((len(self._Database), len(items)), dtype=_np.int32)
        for row, transaction in enumerate(self._Database):
            for item in transaction:
                j = colOf.get(item)
                if j is not None:
                    matrix[row, j] = 1
        return items, matrix

    def _getFaultPatterns(self) -> None:
        """
        Discovers every fault-tolerant frequent pattern using level-wise apriori generation with
        downward-closure pruning. The fault-tolerant support of an itemset ``P`` is the number of
        transactions holding at least ``|P| - faultTolerance`` of its items; this measure is
        anti-monotone, so a candidate is generated only when all of its (k-1)-subsets are frequent.
        """
        items, matrix = self._buildMatrix()
        if not items:
            return
        c = self._faultTolerance
        minSup = self._minSup

        allFrequent: Dict[Tuple[int, ...], int] = {}
        # Level 1: every candidate item (already itemSup-filtered).
        level: List[Tuple[Tuple[int, ...], _np.ndarray]] = []
        for j in range(len(items)):
            presentCount = matrix[:, j]
            support = int((presentCount >= 1 - c).sum())
            if support >= minSup:
                level.append(((j,), presentCount))
                allFrequent[(j,)] = support

        while len(level) > 1:
            level.sort(key=lambda x: x[0])
            prevSet = set(t for t, _ in level)
            nextLevel: List[Tuple[Tuple[int, ...], _np.ndarray]] = []
            m = len(level)
            i = 0
            while i < m:
                j = i + 1
                prefix = level[i][0][:-1]
                while j < m and level[j][0][:-1] == prefix:
                    j += 1
                for a in range(i, j):
                    for b in range(a + 1, j):
                        candidate = level[a][0] + (level[b][0][-1],)
                        # Prune: every (k-1)-subset must be frequent.
                        if any((candidate[:idx] + candidate[idx + 1:]) not in prevSet
                               for idx in range(len(candidate))):
                            continue
                        presentCount = level[a][1] + matrix[:, candidate[-1]]
                        support = int((presentCount >= len(candidate) - c).sum())
                        if support >= minSup:
                            nextLevel.append((candidate, presentCount))
                            allFrequent[candidate] = support
                i = j
            level = nextLevel

        for indexTuple, support in allFrequent.items():
            if len(indexTuple) >= self._minLength:
                self._finalPatterns[tuple(items[col] for col in indexTuple)] = support

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        Fault-tolerant frequent pattern mining process will start from here
        """

        self.mine()

    def mine(self) -> None:
        """
        Fault-tolerant frequent pattern mining process will start from here
        """
        self._Database = []
        self._finalPatterns = {}
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._itemSup = self._convert(self._itemSup)
        self._minLength = int(self._minLength)
        self._faultTolerance = int(self._faultTolerance)
        self._oneLengthFrequentItems()

        self._getFaultPatterns()
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Fault-Tolerant Frequent patterns were generated successfully using FTApriori algorithm ")

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

    def getPatternsAsDataFrame(self) -> pd.DataFrame:
        """

        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame

        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            s = str()
            for i in a:
                s = s + i + ' '
            data.append([s, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile) -> None:
        """

        Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file

        :type outFile: csvfile

        :return: None

        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s = str()
            for i in x:
                s = s + i + '\t'
            s1 = s.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self) -> Dict[Tuple[str, ...], int]:
        """

        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict

        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        This is function is used to print the result
        """
        print("Total number of Fault-Tolerant Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 7 or len(_ab._sys.argv) == 8:
        if len(_ab._sys.argv) == 8:
            _ap = FTApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4],
                            _ab._sys.argv[5], _ab._sys.argv[6], _ab._sys.argv[7])
        if len(_ab._sys.argv) == 7:
            _ap = FTApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])

        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
